"""Резюме отзывов: корпус из БД -> LLM -> UPSERT в `summaries`.

Ключевая деталь по стоимости: перед вызовом модели считается sha256 корпуса, который
реально уйдёт в промпт. Совпал с `summaries.input_hash` — LLM не дёргаем вообще.
Поэтому в хэш входит уже обрезанный и ограниченный по количеству корпус: изменится
лимит выборки — изменится и хэш, резюме честно пересоберётся.
"""

from __future__ import annotations

import hashlib

import structlog
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Game, Review, Summary
from app.llm.client import LlmClient, LlmDisabled, LlmError
from app.llm.embeddings import genre_names
from app.llm.prompts import (
    PURPOSE_BY_KIND,
    SUMMARY_SCHEMA,
    SUMMARY_SYSTEM_BY_KIND,
    build_summary_user_prompt,
    format_review,
)

log = structlog.get_logger(__name__)

KINDS = ("critic", "user")

# Отзыв критика — обычно 2-3 абзаца, пользовательский бывает и на десять экранов.
# Полторы тысячи символов сохраняют суть и не дают одному графоману съесть контекст.
MAX_QUOTE_CHARS = 1500
MAX_ITEMS_IN_LIST = 6


def _reviews_limit(kind: str) -> int:
    return settings.mc_critic_reviews_max if kind == "critic" else settings.mc_user_reviews_max


def _corpus_hash(rows: list[tuple[str, str]]) -> str:
    payload = "\n".join(f"{source_key}\x1f{quote}" for source_key, quote in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clean_list(value: object) -> list[str]:
    """Схема гарантирует список строк, но пустые пункты и дубли всё равно чистим."""
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        result.append(text)
        if len(result) >= MAX_ITEMS_IN_LIST:
            break
    return result


async def _existing_summary(session: AsyncSession, game_id: int, kind: str) -> Summary | None:
    stmt = (
        select(Summary)
        .where(Summary.game_id == game_id, Summary.kind == kind)
        .execution_options(populate_existing=True)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def summarize_game(session: AsyncSession, game: Game, kind: str) -> Summary | None:
    """kind: 'critic' | 'user'. Возвращает актуальное резюме либо None, если его нет.

    Никогда не поднимает исключения LLM-контура: без ключа, при отказе модели или
    сетевом сбое вернётся прежнее резюме (или None), пайплайн продолжит работу.
    """
    if kind not in KINDS:
        raise ValueError(f"неизвестный kind резюме: {kind!r}")

    existing = await _existing_summary(session, game.id, kind)

    stmt = (
        select(Review.source_key, Review.quote, Review.score, Review.author, Review.publication)
        .where(
            Review.game_id == game.id,
            Review.kind == kind,
            Review.quote.is_not(None),
            func.length(func.trim(Review.quote)) > 0,
        )
        .order_by(Review.source_key)  # детерминированный порядок = стабильный хэш
        .limit(_reviews_limit(kind))
    )
    rows = (await session.execute(stmt)).all()
    if not rows:
        log.info("summary_skip_no_reviews", game_id=game.id, kind=kind)
        return existing

    corpus = [(row.source_key, (row.quote or "").strip()[:MAX_QUOTE_CHARS]) for row in rows]
    input_hash = _corpus_hash(corpus)

    if existing is not None and existing.input_hash == input_hash:
        log.info("summary_cache_hit", game_id=game.id, kind=kind, reviews=len(corpus))
        return existing

    prompt_reviews = [
        format_review(
            index=i,
            kind=kind,
            quote=quote,
            score=row.score,
            author=row.author,
            publication=row.publication,
        )
        for i, (row, (_, quote)) in enumerate(zip(rows, corpus, strict=True), start=1)
    ]
    user_prompt = build_summary_user_prompt(
        title=game.title,
        kind=kind,
        reviews=prompt_reviews,
        genres=genre_names(game.genres),
    )

    client = LlmClient(session)
    try:
        data = await client.complete_structured(
            purpose=PURPOSE_BY_KIND[kind],
            system=SUMMARY_SYSTEM_BY_KIND[kind],
            user=user_prompt,
            schema=SUMMARY_SCHEMA,
            game_id=game.id,
        )
    except LlmDisabled:
        log.info("summary_skip_llm_disabled", game_id=game.id, kind=kind)
        return existing
    except LlmError as exc:
        log.warning("summary_llm_failed", game_id=game.id, kind=kind, error=str(exc))
        return existing
    finally:
        await client.aclose()

    values = {
        "game_id": game.id,
        "kind": kind,
        "likes": _clean_list(data.get("likes")),
        "dislikes": _clean_list(data.get("dislikes")),
        "tl_dr": (str(data.get("tl_dr") or "")).strip() or None,
        "model": settings.llm_model,
        "input_hash": input_hash,
        "source_review_count": len(corpus),
        "llm_call_id": client.last_call_id,
    }
    upsert = (
        pg_insert(Summary)
        .values(**values)
        .on_conflict_do_update(
            constraint="uq_summary_kind",
            set_={
                "likes": values["likes"],
                "dislikes": values["dislikes"],
                "tl_dr": values["tl_dr"],
                "model": values["model"],
                "input_hash": values["input_hash"],
                "source_review_count": values["source_review_count"],
                "llm_call_id": values["llm_call_id"],
                "updated_at": func.now(),  # onupdate ORM-а на ON CONFLICT не срабатывает
            },
        )
    )
    await session.execute(upsert)
    await session.flush()

    log.info("summary_written", game_id=game.id, kind=kind, reviews=len(corpus))
    return await _existing_summary(session, game.id, kind)
