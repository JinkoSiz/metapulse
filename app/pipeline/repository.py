"""Идемпотентные UPSERT'ы для пайплайна.

Вынесено из tasks.py, чтобы задача осталась тонкой оркестрацией, а вся работа с
конфликтами уникальных ключей жила в одном месте. Все записи безопасно повторять:
обход может упасть на любой игре и быть перезапущен через час.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Game, GamePlatform, Review

if TYPE_CHECKING:
    from app.metacritic.schemas import GameDetail, PlatformInfo, ReviewItem, ScoreStats

log = structlog.get_logger(__name__)


def _as_int(value: float | int | None) -> int | None:
    return None if value is None else int(round(value))


def _as_numeric(value: float | int | None) -> Decimal | None:
    """asyncpg строг к типам: в Numeric-колонку должен приходить Decimal, не float."""
    return None if value is None else Decimal(str(round(float(value), 1)))


def _rated(stats: ScoreStats | None) -> ScoreStats | None:
    """Оценка есть только при непустом счётчике отзывов.

    У свежих релизов Metacritic отдаёт `score: 0, reviewCount: 0` — это «ещё никто не
    оценил», а не ноль баллов. Записанный как есть, такой ноль выглядел бы в карточке
    провальной оценкой и портил бы сортировку по рейтингу.
    """
    if stats is None or not stats.review_count:
        return None
    return stats


async def game_id_by_mc_id(session: AsyncSession, mc_id: int) -> int | None:
    """Есть ли игра в базе до апсерта — так задача отличает new от updated."""
    return await session.scalar(select(Game.id).where(Game.mc_id == mc_id))


async def upsert_game(session: AsyncSession, detail: GameDetail) -> Game:
    """Создать или обновить игру по mc_id и вернуть ORM-объект."""
    lead = next((p for p in detail.platforms if p.is_lead), None)
    lead_metascore = detail.lead_metascore or None  # ноль здесь означает «оценок ещё нет»
    if lead_metascore is None and lead is not None:
        lead_critic = _rated(lead.metascore)
        if lead_critic is not None:
            lead_metascore = _as_int(lead_critic.score)

    values: dict[str, Any] = {
        "mc_id": detail.mc_id,
        "slug": detail.slug,
        "title": detail.title,
        "description": detail.description,
        "developer": detail.developer,
        "publisher": detail.publisher,
        "release_date": detail.release_date,
        "esrb_rating": detail.esrb_rating,
        "genres": list(detail.genres or []),
        "cover_url": detail.cover_url,
        "trailer_embed_url": detail.trailer_embed_url,
        "trailer_title": detail.trailer_title,
        "lead_metascore": lead_metascore,
        "last_scraped_at": dt.datetime.now(dt.UTC),
    }

    stmt = pg_insert(Game).values(**values)
    # onupdate=func.now() не срабатывает на ON CONFLICT DO UPDATE — ставим время явно
    update_set = {key: stmt.excluded[key] for key in values if key != "mc_id"}
    update_set["updated_at"] = func.now()
    stmt = stmt.on_conflict_do_update(index_elements=[Game.mc_id], set_=update_set).returning(
        Game.id
    )

    game_id = await session.scalar(stmt)
    game = await session.get(Game, game_id, populate_existing=True)
    if game is None:  # pragma: no cover — строку только что вернул RETURNING
        raise RuntimeError(f"игра {detail.slug} исчезла сразу после апсерта")
    return game


async def upsert_platforms(
    session: AsyncSession,
    game: Game,
    platforms: list[PlatformInfo],
    userscores: dict[str, ScoreStats | None] | None = None,
) -> int:
    """Записать платформы игры и денормализовать скоры lead-платформы в games.

    `userscores` — по одному ответу stats-эндпоинта на платформу: userscore берётся
    только оттуда, у деталки его нет.
    """
    userscores = userscores or {}
    written = 0

    for platform in platforms:
        stats = _rated(userscores.get(platform.slug))
        critic = _rated(platform.metascore)
        values: dict[str, Any] = {
            "game_id": game.id,
            "platform_mc_id": platform.mc_id,
            "platform_name": platform.name,
            "platform_slug": platform.slug,
            "metascore": _as_int(critic.score) if critic else None,
            "metascore_review_count": (
                platform.metascore.review_count if platform.metascore else None
            ),
            "metascore_sentiment": critic.sentiment if critic else None,
            "userscore": _as_numeric(stats.score) if stats else None,
            "userscore_review_count": stats.review_count if stats else None,
            "userscore_sentiment": stats.sentiment if stats else None,
            "is_lead": platform.is_lead,
            "platform_release_date": platform.release_date,
        }
        stmt = pg_insert(GamePlatform).values(**values)
        update_set = {
            key: stmt.excluded[key] for key in values if key not in ("game_id", "platform_slug")
        }
        update_set["updated_at"] = func.now()
        await session.execute(
            stmt.on_conflict_do_update(constraint="uq_game_platform", set_=update_set)
        )
        written += 1

    lead = next((p for p in platforms if p.is_lead), None) or (platforms[0] if platforms else None)
    if lead is not None:
        lead_critic = _rated(lead.metascore)
        if lead_critic is not None and lead_critic.score is not None:
            game.lead_metascore = _as_int(lead_critic.score)
        lead_stats = _rated(userscores.get(lead.slug))
        if lead_stats is not None and lead_stats.score is not None:
            game.lead_userscore = _as_numeric(lead_stats.score)

    await session.flush()
    return written


async def upsert_reviews(session: AsyncSession, game: Game, items: list[ReviewItem]) -> int:
    """Сохранить отзывы; вернуть количество ранее не виденных."""
    if not items:
        return 0

    # Дедуп на входе: без него ON CONFLICT падает на повторе ключа внутри одной команды
    unique: dict[tuple[str, str], ReviewItem] = {}
    for item in items:
        unique[(item.kind, item.source_key)] = item

    existing = set(
        (
            await session.execute(
                select(Review.kind, Review.source_key).where(
                    Review.game_id == game.id,
                    Review.source_key.in_([key for _, key in unique]),
                )
            )
        ).all()
    )

    rows = [
        {
            "game_id": game.id,
            "kind": item.kind,
            "source_key": item.source_key,
            "platform_slug": item.platform_slug,
            "author": item.author,
            "publication": item.publication,
            "score": item.score,
            "quote": item.quote,
            "review_date": item.review_date,
            "external_url": item.external_url,
            "spoiler": bool(item.spoiler),
        }
        for item in unique.values()
    ]

    stmt = pg_insert(Review).values(rows)
    update_set = {
        key: stmt.excluded[key] for key in rows[0] if key not in ("game_id", "kind", "source_key")
    }
    await session.execute(
        stmt.on_conflict_do_update(constraint="uq_review_source", set_=update_set)
    )
    await session.flush()

    new_count = len([key for key in unique if key not in existing])
    log.debug("reviews.upsert", game=game.slug, total=len(rows), new=new_count)
    return new_count
