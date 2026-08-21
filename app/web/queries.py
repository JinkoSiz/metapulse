"""Выборки веб-слоя.

Весь SQL живёт здесь: роуты остаются тонкими, а запросы можно проверять отдельно.
Правило — ни одна функция не бросает исключение на пустой базе, вместо этого
возвращает пустые коллекции и нули.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import Select, case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import (
    Game,
    GamePlatform,
    LetsPlay,
    LlmCall,
    PipelineRun,
    Review,
    SimilarGame,
    Summary,
    TaskEvent,
)

# Допустимые сортировки списка: ключ -> подпись для UI.
SORTS: dict[str, str] = {
    "date": "Дата выхода",
    "metascore": "Metascore",
    "userscore": "Userscore",
    "title": "Название",
}
DEFAULT_SORT = "date"
DEFAULT_PER_PAGE = 24
MAX_PER_PAGE = 96

EVENTS_LIMIT = 20
RUNS_LIMIT = 10


def normalize_sort(sort: str | None) -> str:
    return sort if sort in SORTS else DEFAULT_SORT


def _order_by(sort: str) -> list[Any]:
    """NULLS LAST обязателен: игры без оценки не должны вытеснять оценённые наверх."""
    if sort == "metascore":
        return [Game.lead_metascore.desc().nullslast(), Game.title.asc(), Game.id.desc()]
    if sort == "userscore":
        return [Game.lead_userscore.desc().nullslast(), Game.title.asc(), Game.id.desc()]
    if sort == "title":
        return [Game.title.asc(), Game.id.desc()]
    return [Game.release_date.desc().nullslast(), Game.id.desc()]


def _apply_filters(stmt: Select[Any], q: str | None, platform: str | None) -> Select[Any]:
    if q:
        # ILIKE '%...%' обслуживается индексом ix_games_title_trgm (gin_trgm_ops)
        stmt = stmt.where(Game.title.ilike(f"%{q}%"))
    if platform:
        # EXISTS вместо JOIN: у игры несколько платформ, JOIN размножил бы строки
        exists_platform = (
            select(GamePlatform.id)
            .where(
                GamePlatform.game_id == Game.id,
                GamePlatform.platform_slug == platform,
            )
            .exists()
        )
        stmt = stmt.where(exists_platform)
    return stmt


async def list_games(
    session: AsyncSession,
    q: str | None = None,
    platform: str | None = None,
    sort: str = DEFAULT_SORT,
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
) -> tuple[list[Game], int]:
    """Страница каталога с фильтрами. Возвращает (игры, всего подходящих)."""
    q = (q or "").strip() or None
    platform = (platform or "").strip() or None
    sort = normalize_sort(sort)
    page = max(1, int(page or 1))
    per_page = max(1, min(int(per_page or DEFAULT_PER_PAGE), MAX_PER_PAGE))

    total = await session.scalar(
        _apply_filters(select(func.count()).select_from(Game), q, platform)
    )
    total = int(total or 0)

    stmt = (
        _apply_filters(select(Game), q, platform)
        .order_by(*_order_by(sort))
        .limit(per_page)
        .offset((page - 1) * per_page)
    )
    games = list((await session.scalars(stmt)).unique().all())
    return games, total


async def get_game_by_slug(session: AsyncSession, slug: str) -> Game | None:
    """Карточка игры. Платформы, резюме и летсплей подтягиваются selectin-загрузкой модели."""
    if not slug:
        return None
    return await session.scalar(select(Game).where(Game.slug == slug))


async def get_similar(session: AsyncSession, game_id: int) -> list[Game]:
    stmt = (
        select(Game)
        .join(SimilarGame, SimilarGame.similar_id == Game.id)
        .where(SimilarGame.game_id == game_id)
        .order_by(SimilarGame.rank.asc())
        .limit(settings.similar_games_count)
    )
    return list((await session.scalars(stmt)).unique().all())


async def platform_facets(session: AsyncSession) -> list[dict[str, Any]]:
    """Платформы для фильтра — только те, что реально есть в БД, с числом игр."""
    stmt = (
        select(
            GamePlatform.platform_slug.label("slug"),
            func.min(GamePlatform.platform_name).label("name"),
            func.count(func.distinct(GamePlatform.game_id)).label("games"),
        )
        .group_by(GamePlatform.platform_slug)
        .order_by(desc("games"), "name")
    )
    rows = await session.execute(stmt)
    return [{"slug": r.slug, "name": r.name or r.slug, "games": int(r.games)} for r in rows]


def _event_dict(event: TaskEvent) -> dict[str, Any]:
    return {
        "id": int(event.id),
        "ts": event.ts.isoformat() if event.ts else None,
        "level": event.level,
        "worker": event.worker,
        "stage": event.stage,
        "run_id": event.run_id,
        "game_id": event.game_id,
        "message": event.message,
        "payload": event.payload,
    }


def _run_dict(run: PipelineRun) -> dict[str, Any]:
    return {
        "id": int(run.id),
        "trigger": run.trigger,
        "status": run.status,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "duration_sec": (
            int((run.finished_at - run.started_at).total_seconds())
            if run.started_at and run.finished_at
            else None
        ),
        "stats": run.stats,
        "error": run.error,
    }


async def dashboard_stats(session: AsyncSession) -> dict[str, Any]:
    """Счётчики и ленты для /monitor и /api/stats. Всё JSON-сериализуемо."""
    now = dt.datetime.now(settings.tz)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    games_total = await session.scalar(select(func.count()).select_from(Game))
    games_today = await session.scalar(
        select(func.count()).select_from(Game).where(Game.first_seen_at >= day_start)
    )
    summaries_total = await session.scalar(select(func.count()).select_from(Summary))
    reviews_total = await session.scalar(select(func.count()).select_from(Review))
    letsplays_total = await session.scalar(select(func.count()).select_from(LetsPlay))
    letsplays_ok = await session.scalar(
        select(func.count()).select_from(LetsPlay).where(LetsPlay.conclusion.is_not(None))
    )

    llm_row = (
        await session.execute(
            select(
                func.count(LlmCall.id),
                func.coalesce(func.sum(LlmCall.input_tokens), 0),
                func.coalesce(func.sum(LlmCall.output_tokens), 0),
                func.coalesce(func.sum(case((LlmCall.status != "ok", 1), else_=0)), 0),
            ).where(LlmCall.ts >= day_start)
        )
    ).one()
    llm_calls_today, llm_in_tokens, llm_out_tokens, llm_errors_today = (
        int(llm_row[0] or 0),
        int(llm_row[1] or 0),
        int(llm_row[2] or 0),
        int(llm_row[3] or 0),
    )

    events = list(
        (
            await session.scalars(
                select(TaskEvent).order_by(TaskEvent.id.desc()).limit(EVENTS_LIMIT)
            )
        ).all()
    )
    runs = list(
        (
            await session.scalars(
                select(PipelineRun).order_by(PipelineRun.id.desc()).limit(RUNS_LIMIT)
            )
        ).all()
    )

    return {
        "ts": now.isoformat(),
        "games_total": int(games_total or 0),
        "games_today": int(games_today or 0),
        "summaries_total": int(summaries_total or 0),
        "reviews_total": int(reviews_total or 0),
        "letsplays_total": int(letsplays_total or 0),
        "letsplays_with_conclusion": int(letsplays_ok or 0),
        "llm_calls_today": llm_calls_today,
        "llm_input_tokens_today": llm_in_tokens,
        "llm_output_tokens_today": llm_out_tokens,
        "llm_tokens_today": llm_in_tokens + llm_out_tokens,
        "llm_errors_today": llm_errors_today,
        "events": [_event_dict(e) for e in events],
        "runs": [_run_dict(r) for r in runs],
    }
