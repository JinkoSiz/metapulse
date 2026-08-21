"""Дневная выборка игр: «раз в час берём 20 игр, которых сегодня ещё не трогали».

Состояние живёт в таблице `crawl_state` и привязано к дате. Отсутствие строки на дату —
это и есть «каждый новый день начинаем заново»: первый заход дня идёт в New Releases,
все последующие — постранично по browse-выдаче, отсортированной по дате релиза.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CrawlState, DailySeen

if TYPE_CHECKING:  # клиент Metacritic нужен только для типов — избегаем циклов импорта
    from app.metacritic.client import MetacriticClient
    from app.metacritic.schemas import FinderItem

log = structlog.get_logger(__name__)

PHASE_CAROUSEL = "carousel"
PHASE_BROWSE = "browse"

BROWSE_PAGE_LIMIT = 50  # finder отдаёт максимум 50 за страницу: 60+ отвечает HTTP 400
MAX_PAGES_PER_CALL = 10  # страховка от бесконечного листания, если выдача целиком «виденная»


@dataclass
class Selection:
    """Результат выборки: сами игры плюс состояние, в котором остался день."""

    items: list[FinderItem] = field(default_factory=list)
    phase: str = PHASE_CAROUSEL  # откуда взяты игры этого захода
    next_phase: str = PHASE_CAROUSEL  # с чего начнётся следующий заход сегодня
    next_offset: int = 0
    pages_scanned: int = 0


async def select_batch(
    session: AsyncSession,
    client: MetacriticClient,
    day: dt.date,
    size: int = 20,
) -> Selection:
    """Отобрать до `size` игр, которых сегодня ещё не обрабатывали.

    В `daily_seen` функция НЕ пишет: отметку ставит вызывающий код после успешной
    обработки игры, иначе упавший прогон «съел» бы игры до следующего дня.
    """
    state = await _get_or_create_state(session, day)
    source_phase = state.phase

    if state.phase == PHASE_CAROUSEL:
        selection = await _select_from_carousel(session, client, day, size)
        # Карусель отдаётся ровно один раз за день, даже если игр в ней было меньше size:
        # повторный заход дал бы тот же список, поэтому сразу переключаемся на browse.
        state.phase = PHASE_BROWSE
    else:
        selection = await _select_from_browse(session, client, day, size, state.next_offset)
        state.next_offset = selection.next_offset

    state.runs_count += 1
    # В отчёте важна фаза, ИЗ которой взяты игры, а не та, что осталась на следующий заход:
    # иначе первый обход дня рапортует «browse», хотя игры пришли из карусели
    selection.phase = source_phase
    selection.next_phase = state.phase
    selection.next_offset = state.next_offset
    await session.flush()

    log.info(
        "selection.batch",
        day=str(day),
        picked=len(selection.items),
        phase=selection.phase,
        next_phase=selection.next_phase,
        next_offset=selection.next_offset,
        pages=selection.pages_scanned,
    )
    return selection


async def _get_or_create_state(session: AsyncSession, day: dt.date) -> CrawlState:
    """Состояние дня; на новую дату создаётся заново в фазе carousel."""
    state = await session.get(CrawlState, day)
    if state is not None:
        return state

    # ON CONFLICT DO NOTHING: два воркера могут стартовать в одну секунду на границе суток
    await session.execute(
        pg_insert(CrawlState)
        .values(day=day, phase=PHASE_CAROUSEL, next_offset=0, processed_count=0, runs_count=0)
        .on_conflict_do_nothing(index_elements=[CrawlState.day])
    )
    await session.flush()
    state = await session.get(CrawlState, day)
    if state is None:  # pragma: no cover — возможно только при гонке с удалением строки
        raise RuntimeError(f"не удалось создать crawl_state на {day}")
    return state


async def _select_from_carousel(
    session: AsyncSession,
    client: MetacriticClient,
    day: dt.date,
    size: int,
) -> Selection:
    items = await client.list_new_releases(size)
    seen = await _seen_mc_ids(session, day, (item.mc_id for item in items))
    picked, _ = _take_unseen(items, seen, size)
    return Selection(items=picked, pages_scanned=1)


async def _select_from_browse(
    session: AsyncSession,
    client: MetacriticClient,
    day: dt.date,
    size: int,
    start_offset: int,
) -> Selection:
    picked: list[FinderItem] = []
    offset = start_offset
    pages = 0

    while len(picked) < size and pages < MAX_PAGES_PER_CALL:
        items, total = await client.list_browse(offset, BROWSE_PAGE_LIMIT)
        pages += 1
        if not items:
            break

        seen = await _seen_mc_ids(session, day, (item.mc_id for item in items))
        already = {item.mc_id for item in picked}
        fresh, consumed = _take_unseen(items, seen | already, size - len(picked))
        picked.extend(fresh)
        # Сдвигаем offset только на просмотренную часть страницы: если size набрался
        # в середине, остаток страницы достанется следующему заходу, а не потеряется.
        offset += consumed

        if total and offset >= total:
            break

    return Selection(items=picked, next_offset=offset, pages_scanned=pages)


def _take_unseen(
    items: Sequence[FinderItem], seen: set[int], limit: int
) -> tuple[list[FinderItem], int]:
    """Вернуть до `limit` невиданных игр и число фактически просмотренных позиций.

    Дедуп внутри страницы обязателен: игры с одинаковой датой релиза переупорядочиваются
    между запросами, поэтому один и тот же mc_id встречается на соседних страницах.
    """
    picked: list[FinderItem] = []
    taken_ids = set(seen)
    consumed = 0
    for index, item in enumerate(items, start=1):
        consumed = index
        if item.mc_id in taken_ids:
            continue
        picked.append(item)
        taken_ids.add(item.mc_id)
        if len(picked) >= limit:
            break
    return picked, consumed


async def _seen_mc_ids(session: AsyncSession, day: dt.date, mc_ids: Iterable[int]) -> set[int]:
    """Анти-join по daily_seen — только по кандидатам текущей страницы."""
    candidates = list({mc_id for mc_id in mc_ids})
    if not candidates:
        return set()
    rows = await session.execute(
        select(DailySeen.mc_id).where(DailySeen.day == day, DailySeen.mc_id.in_(candidates))
    )
    return set(rows.scalars().all())


async def mark_seen(
    session: AsyncSession,
    day: dt.date,
    mc_id: int,
    *,
    game_id: int | None = None,
    run_id: int | None = None,
) -> None:
    """Отметить игру обработанной за день (идемпотентно)."""
    await session.execute(
        pg_insert(DailySeen)
        .values(day=day, mc_id=mc_id, game_id=game_id, run_id=run_id)
        .on_conflict_do_nothing(index_elements=[DailySeen.day, DailySeen.mc_id])
    )


async def bump_processed(session: AsyncSession, day: dt.date, count: int) -> None:
    """Счётчик обработанных за день — для мониторинга."""
    state = await session.get(CrawlState, day)
    if state is not None:
        state.processed_count += count
