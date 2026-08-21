"""Дневная выборка: «20 игр, которые сегодня не обрабатывали; каждый новый день заново».

Сети нет — клиент Metacritic подменён фейком с предсказуемой выдачей. БД настоящая:
логика построена на анти-join'е и состоянии дня, на моках это не проверить.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select

from app.db.models import CrawlState, DailySeen
from app.metacritic.schemas import FinderItem
from app.pipeline.selection import mark_seen, select_batch
from tests.conftest import postgres_required

pytestmark = postgres_required

TODAY = dt.date(2026, 8, 21)
TOMORROW = dt.date(2026, 8, 22)


def make_items(prefix: str, start: int, count: int) -> list[FinderItem]:
    return [
        FinderItem(
            mc_id=start + i,
            slug=f"{prefix}-{start + i}",
            title=f"Game {start + i}",
            release_date=TODAY,
        )
        for i in range(count)
    ]


class FakeClient:
    """Карусель отдаёт свой список, browse — страницы по 50 из общего пула."""

    def __init__(self, carousel: list[FinderItem], browse: list[FinderItem]) -> None:
        self.carousel = carousel
        self.browse = browse
        self.browse_calls: list[int] = []

    async def list_new_releases(self, limit: int = 20) -> list[FinderItem]:
        return self.carousel[:limit]

    async def list_browse(self, offset: int, limit: int = 50) -> tuple[list[FinderItem], int]:
        self.browse_calls.append(offset)
        return self.browse[offset : offset + limit], len(self.browse)


async def seen_ids(session, day: dt.date) -> set[int]:
    rows = await session.scalars(select(DailySeen.mc_id).where(DailySeen.day == day))
    return set(rows.all())


async def test_first_run_of_day_uses_carousel_then_switches_to_browse(session) -> None:
    client = FakeClient(make_items("carousel", 100, 20), make_items("browse", 500, 200))

    selection = await select_batch(session, client, TODAY, size=20)
    await session.commit()

    assert len(selection.items) == 20
    assert all(item.slug.startswith("carousel-") for item in selection.items)
    assert client.browse_calls == []  # первый заход дня в browse не ходит
    # Отчитываемся источником игр, а не тем, что осталось на следующий заход
    assert selection.phase == "carousel"
    assert selection.next_phase == "browse"

    state = await session.get(CrawlState, TODAY)
    # Карусель за день отдаётся один раз: повторный заход дал бы тот же список
    assert state.phase == "browse"
    assert state.runs_count == 1


async def test_second_run_of_day_goes_to_browse_and_skips_seen(session) -> None:
    client = FakeClient(make_items("carousel", 100, 20), make_items("browse", 500, 200))

    first = await select_batch(session, client, TODAY, size=20)
    for item in first.items:
        await mark_seen(session, TODAY, item.mc_id)
    await session.commit()

    second = await select_batch(session, client, TODAY, size=20)
    await session.commit()

    assert len(second.items) == 20
    assert all(item.slug.startswith("browse-") for item in second.items)
    assert client.browse_calls == [0]
    # Ни одна игра первого захода не попала во второй
    assert not {item.mc_id for item in second.items} & {item.mc_id for item in first.items}

    state = await session.get(CrawlState, TODAY)
    # Смещение сдвигается на просмотренные позиции, а не на всю страницу: остаток
    # страницы достанется следующему заходу, а не потеряется навсегда
    assert state.next_offset == 20
    assert state.runs_count == 2


async def test_rest_of_page_is_not_lost(session) -> None:
    """Заход берёт 20 игр из 50-элементной страницы — остальные 30 обязаны дождаться."""
    browse = make_items("browse", 500, 200)
    client = FakeClient([], browse)
    session.add(CrawlState(day=TODAY, phase="browse", next_offset=0))
    await session.commit()

    first = await select_batch(session, client, TODAY, size=20)
    for item in first.items:
        await mark_seen(session, TODAY, item.mc_id)
    await session.commit()

    second = await select_batch(session, client, TODAY, size=20)
    await session.commit()

    assert [item.mc_id for item in first.items] == [item.mc_id for item in browse[:20]]
    assert [item.mc_id for item in second.items] == [item.mc_id for item in browse[20:40]]


async def test_already_seen_page_forces_next_page(session) -> None:
    """Половина страницы уже обработана — недостающие игры берутся со следующей."""
    browse = make_items("browse", 500, 200)
    client = FakeClient([], browse)

    # день уже в фазе browse, и первые 40 позиций выдачи отмечены как обработанные
    session.add(CrawlState(day=TODAY, phase="browse", next_offset=0))
    for item in browse[:40]:
        await mark_seen(session, TODAY, item.mc_id)
    await session.commit()

    selection = await select_batch(session, client, TODAY, size=20)
    await session.commit()

    assert len(selection.items) == 20
    already = await seen_ids(session, TODAY)
    assert not {item.mc_id for item in selection.items} & already
    assert client.browse_calls == [0, 50]  # первой страницы не хватило


async def test_new_day_starts_from_scratch(session) -> None:
    client = FakeClient(make_items("carousel", 100, 20), make_items("browse", 500, 200))

    first = await select_batch(session, client, TODAY, size=20)
    for item in first.items:
        await mark_seen(session, TODAY, item.mc_id)
    await select_batch(session, client, TODAY, size=20)
    await session.commit()

    # Наступил новый день: строки состояния на него нет, значит выборка начинается заново
    tomorrow = await select_batch(session, client, TOMORROW, size=20)
    await session.commit()

    assert all(item.slug.startswith("carousel-") for item in tomorrow.items)
    assert {item.mc_id for item in tomorrow.items} == {item.mc_id for item in first.items}

    state = await session.get(CrawlState, TOMORROW)
    assert state.phase == "browse"
    assert state.next_offset == 0
    assert (await session.get(CrawlState, TODAY)).next_offset == 20  # вчерашний прогресс цел


async def test_exhausted_listing_returns_what_it_found(session) -> None:
    """Выдача кончилась — возвращаем меньше запрошенного, а не зацикливаемся."""
    browse = make_items("browse", 500, 30)
    client = FakeClient([], browse)
    session.add(CrawlState(day=TODAY, phase="browse", next_offset=0))
    await session.commit()

    selection = await select_batch(session, client, TODAY, size=20)
    await session.commit()

    assert len(selection.items) == 20
    assert len(client.browse_calls) <= 2


async def test_mark_seen_is_idempotent(session) -> None:
    await mark_seen(session, TODAY, 12345)
    await mark_seen(session, TODAY, 12345)
    await session.commit()

    assert await seen_ids(session, TODAY) == {12345}


@pytest.mark.parametrize("size", [1, 5, 20])
async def test_batch_size_is_respected(session, size: int) -> None:
    client = FakeClient(make_items("carousel", 100, 50), [])
    selection = await select_batch(session, client, TODAY, size=size)
    await session.commit()
    assert len(selection.items) == size
