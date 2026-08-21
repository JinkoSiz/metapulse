"""Запись игр и платформ в БД."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import select

from app.db.models import Game, GamePlatform
from app.metacritic.parsers import parse_game_detail
from app.metacritic.schemas import GameDetail, PlatformInfo, ScoreStats
from app.pipeline.repository import upsert_game, upsert_platforms
from tests.conftest import load_fixture, postgres_required

pytestmark = postgres_required


def detail_with(platforms: list[PlatformInfo]) -> GameDetail:
    return GameDetail(
        mc_id=999_001,
        slug="test-game",
        title="Test Game",
        description="Описание",
        developer="Studio",
        release_date=dt.date(2026, 8, 20),
        platforms=platforms,
    )


async def test_unrated_platform_stores_null_not_zero(session) -> None:
    """У свежих релизов API отдаёт score 0 при нуле отзывов — это «нет оценки».

    Записанный как есть, такой ноль выглядел бы в карточке провальной оценкой
    и утаскивал бы игру вниз сортировки по рейтингу.
    """
    detail = detail_with(
        [
            PlatformInfo(
                name="PC",
                slug="pc",
                is_lead=True,
                metascore=ScoreStats(score=0, review_count=0, sentiment="tbd"),
            )
        ]
    )
    game = await upsert_game(session, detail)
    await upsert_platforms(
        session, game, detail.platforms, {"pc": ScoreStats(score=0, review_count=0)}
    )
    await session.commit()

    platform = await session.scalar(select(GamePlatform).where(GamePlatform.game_id == game.id))
    assert platform.metascore is None
    assert platform.userscore is None
    assert game.lead_metascore is None
    assert game.lead_userscore is None


async def test_real_scores_are_stored_per_platform(session) -> None:
    """Каждая платформа хранит свою пару оценок — это отдельное требование задания."""
    detail = parse_game_detail(load_fixture("game_clair-obscur-expedition-33.json"))
    game = await upsert_game(session, detail)
    await upsert_platforms(
        session,
        game,
        detail.platforms,
        {
            "pc": ScoreStats(score=9.6, review_count=6109),
            "playstation-5": ScoreStats(score=9.5, review_count=26806),
            "xbox-series-x": ScoreStats(score=9.6, review_count=1805),
        },
    )
    await session.commit()

    rows = (
        await session.scalars(select(GamePlatform).where(GamePlatform.game_id == game.id))
    ).all()
    scores = {r.platform_slug: (r.metascore, float(r.userscore)) for r in rows}
    assert scores == {
        "playstation-5": (92, 9.5),
        "pc": (91, 9.6),
        "xbox-series-x": (91, 9.6),
    }
    # В games денормализуются оценки основной платформы — по ним сортируется список
    assert game.lead_metascore == 92
    assert float(game.lead_userscore) == 9.5


async def test_upsert_updates_instead_of_duplicating(session) -> None:
    detail = parse_game_detail(load_fixture("game_clair-obscur-expedition-33.json"))
    first = await upsert_game(session, detail)
    await session.commit()

    updated = detail.model_copy(update={"title": "Clair Obscur (обновлено)", "developer": "Other"})
    second = await upsert_game(session, updated)
    await session.commit()

    assert first.id == second.id
    assert second.title == "Clair Obscur (обновлено)"
    assert (await session.scalar(select(Game.id).where(Game.mc_id == detail.mc_id))) == first.id
