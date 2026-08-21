"""Резюме отзывов: сборка промпта, запись и пропуск повторной генерации.

Ключа Anthropic в тестах нет, поэтому подменяется единственный метод, ходящий в сеть;
всё остальное — настоящее: реальные тексты отзывов из фикстур и реальная БД.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select

from app.db.models import Review, Summary
from app.llm.client import LlmClient
from app.llm.summarize import summarize_game
from app.metacritic.parsers import parse_critic_review, parse_game_detail, parse_user_review
from app.pipeline.repository import upsert_game, upsert_reviews
from tests.conftest import load_fixture, postgres_required

pytestmark = postgres_required

ANSWER = {
    "likes": ["боевая система", "музыка и арт-дирекция"],
    "dislikes": ["короткая концовка"],
    "tl_dr": "Критики хвалят бой и стиль, ругают финал.",
}


class Recorder:
    """Ловит вызовы LLM и подсовывает готовый ответ."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return dict(ANSWER)


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(LlmClient, "complete_structured", lambda self, **kw: recorder(**kw))
    return recorder


async def seed_game(session) -> Any:
    """Настоящая игра с настоящими отзывами из сохранённых ответов API."""
    detail = parse_game_detail(load_fixture("game_clair-obscur-expedition-33.json"))
    game = await upsert_game(session, detail)

    critic = [
        parse_critic_review(raw) for raw in load_fixture("reviews_critic.json")["data"]["items"]
    ]
    user = [parse_user_review(raw) for raw in load_fixture("reviews_user.json")["data"]["items"]]
    await upsert_reviews(session, game, [*critic, *user])
    await session.commit()
    return game


async def test_summary_is_written_from_real_reviews(session, llm: Recorder) -> None:
    game = await seed_game(session)

    summary = await summarize_game(session, game, "critic")
    await session.commit()

    assert summary is not None
    assert summary.likes == ANSWER["likes"]
    assert summary.dislikes == ANSWER["dislikes"]
    assert summary.tl_dr == ANSWER["tl_dr"]
    assert summary.input_hash
    assert summary.source_review_count == 10

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["purpose"] == "critic_summary"
    # В промпт уходят настоящие цитаты, а не пересказ
    assert "Clair Obscur" in call["user"] or "4P.de" in call["user"]
    assert call["schema"]["properties"].keys() >= {"likes", "dislikes", "tl_dr"}


async def test_critic_and_user_summaries_are_separate(session, llm: Recorder) -> None:
    """Задание требует раздельные резюме — они не должны перетирать друг друга."""
    game = await seed_game(session)

    await summarize_game(session, game, "critic")
    await summarize_game(session, game, "user")
    await session.commit()

    kinds = set(
        (await session.scalars(select(Summary.kind).where(Summary.game_id == game.id))).all()
    )
    assert kinds == {"critic", "user"}
    assert {c["purpose"] for c in llm.calls} == {"critic_summary", "user_summary"}


async def test_unchanged_reviews_skip_the_model(session, llm: Recorder) -> None:
    """Cost-control: повторный обход без новых отзывов не должен жечь токены."""
    game = await seed_game(session)

    await summarize_game(session, game, "critic")
    await session.commit()
    assert len(llm.calls) == 1

    again = await summarize_game(session, game, "critic")
    await session.commit()

    assert len(llm.calls) == 1  # второго обращения к модели не было
    assert again is not None and again.likes == ANSWER["likes"]


async def test_new_reviews_trigger_regeneration(session, llm: Recorder) -> None:
    """А вот появление новых отзывов обязано обновить резюме."""
    game = await seed_game(session)
    first = await summarize_game(session, game, "critic")
    await session.commit()
    first_hash = first.input_hash

    session.add(
        Review(
            game_id=game.id,
            kind="critic",
            source_key="new-publication:Автор:2026-08-21",
            publication="Новое издание",
            score=95,
            quote="Свежий отзыв, которого не было при первой генерации.",
        )
    )
    await session.commit()

    second = await summarize_game(session, game, "critic")
    await session.commit()

    assert len(llm.calls) == 2
    assert second.input_hash != first_hash
    assert second.source_review_count == 11


async def test_no_reviews_means_no_call(session, llm: Recorder) -> None:
    detail = parse_game_detail(load_fixture("game_the-sinking-city-2.json"))
    game = await upsert_game(session, detail)
    await session.commit()

    assert await summarize_game(session, game, "user") is None
    assert llm.calls == []
