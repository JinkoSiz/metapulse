"""Пакетный расчёт эмбеддингов."""

from __future__ import annotations

from typing import Any

import pytest

from app.db.models import Game
from app.llm.client import LlmClient
from app.llm.embeddings import embed_games

DIM = 1024


class Recorder:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def __call__(self, texts: list[str], **_: Any) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[0.01 * (i + 1)] * DIM for i in range(len(texts))]


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(LlmClient, "embed", lambda self, texts, **kw: recorder(texts, **kw))
    return recorder


def game(idx: int, title: str) -> Game:
    return Game(id=idx, mc_id=idx, slug=f"game-{idx}", title=title, description="Описание")


async def test_whole_batch_goes_in_one_call(embedder: Recorder) -> None:
    """Один вызов на батч: локальная модель иначе гоняет веса туда-обратно."""
    games = [game(i, f"Игра {i}") for i in range(1, 6)]

    updated = await embed_games(None, games)

    assert updated == 5
    assert len(embedder.batches) == 1
    assert len(embedder.batches[0]) == 5
    assert all(g.embedding is not None and len(g.embedding) == DIM for g in games)
    assert all(g.embedding_hash for g in games)


async def test_unchanged_games_are_skipped(embedder: Recorder) -> None:
    """Повторный обход не должен пересчитывать то, что не менялось."""
    games = [game(1, "Игра"), game(2, "Другая")]
    await embed_games(None, games)
    assert len(embedder.batches) == 1

    await embed_games(None, games)
    assert len(embedder.batches) == 1  # второго обращения не было

    games[0].title = "Игра, переименованная"
    await embed_games(None, games)
    assert len(embedder.batches) == 2
    assert len(embedder.batches[1]) == 1  # ушла только изменившаяся


async def test_nothing_to_do_means_no_call(embedder: Recorder) -> None:
    assert await embed_games(None, []) == 0
    assert await embed_games(None, [Game(id=9, mc_id=9, slug="s", title="")]) == 0
    assert embedder.batches == []
