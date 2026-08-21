"""Веб-слой: страницы должны работать и на пустой базе, и с данными."""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio

from app.db.models import Game, GamePlatform, SimilarGame, Summary
from tests.conftest import postgres_required

pytestmark = postgres_required


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    from app.web.main import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


async def make_game(session, slug: str, title: str, metascore: int | None = 88) -> Game:
    game = Game(
        mc_id=abs(hash(slug)) % 10_000_000,
        slug=slug,
        title=title,
        description="Описание игры для теста.",
        developer="Sandfall Interactive",
        release_date=dt.date(2025, 4, 24),
        genres=[{"name": "JRPG"}],
        cover_url="https://www.metacritic.com/a/img/catalog/x.jpg",
        trailer_embed_url="https://cdn.jwplayer.com/players/abc.html",
        lead_metascore=metascore,
        lead_userscore=9.1,
    )
    session.add(game)
    await session.flush()
    session.add(
        GamePlatform(
            game_id=game.id,
            platform_name="PlayStation 5",
            platform_slug="playstation-5",
            metascore=metascore,
            metascore_review_count=86,
            userscore=9.5,
            userscore_review_count=26806,
            is_lead=True,
        )
    )
    await session.commit()
    return game


async def test_healthz(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_index_on_empty_db(client: httpx.AsyncClient, session) -> None:
    """Пустая база — это состояние «ещё не собрали», а не ошибка."""
    response = await client.get("/")
    assert response.status_code == 200
    assert "База пока пуста" in response.text


async def test_index_lists_game_and_filters(client: httpx.AsyncClient, session) -> None:
    await make_game(session, "clair-obscur", "Clair Obscur: Expedition 33")

    listing = await client.get("/")
    assert listing.status_code == 200
    assert "Clair Obscur: Expedition 33" in listing.text
    assert "PlayStation 5" in listing.text

    found = await client.get("/", params={"q": "clair"})
    assert "Clair Obscur" in found.text

    missing = await client.get("/", params={"q": "нет такой игры"})
    assert "Ничего не найдено" in missing.text

    by_platform = await client.get("/", params={"platform": "playstation-5"})
    assert "Clair Obscur" in by_platform.text

    other_platform = await client.get("/", params={"platform": "nintendo-switch"})
    assert "Ничего не найдено" in other_platform.text


async def test_sort_by_rating(client: httpx.AsyncClient, session) -> None:
    await make_game(session, "low-score", "Слабая игра", metascore=42)
    await make_game(session, "high-score", "Сильная игра", metascore=95)

    page = await client.get("/", params={"sort": "metascore"})
    assert page.status_code == 200
    assert page.text.index("Сильная игра") < page.text.index("Слабая игра")


async def test_htmx_request_returns_only_grid(client: httpx.AsyncClient, session) -> None:
    response = await client.get("/", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert '<div id="game-grid">' in response.text
    assert "<!DOCTYPE html>" not in response.text


async def test_game_card(client: httpx.AsyncClient, session) -> None:
    game = await make_game(session, "clair-obscur", "Clair Obscur: Expedition 33")
    similar = await make_game(session, "sinking-city", "The Sinking City 2", metascore=70)
    session.add(
        Summary(
            game_id=game.id,
            kind="critic",
            likes=["боевая система"],
            dislikes=["короткая концовка"],
            tl_dr="Критики в восторге от боя.",
            model="claude-haiku-4-5",
            source_review_count=86,
        )
    )
    session.add(SimilarGame(game_id=game.id, similar_id=similar.id, rank=1, score=0.87))
    await session.commit()

    response = await client.get(f"/game/{game.slug}")
    assert response.status_code == 200
    body = response.text
    assert "Sandfall Interactive" in body
    assert "PlayStation 5" in body
    assert "9.5" in body  # userscore платформы
    assert "боевая система" in body and "короткая концовка" in body
    assert "cdn.jwplayer.com" in body  # трейлер
    assert "/game/sinking-city" in body  # похожая игра кликабельна


async def test_missing_game_returns_404(client: httpx.AsyncClient, session) -> None:
    assert (await client.get("/game/no-such-game")).status_code == 404


async def test_monitor_page(client: httpx.AsyncClient, session) -> None:
    response = await client.get("/monitor")
    assert response.status_code == 200
    assert "Мониторинг обработки" in response.text
    assert "Запустить обработку" in response.text


async def test_stats_endpoint(client: httpx.AsyncClient, session) -> None:
    payload = (await client.get("/api/stats")).json()
    assert payload["games_total"] == 0
    assert payload["events"] == []


@pytest.mark.parametrize("headers", [{}, {"X-Admin-Token": "wrong"}])
async def test_admin_run_requires_token(client: httpx.AsyncClient, headers: dict) -> None:
    response = await client.post("/api/admin/run", headers=headers)
    assert response.status_code == 401


async def test_llm_logs_page(client: httpx.AsyncClient) -> None:
    response = await client.get("/llm-logs")
    assert response.status_code == 200
    assert "Логи переписки с нейросетью" in response.text


async def test_llm_log_path_traversal_blocked(client: httpx.AsyncClient) -> None:
    for name in ("../../.env", "..%2f..%2f.env", "secret.txt"):
        assert (await client.get(f"/llm-logs/{name}")).status_code == 404
