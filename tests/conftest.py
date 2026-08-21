"""Общие фикстуры тестов.

Тесты парсеров, логирования и выбора летсплея работают офлайн. Тесты дневной выборки
и веб-слоя требуют локального Postgres из docker-compose и пропускаются, если его нет.
"""

from __future__ import annotations

import json
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import pytest_asyncio

FIXTURES = Path(__file__).parent / "fixtures"

# Таблицы, которые чистятся между тестами БД (в порядке, безопасном для внешних ключей).
TRUNCATE_TABLES = (
    "task_events",
    "daily_seen",
    "similar_games",
    "llm_calls",
    "letsplays",
    "summaries",
    "reviews",
    "game_platforms",
    "pipeline_runs",
    "crawl_state",
    "games",
)


def load_fixture(name: str) -> dict[str, Any]:
    """Реальный ответ Metacritic, скачанный при разработке."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def fixtures() -> dict[str, dict[str, Any]]:
    return {path.stem: load_fixture(path.name) for path in FIXTURES.glob("*.json")}


def _postgres_reachable() -> bool:
    from app.config import settings

    url = urlparse(settings.database_url.replace("postgresql+asyncpg", "postgresql"))
    try:
        with socket.create_connection((url.hostname or "localhost", url.port or 5432), timeout=1.5):
            return True
    except OSError:
        return False


postgres_required = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="нужен Postgres из docker-compose (docker compose up -d postgres)",
)


@pytest_asyncio.fixture(autouse=True)
async def _dispose_engine_between_tests() -> AsyncIterator[None]:
    """Пул соединений нельзя переносить между тестами.

    pytest-asyncio закрывает event loop после каждого теста, а asyncpg-соединения в
    пуле остаются привязанными к нему — следующий тест падает на «Event loop is closed».
    """
    yield
    from app.db.session import engine

    await engine.dispose()


@pytest_asyncio.fixture
async def session() -> AsyncIterator[Any]:
    """Чистая сессия к локальной БД на собственном движке без пула."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from app.config import settings

    truncate = text(f"TRUNCATE {', '.join(TRUNCATE_TABLES)} RESTART IDENTITY CASCADE")
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as db:
            await db.execute(truncate)
            await db.commit()
            try:
                yield db
            finally:
                await db.rollback()
                await db.execute(truncate)
                await db.commit()
    finally:
        await engine.dispose()
