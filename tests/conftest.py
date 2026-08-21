"""Общие фикстуры тестов.

Тесты парсеров, логирования и выбора летсплея работают офлайн. Тесты дневной выборки
и веб-слоя требуют локального Postgres из docker-compose и пропускаются, если его нет.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import pytest_asyncio

FIXTURES = Path(__file__).parent / "fixtures"

# Тесты чистят таблицы, поэтому работают в собственной базе. Подменяем URL до первого
# импорта app.*: иначе движок успеет создаться на рабочей базе и прогон тестов сотрёт
# собранные игры — в том числе на сервере, если кто-то запустит pytest там.
_MAIN_DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql+asyncpg://metapulse:metapulse@localhost:5432/metapulse"
)
TEST_DB_NAME = "metapulse_test"
TEST_DB_URL = _MAIN_DB_URL.rsplit("/", 1)[0] + f"/{TEST_DB_NAME}"
os.environ["DATABASE_URL"] = TEST_DB_URL

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
    url = urlparse(TEST_DB_URL.replace("postgresql+asyncpg", "postgresql"))
    try:
        with socket.create_connection((url.hostname or "localhost", url.port or 5432), timeout=1.5):
            return True
    except OSError:
        return False


postgres_required = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="нужен Postgres из docker-compose (docker compose up -d postgres)",
)


@pytest.fixture(scope="session", autouse=True)
def _prepare_test_database() -> None:
    """Создаёт тестовую базу и схему в ней. Рабочая база не затрагивается."""
    if not _postgres_reachable():
        return

    import asyncio

    import asyncpg

    from app.db.models import Base

    dsn = TEST_DB_URL.replace("postgresql+asyncpg://", "postgresql://")
    admin_dsn = dsn.rsplit("/", 1)[0] + "/postgres"

    async def prepare() -> None:
        admin = await asyncpg.connect(admin_dsn)
        try:
            exists = await admin.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", TEST_DB_NAME
            )
            if not exists:
                await admin.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
        finally:
            await admin.close()

        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool

        engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
        try:
            async with engine.begin() as conn:
                from sqlalchemy import text

                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                await conn.run_sync(Base.metadata.create_all)
        finally:
            await engine.dispose()

    asyncio.run(prepare())


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

    truncate = text(f"TRUNCATE {', '.join(TRUNCATE_TABLES)} RESTART IDENTITY CASCADE")
    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
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
