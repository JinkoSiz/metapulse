"""Точка входа веб-приложения."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from arq.connections import RedisSettings, create_pool
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from redis.asyncio import from_url

from app.config import settings
from app.web.routes import STATIC_DIR, router

log = structlog.get_logger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level.upper())
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Redis нужен для SSE и постановки задач. Его недоступность не должна ронять витрину."""
    app.state.redis = None
    app.state.arq = None
    try:
        app.state.redis = from_url(settings.redis_url)
        await app.state.redis.ping()
    except Exception as exc:
        log.warning("redis_unavailable", error=str(exc))
        app.state.redis = None
    try:
        app.state.arq = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    except Exception as exc:
        log.warning("arq_pool_unavailable", error=str(exc))
        app.state.arq = None

    yield

    for attr in ("redis", "arq"):
        conn = getattr(app.state, attr, None)
        if conn is not None:
            with contextlib.suppress(Exception):
                await conn.aclose()


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="MetaPulse",
        description="Агрегатор игр Metacritic с LLM-резюме отзывов",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(router)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return app


app = create_app()
