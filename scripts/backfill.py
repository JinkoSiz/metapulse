"""Наполнение витрины заметными играми (для демо и первичного прогрева базы).

Штатный обход берёт игры строго по порядку из ленты новинок, как требует задание, —
а там много мелких релизов вовсе без оценок и отзывов. Этот скрипт прогоняет тот же
конвейер по играм, выбранным иначе (по популярности или метаскору), чтобы витрина и
резюме отзывов выглядели содержательно.

Запуск:
    python scripts/backfill.py                # 20 популярных игр
    python scripts/backfill.py 40 metascore   # 40 игр с лучшим метаскором
    python scripts/backfill.py elden-ring hades   # конкретные игры по слагам
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import structlog

from app.config import settings
from app.db.session import session_scope
from app.llm.client import LlmDisabled
from app.llm.embeddings import ensure_embedding, recompute_similar
from app.llm.summarize import summarize_game
from app.metacritic.client import BACKEND_BASE, FINDER_PATH, MetacriticClient
from app.pipeline.repository import upsert_game, upsert_platforms, upsert_reviews
from app.youtube.service import process_letsplay

log = structlog.get_logger(__name__)

SORTS = {
    "popular": "-popularityCount",
    "metascore": "-metaScore",
    "new": "-releaseDate",
}


async def discover(client: MetacriticClient, sort: str, limit: int) -> list[str]:
    """Слаги игр из finder-а с нужной сортировкой (метаскор > 0, чтобы не тянуть пустышки)."""
    params: dict[str, Any] = {
        "sortBy": SORTS.get(sort, SORTS["popular"]),
        "productType": "games",
        "metaScoreMin": 1,
        "offset": 0,
        "limit": min(limit, 50),
    }
    payload = await client._get_json(FINDER_PATH, params)  # noqa: SLF001 — служебный скрипт
    items = (payload.get("data") or {}).get("items") or []
    return [item["slug"] for item in items if item.get("slug")]


async def process(client: MetacriticClient, session: Any, slug: str) -> str:
    detail = await client.get_game(slug)
    game = await upsert_game(session, detail)

    userscores = {}
    for platform in detail.platforms:
        userscores[platform.slug] = await client.get_platform_userscore(detail.slug, platform.slug)
    await upsert_platforms(session, game, detail.platforms, userscores)

    critic = await client.get_critic_reviews(detail.slug, settings.mc_critic_reviews_max)
    user = await client.get_user_reviews(detail.slug, settings.mc_user_reviews_max)
    await upsert_reviews(session, game, [*critic, *user])
    await session.flush()

    summaries = 0
    for kind in ("critic", "user"):
        try:
            if await summarize_game(session, game, kind):
                summaries += 1
        except LlmDisabled:
            pass
    try:
        await ensure_embedding(session, game)
    except LlmDisabled:
        pass
    if settings.youtube_enabled:
        try:
            await process_letsplay(session, game)
        except Exception as exc:  # noqa: BLE001 — летсплей не должен ронять наполнение
            log.warning("backfill.letsplay_failed", slug=slug, error=str(exc))

    await session.flush()
    return (
        f"{game.title}: платформ {len(detail.platforms)}, "
        f"отзывов {len(critic)}+{len(user)}, резюме {summaries}"
    )


async def main(argv: list[str]) -> None:
    slugs: list[str] = [a for a in argv if not a.isdigit() and a not in SORTS]
    limit = next((int(a) for a in argv if a.isdigit()), 20)
    sort = next((a for a in argv if a in SORTS), "popular")

    async with MetacriticClient() as client, session_scope() as session:
        targets = slugs or await discover(client, sort, limit)
        print(f"К обработке: {len(targets)} игр (источник: {'слаги' if slugs else sort})\n")

        for index, slug in enumerate(targets, start=1):
            try:
                line = await process(client, session, slug)
                print(f"[{index}/{len(targets)}] {line}")
            except Exception as exc:  # noqa: BLE001 — пропускаем проблемную игру
                print(f"[{index}/{len(targets)}] {slug}: ОШИБКА {type(exc).__name__}: {exc}")
                await session.rollback()

        updated = await recompute_similar(session)
        print(f"\nПохожие игры пересчитаны для {updated} игр")


if __name__ == "__main__":
    print(f"backend: {BACKEND_BASE}")
    asyncio.run(main(sys.argv[1:]))
