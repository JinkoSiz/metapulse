"""Живая проверка конвейера сбора: Metacritic -> БД, без LLM.

Берёт несколько игр из карусели New Releases, сохраняет их вместе с платформами,
пер-платформенными оценками и отзывами, затем печатает, что реально легло в базу.

Запуск: python scripts/smoke_crawl.py [сколько игр]
"""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import func, select

from app.config import settings
from app.db.models import GamePlatform, Review
from app.db.session import session_scope
from app.metacritic.client import MetacriticClient
from app.pipeline.repository import upsert_game, upsert_platforms, upsert_reviews


async def main(limit: int = 3, slugs: list[str] | None = None) -> None:
    async with MetacriticClient() as client, session_scope() as session:
        if slugs:
            targets = slugs
            print(f"Проверяем игры по слагам: {', '.join(slugs)}\n")
        else:
            items = await client.list_new_releases(limit)
            targets = [item.slug for item in items[:limit]]
            print(f"Карусель New Releases: {len(items)} игр\n")

        for slug in targets:
            detail = await client.get_game(slug)
            game = await upsert_game(session, detail)

            userscores = {}
            for platform in detail.platforms:
                userscores[platform.slug] = await client.get_platform_userscore(
                    detail.slug, platform.slug
                )
            await upsert_platforms(session, game, detail.platforms, userscores)

            critic = await client.get_critic_reviews(detail.slug, 20)
            user = await client.get_user_reviews(detail.slug, 20)
            new_reviews = await upsert_reviews(session, game, [*critic, *user])
            await session.flush()

            print(f"=== {game.title} ({game.slug})")
            print(f"    разработчик: {game.developer or '—'} | издатель: {game.publisher or '—'}")
            print(f"    дата: {game.release_date} | жанры: {game.genres}")
            print(f"    обложка: {game.cover_url}")
            print(f"    трейлер: {game.trailer_embed_url or '—'}")
            print(f"    описание: {(game.description or '')[:90]}…")

            rows = await session.scalars(
                select(GamePlatform).where(GamePlatform.game_id == game.id)
            )
            for platform in rows:
                lead = " (основная)" if platform.is_lead else ""
                print(
                    f"    · {platform.platform_name}{lead}: "
                    f"Metascore {platform.metascore or '—'} "
                    f"({platform.metascore_review_count or 0} отз.), "
                    f"Userscore {platform.userscore or '—'} "
                    f"({platform.userscore_review_count or 0} отз.)"
                )
            print(
                f"    отзывов: критики {len(critic)}, игроки {len(user)}, "
                f"новых в БД {new_reviews}\n"
            )

        totals = await session.execute(select(Review.kind, func.count()).group_by(Review.kind))
        print("Итого в reviews:", dict(totals.all()))
        with_userscore = await session.scalar(
            select(func.count())
            .select_from(GamePlatform)
            .where(GamePlatform.userscore.is_not(None))
        )
        print(f"Платформ с заполненным Userscore: {with_userscore}")
        print(
            f"LLM включён: {bool(settings.anthropic_api_key)} | "
            f"Voyage: {bool(settings.voyage_api_key)}"
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and not args[0].isdigit():
        asyncio.run(main(slugs=args))
    else:
        asyncio.run(main(limit=int(args[0]) if args else 3))
