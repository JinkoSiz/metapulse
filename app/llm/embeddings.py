"""Похожие игры: эмбеддинги Voyage + косинусный поиск в pgvector.

Без ключа Voyage модуль не отключается, а переходит на лексический фолбэк
(пересечение жанров + похожесть названий через pg_trgm): демо должно показывать
похожие игры даже на установке без единого ключа API.

ANN-индекс на векторной колонке сознательно не создаётся — при сотнях игр точный
перебор оператором `<=>` занимает микросекунды, а индекс пришлось бы обслуживать.
"""

from __future__ import annotations

import hashlib

import structlog
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Game, SimilarGame
from app.llm.client import LlmClient, LlmDisabled, LlmError

log = structlog.get_logger(__name__)

# Описание обрезаем: для сходства хватает завязки, а хвост лишь размывает вектор.
DESCRIPTION_CHARS = 1200


def build_embedding_text(game: Game) -> str:
    """Текст, который отправляется в эмбеддер: название, жанры, описание."""
    genres = ", ".join(genre_names(game.genres))
    description = (game.description or "").strip()[:DESCRIPTION_CHARS]
    parts = [game.title or ""]
    if genres:
        parts.append(f"Жанры: {genres}")
    if description:
        parts.append(description)
    return "\n".join(p for p in parts if p)


def genre_names(raw: object) -> list[str]:
    """Названия жанров: в JSONB лежит либо список строк, либо список объектов {name: ...}."""
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        name = item.get("name") if isinstance(item, dict) else item
        if name and str(name) not in names:
            names.append(str(name))
    return names


def embedding_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def ensure_embedding(session: AsyncSession, game: Game) -> bool:
    """Считает эмбеддинг игры, если входной текст изменился.

    Возвращает True, если вектор был пересчитан. `LlmDisabled` пробрасывается наверх:
    вызывающий сам решает, тихо пропустить или залогировать.
    """
    payload = build_embedding_text(game)
    if not payload.strip():
        return False

    digest = embedding_hash(payload)
    if game.embedding_hash == digest and game.embedding is not None:
        return False

    client = LlmClient(session)
    try:
        vectors = await client.embed([payload], game_id=game.id, input_type="document")
    finally:
        await client.aclose()

    if not vectors:
        return False
    vector = vectors[0]
    if len(vector) != settings.embedding_dim:
        # Размерность колонки фиксирована миграцией: чужая модель молча всё сломает.
        raise LlmError(
            f"Размерность эмбеддинга {len(vector)} не совпадает с колонкой "
            f"{settings.embedding_dim} (модель {settings.embedding_model})"
        )

    game.embedding = vector
    game.embedding_hash = digest
    await session.flush()
    log.debug("embedding.updated", slug=game.slug)
    return True


async def recompute_similar(session: AsyncSession, game_ids: list[int] | None = None) -> int:
    """Пересчитывает similar_games. Возвращает число игр, для которых нашлись похожие."""
    limit = settings.similar_games_count

    targets = await _target_ids(session, game_ids)
    if not targets:
        return 0

    vector_ready = await session.scalar(
        select(func.count()).select_from(Game).where(Game.embedding.is_not(None))
    )
    use_vectors = int(vector_ready or 0) >= 2

    updated = 0
    for game_id in targets:
        rows = (
            await _similar_by_vector(session, game_id, limit)
            if use_vectors
            else await _similar_by_lexical(session, game_id, limit)
        )
        if not rows:
            continue
        await session.execute(delete(SimilarGame).where(SimilarGame.game_id == game_id))
        for rank, (similar_id, score) in enumerate(rows, start=1):
            session.add(
                SimilarGame(game_id=game_id, similar_id=similar_id, rank=rank, score=float(score))
            )
        updated += 1

    await session.flush()
    log.info("similar.recomputed", games=updated, vectors=use_vectors)
    return updated


async def _target_ids(session: AsyncSession, game_ids: list[int] | None) -> list[int]:
    if game_ids:
        return [int(g) for g in game_ids]
    return [int(x) for x in (await session.scalars(select(Game.id))).all()]


async def _similar_by_vector(
    session: AsyncSession, game_id: int, limit: int
) -> list[tuple[int, float]]:
    """Косинусная близость pgvector. Игры без вектора в выдачу не попадают."""
    source = await session.get(Game, game_id)
    if source is None or source.embedding is None:
        return []

    distance = Game.embedding.cosine_distance(source.embedding)
    stmt = (
        select(Game.id, distance.label("distance"))
        .where(Game.id != game_id, Game.embedding.is_not(None))
        .order_by(distance)
        .limit(limit)
    )
    rows = await session.execute(stmt)
    return [(int(row.id), 1.0 - float(row.distance)) for row in rows]


# Лексический фолбэк: 3 балла за каждый общий жанр + похожесть названий (pg_trgm).
# Формула грубая, но даёт осмысленный порядок и не требует ключей API.
_LEXICAL_SQL = text(
    """
    SELECT g.id,
           COALESCE(shared.cnt, 0) * 3.0 + similarity(g.title, :title) AS score
      FROM games AS g
      LEFT JOIN LATERAL (
           SELECT count(*) AS cnt
             FROM jsonb_array_elements(COALESCE(g.genres, '[]'::jsonb)) AS gj
            WHERE COALESCE(gj->>'name', gj#>>'{}') = ANY(:genres)
      ) AS shared ON TRUE
     WHERE g.id <> :game_id
       AND (COALESCE(shared.cnt, 0) > 0 OR similarity(g.title, :title) > 0.1)
     ORDER BY score DESC, g.lead_metascore DESC NULLS LAST, g.id DESC
     LIMIT :limit
    """
)


async def _similar_by_lexical(
    session: AsyncSession, game_id: int, limit: int
) -> list[tuple[int, float]]:
    source = await session.get(Game, game_id)
    if source is None:
        return []
    genres = genre_names(source.genres)
    rows = await session.execute(
        _LEXICAL_SQL,
        {
            "game_id": game_id,
            "title": source.title or "",
            "genres": genres or [""],
            "limit": limit,
        },
    )
    return [(int(row.id), float(row.score)) for row in rows]


__all__ = [
    "LlmDisabled",
    "build_embedding_text",
    "ensure_embedding",
    "genre_names",
    "recompute_similar",
]
