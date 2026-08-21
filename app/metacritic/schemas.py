"""Pydantic-модели ответов Metacritic.

Модели намеренно «плоские»: это контракт между скрапером и остальными модулями,
а не зеркало сырого JSON. Всё лишнее из ответа отбрасывается (`extra="ignore"`).
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict


class ScoreStats(BaseModel):
    """Агрегированный балл: metascore платформы или userscore платформы."""

    model_config = ConfigDict(extra="ignore")

    score: float | None = None
    review_count: int | None = None
    sentiment: str | None = None


class PlatformInfo(BaseModel):
    """Платформа игры. `metascore` приходит из деталки, userscore — отдельным запросом."""

    model_config = ConfigDict(extra="ignore")

    mc_id: int | None = None
    name: str
    slug: str
    is_lead: bool = False
    release_date: dt.date | None = None
    metascore: ScoreStats | None = None


class FinderItem(BaseModel):
    """Элемент листинга (карусель New Releases или browse).

    В finder-выдаче нет ни списка платформ, ни userscore — их добирает деталка.
    """

    model_config = ConfigDict(extra="ignore")

    mc_id: int
    slug: str
    title: str
    release_date: dt.date | None = None
    description: str | None = None
    cover_url: str | None = None
    metascore: int | None = None
    genres: list[str] = []


class GameDetail(BaseModel):
    """Деталка игры со всеми платформами и их метаскорами."""

    model_config = ConfigDict(extra="ignore")

    mc_id: int
    slug: str
    title: str
    description: str | None = None
    developer: str | None = None
    publisher: str | None = None
    release_date: dt.date | None = None
    esrb_rating: str | None = None
    genres: list[str] = []
    cover_url: str | None = None
    trailer_embed_url: str | None = None
    trailer_title: str | None = None
    platforms: list[PlatformInfo] = []
    lead_metascore: int | None = None

    @property
    def lead_platform(self) -> PlatformInfo | None:
        """Lead-платформа: к ней относятся все агрегаты верхнего уровня."""
        for platform in self.platforms:
            if platform.is_lead:
                return platform
        return None


class ReviewItem(BaseModel):
    """Отзыв критика или пользователя.

    `score`: критики — 0..100, пользователи — 0..10 (шкалы намеренно не сводятся).
    """

    model_config = ConfigDict(extra="ignore")

    source_key: str
    kind: str
    score: int | None = None
    quote: str | None = None
    author: str | None = None
    publication: str | None = None
    review_date: dt.date | None = None
    external_url: str | None = None
    platform_slug: str | None = None
    spoiler: bool = False
