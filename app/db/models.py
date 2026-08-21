"""Схема БД. Единый контракт для скрапера, LLM-контура, воркера и веб-слоя."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 1024


class Base(DeclarativeBase):
    pass


def _now() -> Mapped[dt.datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Game(Base):
    """Игра. lead_* денормализованы для сортировки списка без JOIN."""

    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mc_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False, index=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    developer: Mapped[str | None] = mapped_column(Text)
    publisher: Mapped[str | None] = mapped_column(Text)
    release_date: Mapped[dt.date | None] = mapped_column(Date)
    esrb_rating: Mapped[str | None] = mapped_column(String(16))
    genres: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    cover_url: Mapped[str | None] = mapped_column(Text)
    trailer_embed_url: Mapped[str | None] = mapped_column(Text)
    trailer_title: Mapped[str | None] = mapped_column(Text)

    # денормализация lead-платформы: сортировка списка по рейтингу
    lead_metascore: Mapped[int | None] = mapped_column(Integer, index=True)
    lead_userscore: Mapped[float | None] = mapped_column(Numeric(3, 1))

    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_hash: Mapped[str | None] = mapped_column(String(64))

    first_seen_at: Mapped[dt.datetime] = _now()
    last_scraped_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    platforms: Mapped[list[GamePlatform]] = relationship(
        back_populates="game", cascade="all, delete-orphan", lazy="selectin"
    )
    summaries: Mapped[list[Summary]] = relationship(
        back_populates="game", cascade="all, delete-orphan", lazy="selectin"
    )
    letsplay: Mapped[LetsPlay | None] = relationship(
        back_populates="game", cascade="all, delete-orphan", uselist=False, lazy="selectin"
    )

    __table_args__ = (
        Index(
            "ix_games_title_trgm",
            "title",
            postgresql_using="gin",
            postgresql_ops={"title": "gin_trgm_ops"},
        ),
    )


class GamePlatform(Base):
    """Платформа игры со своим Metascore и Userscore.

    metascore    — из detail-эндпоинта, platforms[].criticScoreSummary
    userscore    — из /reviews/.../user/games/{slug}/platform/{slug}/stats/web
                   ВАЖНО: платформа только path-сегментом, query-параметр молча игнорируется.
    """

    __tablename__ = "game_platforms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    platform_mc_id: Mapped[int | None] = mapped_column(BigInteger)
    platform_name: Mapped[str] = mapped_column(String(128), nullable=False)
    platform_slug: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    metascore: Mapped[int | None] = mapped_column(Integer)
    metascore_review_count: Mapped[int | None] = mapped_column(Integer)
    metascore_sentiment: Mapped[str | None] = mapped_column(String(64))
    userscore: Mapped[float | None] = mapped_column(Numeric(3, 1))
    userscore_review_count: Mapped[int | None] = mapped_column(Integer)
    userscore_sentiment: Mapped[str | None] = mapped_column(String(64))

    is_lead: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    platform_release_date: Mapped[dt.date | None] = mapped_column(Date)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    game: Mapped[Game] = relationship(back_populates="platforms")

    __table_args__ = (UniqueConstraint("game_id", "platform_slug", name="uq_game_platform"),)


class Review(Base):
    """Кэш сырых отзывов: резюме пересобирается без повторного скрапа."""

    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(8), nullable=False)  # 'critic' | 'user'
    source_key: Mapped[str] = mapped_column(String(255), nullable=False)
    platform_slug: Mapped[str | None] = mapped_column(String(128))
    author: Mapped[str | None] = mapped_column(Text)
    publication: Mapped[str | None] = mapped_column(Text)
    score: Mapped[int | None] = mapped_column(Integer)  # критики 0-100, юзеры 0-10
    quote: Mapped[str | None] = mapped_column(Text)
    review_date: Mapped[dt.date | None] = mapped_column(Date)
    external_url: Mapped[str | None] = mapped_column(Text)
    spoiler: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    fetched_at: Mapped[dt.datetime] = _now()

    __table_args__ = (UniqueConstraint("game_id", "kind", "source_key", name="uq_review_source"),)


class Summary(Base):
    """LLM-резюме отзывов. kind: critic | user | letsplay.

    input_hash — sha256 корпуса-входа: если не изменился, LLM не дёргаем.
    """

    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)

    likes: Mapped[list[str] | None] = mapped_column(JSONB)
    dislikes: Mapped[list[str] | None] = mapped_column(JSONB)
    tl_dr: Mapped[str | None] = mapped_column(Text)

    model: Mapped[str | None] = mapped_column(String(64))
    input_hash: Mapped[str | None] = mapped_column(String(64))
    source_review_count: Mapped[int | None] = mapped_column(Integer)
    llm_call_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    created_at: Mapped[dt.datetime] = _now()
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    game: Mapped[Game] = relationship(back_populates="summaries")

    __table_args__ = (UniqueConstraint("game_id", "kind", name="uq_summary_kind"),)


class LetsPlay(Base):
    """Доп. часть 1: самый популярный летсплей + транскрипт + заключение."""

    __tablename__ = "letsplays"

    game_id: Mapped[int] = mapped_column(
        ForeignKey("games.id", ondelete="CASCADE"), primary_key=True
    )
    video_id: Mapped[str | None] = mapped_column(String(32))
    video_url: Mapped[str | None] = mapped_column(Text)
    video_title: Mapped[str | None] = mapped_column(Text)
    channel: Mapped[str | None] = mapped_column(Text)
    view_count: Mapped[int | None] = mapped_column(BigInteger)
    duration_sec: Mapped[int | None] = mapped_column(Integer)
    # 'captions' | 'ytdlp' | 'stt' | 'none'
    transcript_source: Mapped[str | None] = mapped_column(String(16))
    transcript_len: Mapped[int | None] = mapped_column(Integer)
    conclusion: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(64))
    error: Mapped[str | None] = mapped_column(Text)
    fetched_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    game: Mapped[Game] = relationship(back_populates="letsplay")


class CrawlState(Base):
    """Состояние выборки за день. Новая дата -> строки нет -> «начинаем заново»."""

    __tablename__ = "crawl_state"

    day: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    phase: Mapped[str] = mapped_column(String(16), nullable=False, default="carousel")
    next_offset: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    runs_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class DailySeen(Base):
    """«Сегодня ещё не обрабатывал» — анти-join + дедуп при переупорядочивании выдачи."""

    __tablename__ = "daily_seen"

    day: Mapped[dt.date] = mapped_column(Date, nullable=False)
    mc_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    game_id: Mapped[int | None] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"))
    run_id: Mapped[int | None] = mapped_column(ForeignKey("pipeline_runs.id", ondelete="SET NULL"))
    processed_at: Mapped[dt.datetime] = _now()

    __table_args__ = (PrimaryKeyConstraint("day", "mc_id", name="pk_daily_seen"),)


class PipelineRun(Base):
    """Один обход: cron или ручной запуск."""

    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trigger: Mapped[str] = mapped_column(String(16), nullable=False)  # cron | manual | catchup
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    started_at: Mapped[dt.datetime] = _now()
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    stats: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)


class TaskEvent(Base):
    """Персистентная лента событий: мониторинг переживает рестарт."""

    __tablename__ = "task_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int | None] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="CASCADE"), index=True
    )
    ts: Mapped[dt.datetime] = _now()
    level: Mapped[str] = mapped_column(String(16), nullable=False, default="info")
    worker: Mapped[str | None] = mapped_column(String(64))
    stage: Mapped[str | None] = mapped_column(String(64))
    game_id: Mapped[int | None] = mapped_column(Integer)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class LlmCall(Base):
    """Индекс JSONL-строк для UI. Первичный артефакт — сами файлы logs/llm/*.jsonl."""

    __tablename__ = "llm_calls"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    ts: Mapped[dt.datetime] = _now()
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    # critic_summary | user_summary | letsplay_conclusion | embedding
    purpose: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    game_id: Mapped[int | None] = mapped_column(Integer, index=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ok")
    error: Mapped[str | None] = mapped_column(Text)
    jsonl_file: Mapped[str | None] = mapped_column(Text)


class SimilarGame(Base):
    """Предрасчитанные похожие игры (обновляются после каждого прогона)."""

    __tablename__ = "similar_games"

    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    similar_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"))
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (PrimaryKeyConstraint("game_id", "rank", name="pk_similar_games"),)
