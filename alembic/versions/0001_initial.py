"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-08-20
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1024


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "games",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("mc_id", sa.BigInteger(), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("developer", sa.Text(), nullable=True),
        sa.Column("publisher", sa.Text(), nullable=True),
        sa.Column("release_date", sa.Date(), nullable=True),
        sa.Column("esrb_rating", sa.String(length=16), nullable=True),
        sa.Column("genres", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("cover_url", sa.Text(), nullable=True),
        sa.Column("trailer_embed_url", sa.Text(), nullable=True),
        sa.Column("trailer_title", sa.Text(), nullable=True),
        sa.Column("lead_metascore", sa.Integer(), nullable=True),
        sa.Column("lead_userscore", sa.Numeric(precision=3, scale=1), nullable=True),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("embedding_hash", sa.String(length=64), nullable=True),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("last_scraped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("mc_id"),
        sa.UniqueConstraint("slug"),
    )
    op.create_index("ix_games_mc_id", "games", ["mc_id"])
    op.create_index("ix_games_slug", "games", ["slug"])
    op.create_index("ix_games_lead_metascore", "games", ["lead_metascore"])
    op.create_index(
        "ix_games_title_trgm", "games", ["title"],
        postgresql_using="gin", postgresql_ops={"title": "gin_trgm_ops"},
    )

    op.create_table(
        "game_platforms",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("platform_mc_id", sa.BigInteger(), nullable=True),
        sa.Column("platform_name", sa.String(length=128), nullable=False),
        sa.Column("platform_slug", sa.String(length=128), nullable=False),
        sa.Column("metascore", sa.Integer(), nullable=True),
        sa.Column("metascore_review_count", sa.Integer(), nullable=True),
        sa.Column("metascore_sentiment", sa.String(length=64), nullable=True),
        sa.Column("userscore", sa.Numeric(precision=3, scale=1), nullable=True),
        sa.Column("userscore_review_count", sa.Integer(), nullable=True),
        sa.Column("userscore_sentiment", sa.String(length=64), nullable=True),
        sa.Column("is_lead", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("platform_release_date", sa.Date(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("game_id", "platform_slug", name="uq_game_platform"),
    )
    op.create_index("ix_game_platforms_game_id", "game_platforms", ["game_id"])
    op.create_index("ix_game_platforms_platform_slug", "game_platforms", ["platform_slug"])

    op.create_table(
        "reviews",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=8), nullable=False),
        sa.Column("source_key", sa.String(length=255), nullable=False),
        sa.Column("platform_slug", sa.String(length=128), nullable=True),
        sa.Column("author", sa.Text(), nullable=True),
        sa.Column("publication", sa.Text(), nullable=True),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("quote", sa.Text(), nullable=True),
        sa.Column("review_date", sa.Date(), nullable=True),
        sa.Column("external_url", sa.Text(), nullable=True),
        sa.Column("spoiler", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("game_id", "kind", "source_key", name="uq_review_source"),
    )
    op.create_index("ix_reviews_game_id", "reviews", ["game_id"])

    op.create_table(
        "summaries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("likes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("dislikes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("tl_dr", sa.Text(), nullable=True),
        sa.Column("model", sa.String(length=64), nullable=True),
        sa.Column("input_hash", sa.String(length=64), nullable=True),
        sa.Column("source_review_count", sa.Integer(), nullable=True),
        sa.Column("llm_call_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("game_id", "kind", name="uq_summary_kind"),
    )
    op.create_index("ix_summaries_game_id", "summaries", ["game_id"])

    op.create_table(
        "letsplays",
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.String(length=32), nullable=True),
        sa.Column("video_url", sa.Text(), nullable=True),
        sa.Column("video_title", sa.Text(), nullable=True),
        sa.Column("channel", sa.Text(), nullable=True),
        sa.Column("view_count", sa.BigInteger(), nullable=True),
        sa.Column("duration_sec", sa.Integer(), nullable=True),
        sa.Column("transcript_source", sa.String(length=16), nullable=True),
        sa.Column("transcript_len", sa.Integer(), nullable=True),
        sa.Column("conclusion", sa.Text(), nullable=True),
        sa.Column("model", sa.String(length=64), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("game_id"),
    )

    op.create_table(
        "crawl_state",
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("phase", sa.String(length=16), nullable=False, server_default="carousel"),
        sa.Column("next_offset", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("runs_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("day"),
    )

    op.create_table(
        "pipeline_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trigger", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="running"),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stats", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "daily_seen",
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("mc_id", sa.BigInteger(), nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=True),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["pipeline_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("day", "mc_id", name="pk_daily_seen"),
    )

    op.create_table(
        "task_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("level", sa.String(length=16), nullable=False, server_default="info"),
        sa.Column("worker", sa.String(length=64), nullable=True),
        sa.Column("stage", sa.String(length=64), nullable=True),
        sa.Column("game_id", sa.Integer(), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["pipeline_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_task_events_run_id", "task_events", ["run_id"])

    op.create_table(
        "llm_calls",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="ok"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("jsonl_file", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_llm_calls_purpose", "llm_calls", ["purpose"])
    op.create_index("ix_llm_calls_game_id", "llm_calls", ["game_id"])

    op.create_table(
        "similar_games",
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("similar_id", sa.Integer(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["similar_id"], ["games.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("game_id", "rank", name="pk_similar_games"),
    )
    op.create_index("ix_similar_games_game_id", "similar_games", ["game_id"])


def downgrade() -> None:
    op.drop_table("similar_games")
    op.drop_table("llm_calls")
    op.drop_table("task_events")
    op.drop_table("daily_seen")
    op.drop_table("pipeline_runs")
    op.drop_table("crawl_state")
    op.drop_table("letsplays")
    op.drop_table("summaries")
    op.drop_table("reviews")
    op.drop_table("game_platforms")
    op.drop_table("games")
