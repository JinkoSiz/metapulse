"""Конфигурация сервиса. Все значения приходят из переменных окружения (.env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- инфраструктура ---
    database_url: str = "postgresql+asyncpg://metapulse:metapulse@localhost:5432/metapulse"
    redis_url: str = "redis://localhost:6379/0"
    app_tz: str = "Europe/Moscow"
    admin_token: str = "change-me"
    log_level: str = "INFO"

    # --- расписание и объём выборки ---
    batch_size: int = 20
    schedule_cron_minute: int = 5  # ежечасно в HH:05
    catch_up_on_start: bool = True

    # --- Metacritic ---
    mc_api_key: str = "1MOZgmNFxvmljaQR1X9KAij9Mo4xAY3u"
    mc_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
    )
    mc_rate_limit_rps: float = 2.0
    mc_timeout_s: float = 20.0
    mc_max_retries: int = 4
    mc_proxy: str | None = None  # резидентный прокси, если Cloudflare ужесточит посадку
    mc_critic_reviews_max: int = 40  # сколько отзывов критиков забирать на игру
    mc_user_reviews_max: int = 60

    # --- LLM ---
    anthropic_api_key: str | None = None
    llm_model: str = "claude-haiku-4-5"
    llm_max_tokens: int = 2000
    llm_enabled: bool = True

    # --- эмбеддинги ---
    voyage_api_key: str | None = None
    embedding_model: str = "voyage-4-lite"
    embedding_dim: int = 1024
    similar_games_count: int = 8

    # --- YouTube (доп. часть 1) ---
    youtube_api_key: str | None = None
    youtube_enabled: bool = True
    youtube_proxy: str | None = None  # резидентный прокси для транскриптов
    youtube_transcript_max_chars: int = 60_000
    letsplay_ttl_days: int = 7

    # --- логи переписки с нейросетью ---
    llm_log_dir: Path = Field(default=REPO_ROOT / "logs" / "llm")

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.app_tz)

    @property
    def sync_database_url(self) -> str:
        """URL для Alembic (psycopg/sync-драйвер не нужен — alembic работает через async engine)."""
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
