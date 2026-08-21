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

    # --- LLM: сменные бэкенды ---
    # ollama — локальная модель на своём сервере (без ключей и оплаты),
    # anthropic — облачный Claude (лучше качество резюме на русском).
    llm_provider: str = "ollama"
    llm_enabled: bool = True
    llm_max_tokens: int = 2000

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-4-5"

    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:7b"
    # Локальная 7B на CPU читает промпт ~20 ток/с: 40 отзывов превратили бы обход
    # в многочасовой. Для локального бэкенда корпус режется до вменяемого объёма.
    ollama_reviews_limit: int = 25
    ollama_review_chars: int = 500
    # Транскрипт четырёхчасового прохождения — под сотню тысяч символов; локальной
    # модели достаётся начало, где блогер делится впечатлениями.
    ollama_transcript_chars: int = 9000
    ollama_timeout_s: float = 900.0
    # Сколько потоков отдавать модели. По умолчанию Ollama забирает все ядра и поднимает
    # load average выше числа процессоров — на сервере, где живут другие сервисы, это
    # заметно по ним. Половина ядер сохраняет им воздух ценой примерно трети скорости.
    ollama_num_threads: int = 4

    # Потолки времени внутри одного обхода. Локальная модель тратит минуты на игру, и без
    # ограничений прогон из 20 игр не укладывался в отведённое время: задачу убивал
    # job_timeout, и часть игр оставалась необработанной. Собранные игры видны на витрине
    # сразу, а тексты догоняются следующими заходами — input_hash и кэш летсплеев не дают
    # пересчитывать уже готовое. 0 — без ограничения.
    llm_budget_seconds: int = 900
    letsplay_budget_seconds: int = 600

    # --- эмбеддинги: сменные бэкенды ---
    # ollama — bge-m3 на своём сервере (1024 измерения, как колонка в БД),
    # voyage — облачный voyage-4-lite, none — лексический фолбэк по жанрам и названиям.
    embedding_provider: str = "ollama"
    embedding_dim: int = 1024
    similar_games_count: int = 8

    ollama_embedding_model: str = "bge-m3"
    voyage_api_key: str | None = None
    voyage_model: str = "voyage-4-lite"

    # --- YouTube (доп. часть 1) ---
    youtube_enabled: bool = True
    # yt-dlp ищет без ключа и без суточной квоты; official требует YOUTUBE_API_KEY
    youtube_search_backend: str = "yt-dlp"
    youtube_api_key: str | None = None
    # HTTP-прокси для запросов к YouTube: из РФ домен недоступен напрямую.
    # На своём сервере это singbox: http://singbox:1081
    youtube_proxy: str | None = None
    youtube_transcript_max_chars: int = 60_000
    youtube_search_results: int = 12
    letsplay_ttl_days: int = 7

    # --- логи переписки с нейросетью ---
    llm_log_dir: Path = Field(default=REPO_ROOT / "logs" / "llm")

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.app_tz)

    @property
    def llm_model(self) -> str:
        """Модель активного бэкенда — попадает в JSONL-логи и в карточку игры."""
        return self.ollama_model if self.llm_provider == "ollama" else self.anthropic_model

    @property
    def embedding_model(self) -> str:
        return (
            self.ollama_embedding_model
            if self.embedding_provider == "ollama"
            else self.voyage_model
        )

    @property
    def sync_database_url(self) -> str:
        """URL для Alembic (psycopg/sync-драйвер не нужен — alembic работает через async engine)."""
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
