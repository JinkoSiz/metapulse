"""Настройки arq-воркера: часовой cron + ручные запуски из веб-интерфейса.

Запуск: `arq app.pipeline.worker.WorkerSettings`
"""

from __future__ import annotations

from arq import cron
from arq.connections import RedisSettings

from app.config import settings
from app.pipeline.tasks import crawl_batch, shutdown, startup

# Час обходим в фиксированную минуту, а не в :00 — так прогон не совпадает с пиком
# нагрузки на Metacritic в начале часа.
CRON_MINUTE = max(0, min(59, settings.schedule_cron_minute))


class WorkerSettings:
    functions = [crawl_batch]
    cron_jobs = [
        cron(
            crawl_batch,
            minute=CRON_MINUTE,
            run_at_startup=False,
            timeout=1800,
            max_tries=1,
            unique=True,
        )
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.redis_url)

    # Обход берёт 20 игр и ходит по внешним API с троттлингом: параллелить нечего,
    # зато при max_jobs=1 рассуждать о лимитах запросов можно локально.
    max_jobs = 1
    job_timeout = 1800
    keep_result = 3600
    max_tries = 1
    health_check_interval = 30
