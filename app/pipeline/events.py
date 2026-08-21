"""Шина событий пайплайна: персистентная лента в БД + live-канал в Redis.

Два адресата у одного события не дублирование, а разделение ролей: `task_events`
переживает рестарт и рисует историю на /monitor, Redis pub/sub кормит SSE в реальном
времени. Сбой Redis не должен ронять обход, поэтому публикация в канал — best effort.
"""

from __future__ import annotations

import datetime as dt
import json
import socket
from typing import Any

import structlog
from redis.asyncio import Redis, from_url
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import TaskEvent

log = structlog.get_logger(__name__)

EVENTS_CHANNEL = "metapulse:events"
WORKER_KEY_PREFIX = "metapulse:worker:"
WORKER_TTL_S = 15  # heartbeat раз в несколько секунд: ключ протухает раньше, чем «умерший» воркер


def default_worker_name() -> str:
    """Имя воркера — hostname контейнера: в compose оно уникально и читаемо."""
    return socket.gethostname()


def _as_text(value: str | bytes) -> str:
    """arq отдаёт свой Redis без decode_responses (в очереди лежат бинарные джобы)."""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


class EventBus:
    """Публикация событий обхода. Redis-соединение можно передать снаружи (arq ctx)."""

    def __init__(self, redis: Redis | None = None, *, worker: str | None = None) -> None:
        self._redis = redis
        self._owns_redis = redis is None
        self.worker = worker or default_worker_name()

    async def _conn(self) -> Redis:
        if self._redis is None:
            self._redis = from_url(settings.redis_url, decode_responses=True)
        return self._redis

    async def close(self) -> None:
        """Закрывать соединение вправе только тот, кто его открыл."""
        if self._owns_redis and self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def publish(
        self,
        session: AsyncSession,
        run_id: int | None,
        *,
        stage: str,
        message: str,
        level: str = "info",
        game_id: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Записать событие в task_events и разослать его подписчикам Redis."""
        ts = dt.datetime.now(dt.UTC)
        event = TaskEvent(
            run_id=run_id,
            ts=ts,
            level=level,
            worker=self.worker,
            stage=stage,
            game_id=game_id,
            message=message,
            payload=payload,
        )
        session.add(event)
        await session.flush()

        body = {
            "id": event.id,
            "run_id": run_id,
            "ts": ts.isoformat(),
            "level": level,
            "worker": self.worker,
            "stage": stage,
            "game_id": game_id,
            "message": message,
            "payload": payload,
        }
        await self._fanout(body)
        log.info("pipeline.event", stage=stage, level=level, message=message, run_id=run_id)

    async def _fanout(self, body: dict[str, Any]) -> None:
        try:
            conn = await self._conn()
            await conn.publish(EVENTS_CHANNEL, json.dumps(body, ensure_ascii=False, default=str))
        except Exception as exc:  # мониторинг не важнее самого обхода
            log.warning("events.publish_failed", error=str(exc))

    async def heartbeat(self, worker: str, state: dict[str, Any]) -> None:
        """SETEX metapulse:worker:{worker} 15 <json> — признак живого воркера."""
        body = dict(state)
        body.setdefault("worker", worker)
        body["ts"] = dt.datetime.now(dt.UTC).isoformat()
        try:
            conn = await self._conn()
            await conn.setex(
                f"{WORKER_KEY_PREFIX}{worker}",
                WORKER_TTL_S,
                json.dumps(body, ensure_ascii=False, default=str),
            )
        except Exception as exc:
            log.warning("events.heartbeat_failed", worker=worker, error=str(exc))

    async def workers(self) -> list[dict[str, Any]]:
        """Живые воркеры для /monitor: непротухшие ключи metapulse:worker:*."""
        result: list[dict[str, Any]] = []
        try:
            conn = await self._conn()
            keys = [key async for key in conn.scan_iter(match=f"{WORKER_KEY_PREFIX}*")]
            if not keys:
                return []
            for key, raw in zip(keys, await conn.mget(keys), strict=True):
                if not raw:
                    continue
                name = _as_text(key).split(WORKER_KEY_PREFIX, 1)[-1]
                try:
                    body = json.loads(_as_text(raw))
                except (TypeError, ValueError):
                    body = {}
                body.setdefault("worker", name)
                result.append(body)
        except Exception as exc:
            log.warning("events.workers_failed", error=str(exc))
            return []
        return sorted(result, key=lambda item: str(item.get("worker", "")))
