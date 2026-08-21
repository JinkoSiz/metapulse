"""HTTP-маршруты веб-интерфейса.

Страницы рендерятся на сервере и полностью работают без JavaScript;
HTMX только подменяет сетку карточек без перезагрузки, а SSE оживляет монитор.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import decimal
import json
import re
import secrets
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import structlog
from arq.connections import ArqRedis, RedisSettings, create_pool
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.config import settings
from app.db.models import Game
from app.db.session import SessionLocal
from app.web.queries import (
    SORTS,
    dashboard_stats,
    get_game_by_slug,
    get_similar,
    list_games,
    normalize_sort,
    platform_facets,
)

log = structlog.get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

EVENTS_CHANNEL = "metapulse:events"
WORKER_KEY_PREFIX = "metapulse:worker:"
SNAPSHOT_INTERVAL_S = 3.0
SSE_PING_S = 15
PER_PAGE = 24

# Имя файла лога: только дата и расширение. Всё остальное — попытка выйти из каталога.
LOG_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.jsonl$")

router = APIRouter()


# --------------------------------------------------------------------------- #
# Шаблоны и фильтры
# --------------------------------------------------------------------------- #

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, decimal.Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_class(value: Any, scale: int = 100) -> str:
    """Цвет бейджа: >=75 зелёный, 50-74 жёлтый, <50 красный, нет данных — серый."""
    num = _as_float(value)
    if num is None:
        return "na"
    normalized = num * 10 if scale == 10 else num
    if normalized >= 75:
        return "good"
    if normalized >= 50:
        return "mid"
    return "bad"


def fmt_metascore(value: Any) -> str:
    num = _as_float(value)
    return "—" if num is None else str(int(round(num)))


def fmt_userscore(value: Any) -> str:
    num = _as_float(value)
    return "—" if num is None else f"{num:.1f}"


def fmt_date(value: Any) -> str:
    if isinstance(value, dt.datetime):
        return value.astimezone(settings.tz).strftime("%d.%m.%Y %H:%M")
    if isinstance(value, dt.date):
        return value.strftime("%d.%m.%Y")
    return "—"


def fmt_int(value: Any) -> str:
    num = _as_float(value)
    if num is None:
        return "—"
    return f"{int(num):,}".replace(",", " ")


def genre_names(value: Any) -> list[str]:
    """genres в БД — JSONB: пайплайн может положить и список строк, и список объектов."""
    if not value:
        return []
    names: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = item.get("name") or item.get("title") or item.get("slug")
            if name:
                names.append(str(name))
        elif item:
            names.append(str(item))
    return names


def summary_of(game: Game, kind: str) -> Any:
    for summary in game.summaries or []:
        if summary.kind == kind:
            return summary
    return None


templates.env.filters["score_class"] = score_class
templates.env.filters["metascore"] = fmt_metascore
templates.env.filters["userscore"] = fmt_userscore
templates.env.filters["date_ru"] = fmt_date
templates.env.filters["int_ru"] = fmt_int
templates.env.filters["genres"] = genre_names
templates.env.globals["summary_of"] = summary_of
templates.env.globals["SORTS"] = SORTS


# --------------------------------------------------------------------------- #
# Инфраструктурные помощники
# --------------------------------------------------------------------------- #

_arq_lock = asyncio.Lock()


def get_redis(request: Request) -> Any:
    """Redis-пул из lifespan. None означает «Redis недоступен» — не повод падать."""
    return getattr(request.app.state, "redis", None)


async def get_arq_pool(request: Request) -> ArqRedis:
    pool = getattr(request.app.state, "arq", None)
    if pool is not None:
        return pool
    async with _arq_lock:
        pool = getattr(request.app.state, "arq", None)
        if pool is None:
            pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
            request.app.state.arq = pool
        return pool


async def list_workers(redis: Any) -> list[dict[str, Any]]:
    """Живые воркеры = непротухшие ключи metapulse:worker:* (SETEX из EventBus.heartbeat)."""
    if redis is None:
        return []
    workers: list[dict[str, Any]] = []
    try:
        async for key in redis.scan_iter(match=f"{WORKER_KEY_PREFIX}*", count=100):
            name = key.decode() if isinstance(key, bytes) else str(key)
            raw = await redis.get(name)
            ttl = await redis.ttl(name)
            state: Any = {}
            if raw:
                raw_text = raw.decode() if isinstance(raw, bytes) else str(raw)
                try:
                    state = json.loads(raw_text)
                except json.JSONDecodeError:
                    state = {"raw": raw_text}
            workers.append(
                {
                    "worker": name[len(WORKER_KEY_PREFIX) :],
                    "ttl": int(ttl) if ttl is not None else None,
                    "state": state if isinstance(state, dict) else {"value": state},
                }
            )
    except Exception as exc:  # Redis мог упасть уже после старта
        log.warning("workers_scan_failed", error=str(exc))
        return []
    return sorted(workers, key=lambda w: w["worker"])


async def build_snapshot(request: Request) -> dict[str, Any]:
    async with SessionLocal() as session:
        stats = await dashboard_stats(session)
    stats["workers"] = await list_workers(get_redis(request))
    return stats


def _is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true" and not request.headers.get(
        "HX-History-Restore-Request"
    )


# --------------------------------------------------------------------------- #
# Каталог
# --------------------------------------------------------------------------- #


async def _catalog_context(
    session: AsyncSession,
    request: Request,
    q: str,
    platform: str,
    sort: str,
    page: int,
) -> dict[str, Any]:
    games, total = await list_games(
        session, q=q, platform=platform, sort=sort, page=page, per_page=PER_PAGE
    )
    total_pages = max(1, -(-total // PER_PAGE))  # ceil
    filters = {k: v for k, v in (("q", q), ("platform", platform), ("sort", sort)) if v}
    return {
        "request": request,
        "games": games,
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "q": q,
        "platform": platform,
        "sort": sort,
        # Готовая query-строка без page: пагинация не теряет фильтры и работает без JS
        "base_query": urlencode(filters),
        "has_filters": bool(filters and (q or platform)),
    }


@router.get("/", response_class=HTMLResponse, summary="Каталог игр")
async def index(
    request: Request,
    q: str = Query("", max_length=200),
    platform: str = Query("", max_length=128),
    sort: str = Query("date"),
    page: int = Query(1),
) -> HTMLResponse:
    q = q.strip()
    platform = platform.strip()
    sort = normalize_sort(sort)
    page = max(1, page)

    async with SessionLocal() as session:
        context = await _catalog_context(session, request, q, platform, sort, page)
        context["facets"] = await platform_facets(session)

    # HTMX подменяет только сетку; обычный запрос получает страницу целиком
    template = "_game_grid.html" if _is_htmx(request) else "index.html"
    return templates.TemplateResponse(request, template, context)


@router.get("/game/{slug}", response_class=HTMLResponse, summary="Карточка игры")
async def game_page(request: Request, slug: str) -> HTMLResponse:
    async with SessionLocal() as session:
        game = await get_game_by_slug(session, slug)
        if game is None:
            raise HTTPException(status_code=404, detail="Игра не найдена")
        similar = await get_similar(session, game.id)
        platforms = sorted(
            game.platforms or [],
            key=lambda p: (not p.is_lead, p.platform_name or p.platform_slug),
        )

    letsplay = game.letsplay
    # Блок летсплея бессмысленен без транскрипта — пайплайн помечает такие записи 'none'
    show_letsplay = bool(letsplay and (letsplay.transcript_source or "none") != "none")

    return templates.TemplateResponse(
        request,
        "game.html",
        {
            "request": request,
            "game": game,
            "platforms": platforms,
            "similar": similar,
            "letsplay": letsplay if show_letsplay else None,
            "critic_summary": summary_of(game, "critic"),
            "user_summary": summary_of(game, "user"),
        },
    )


# --------------------------------------------------------------------------- #
# Монитор
# --------------------------------------------------------------------------- #


@router.get("/monitor", response_class=HTMLResponse, summary="Мониторинг обработки")
async def monitor(request: Request) -> HTMLResponse:
    stats = await build_snapshot(request)
    return templates.TemplateResponse(request, "monitor.html", {"request": request, "stats": stats})


@router.get("/api/stats", summary="Счётчики монитора")
async def api_stats(request: Request) -> JSONResponse:
    return JSONResponse(await build_snapshot(request))


async def _event_stream(request: Request) -> AsyncIterator[ServerSentEvent]:
    """События из Redis-канала + снапшот счётчиков раз в SNAPSHOT_INTERVAL_S секунд."""
    redis = get_redis(request)
    pubsub = None
    if redis is not None:
        try:
            pubsub = redis.pubsub()
            await pubsub.subscribe(EVENTS_CHANNEL)
        except Exception as exc:
            log.warning("sse_subscribe_failed", error=str(exc))
            pubsub = None

    last_snapshot = 0.0
    try:
        while True:
            if await request.is_disconnected():
                break

            now = time.monotonic()
            if now - last_snapshot >= SNAPSHOT_INTERVAL_S:
                last_snapshot = now
                try:
                    snapshot = await build_snapshot(request)
                    yield ServerSentEvent(
                        event="stats", data=json.dumps(snapshot, ensure_ascii=False, default=str)
                    )
                except Exception as exc:
                    log.warning("sse_snapshot_failed", error=str(exc))
                    yield ServerSentEvent(
                        event="error", data=json.dumps({"message": str(exc)}, ensure_ascii=False)
                    )

            if pubsub is None:
                await asyncio.sleep(1.0)
                continue

            try:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            except Exception as exc:
                log.warning("sse_pubsub_failed", error=str(exc))
                pubsub = None
                continue
            if message and message.get("type") == "message":
                data = message.get("data")
                if isinstance(data, bytes):
                    data = data.decode("utf-8", "replace")
                yield ServerSentEvent(event="task", data=str(data))
    except asyncio.CancelledError:  # клиент отвалился — сворачиваемся молча
        raise
    finally:
        if pubsub is not None:
            with contextlib.suppress(Exception):
                await pubsub.unsubscribe(EVENTS_CHANNEL)
            with contextlib.suppress(Exception):
                await pubsub.aclose()


@router.get("/api/events", summary="SSE-поток событий")
async def api_events(request: Request) -> EventSourceResponse:
    # ping шлёт comment-строки: без них прокси рвут «молчащее» соединение
    return EventSourceResponse(_event_stream(request), ping=SSE_PING_S)


@router.post("/api/admin/run", summary="Ручной запуск обработки")
async def api_admin_run(request: Request) -> JSONResponse:
    token = request.headers.get("X-Admin-Token", "")
    if not token or not secrets.compare_digest(token, settings.admin_token):
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-Admin-Token")

    try:
        pool = await get_arq_pool(request)
        job = await pool.enqueue_job("crawl_batch", "manual")
    except HTTPException:
        raise
    except Exception as exc:
        log.error("enqueue_failed", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Очередь недоступна: {exc}") from exc

    if job is None:
        # arq вернул None: задача с таким job_id уже стоит в очереди
        return JSONResponse({"status": "duplicate", "job_id": None})
    return JSONResponse({"status": "queued", "job_id": job.job_id})


# --------------------------------------------------------------------------- #
# Логи переписки с нейросетью (deliverable задания)
# --------------------------------------------------------------------------- #

# Страница собрана строковым шаблоном: список файлов в templates/ фиксирован контрактом,
# а base.html подтягивается штатным загрузчиком окружения.
LLM_LOGS_TEMPLATE = """
{% extends "base.html" %}
{% block title %}Логи LLM — MetaPulse{% endblock %}
{% block content %}
<h1 class="page-title">Логи переписки с нейросетью</h1>
<p class="muted">Каталог: <code>{{ log_dir }}</code>. Каждая строка файла — один вызов
  (запрос, ответ, usage, задержка) в формате JSONL.</p>
{% if files %}
<table class="table">
  <thead><tr><th>Файл</th><th>Строк</th><th>Размер</th><th>Изменён</th><th></th></tr></thead>
  <tbody>
  {% for f in files %}
    <tr>
      <td><a href="/llm-logs/{{ f.name }}">{{ f.name }}</a></td>
      <td class="num">{{ f.lines | int_ru }}</td>
      <td class="num">{{ f.size_kb }} КБ</td>
      <td>{{ f.mtime | date_ru }}</td>
      <td><a class="btn btn-sm" href="/llm-logs/{{ f.name }}?download=1">Скачать</a></td>
    </tr>
  {% endfor %}
  </tbody>
</table>
{% else %}
<div class="empty">
  <p>Пока ни одного файла: LLM ещё не вызывалась либо не задан ключ Anthropic.</p>
</div>
{% endif %}
{% endblock %}
"""


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


@router.get("/llm-logs", response_class=HTMLResponse, summary="Логи LLM: список файлов")
async def llm_logs(request: Request) -> HTMLResponse:
    log_dir = Path(settings.llm_log_dir)
    files: list[dict[str, Any]] = []
    if log_dir.is_dir():
        for path in sorted(log_dir.glob("*.jsonl"), reverse=True):
            stat = path.stat()
            files.append(
                {
                    "name": path.name,
                    "size_kb": round(stat.st_size / 1024, 1),
                    "lines": _count_lines(path),
                    "mtime": dt.datetime.fromtimestamp(stat.st_mtime, tz=settings.tz),
                }
            )
    html = templates.env.from_string(LLM_LOGS_TEMPLATE).render(
        request=request, files=files, log_dir=str(log_dir)
    )
    return HTMLResponse(html)


@router.get("/llm-logs/{name}", summary="Логи LLM: файл")
async def llm_log_file(name: str, download: int = 0) -> FileResponse:
    log_dir = Path(settings.llm_log_dir).resolve()
    if not LOG_NAME_RE.match(name):
        raise HTTPException(status_code=404, detail="Файл не найден")
    path = (log_dir / name).resolve()
    # Двойная защита: и проверка имени, и проверка, что путь не вылез из каталога логов
    if path.parent != log_dir or not path.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")
    headers = {"Content-Disposition": f'attachment; filename="{name}"'} if download else None
    return FileResponse(
        path,
        media_type="application/x-ndjson" if download else "text/plain; charset=utf-8",
        headers=headers,
    )
