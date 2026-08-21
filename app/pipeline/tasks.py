"""Задача обхода: раз в час берём батч игр, обогащаем и резюмируем.

Главный инвариант — падение на одной игре не должно ронять весь прогон: каждая игра
обрабатывается в своей транзакции, ошибка попадает в ленту событий и счётчик errors,
цикл идёт дальше. Отметка в daily_seen ставится только после успешной обработки,
поэтому упавшая игра честно вернётся в выборку следующим заходом.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any
from uuid import uuid4

import structlog
from redis.asyncio import Redis, from_url
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Game, LlmCall, PipelineRun
from app.db.session import SessionLocal, session_scope
from app.llm.client import LlmDisabled
from app.llm.embeddings import embed_games, recompute_similar
from app.llm.summarize import summarize_game
from app.metacritic.client import MetacriticClient
from app.pipeline.events import EventBus, default_worker_name
from app.pipeline.repository import (
    game_id_by_mc_id,
    upsert_game,
    upsert_platforms,
    upsert_reviews,
)
from app.pipeline.selection import bump_processed, mark_seen, select_batch
from app.youtube.service import process_letsplay

log = structlog.get_logger(__name__)

LOCK_KEY = "metapulse:lock:crawl"
LOCK_TTL_S = 3600  # чуть меньше часового слота: зависший воркер не заблокирует следующий

# Снимаем лок только если он всё ещё наш: за час TTL мог истечь и его перехватил сосед
_RELEASE_LOCK_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


def _empty_stats() -> dict[str, int]:
    return {
        "selected": 0,
        "new": 0,
        "updated": 0,
        "reviews_fetched": 0,
        "summaries_written": 0,
        "summaries_deferred": 0,
        "llm_calls": 0,
        "letsplays": 0,
        "letsplays_deferred": 0,
        "errors": 0,
    }


def _redis_from_ctx(ctx: dict[str, Any] | None) -> tuple[Redis, bool]:
    """Соединение arq переиспользуем; вне воркера открываем своё."""
    redis = (ctx or {}).get("redis")
    if redis is not None:
        return redis, False
    return from_url(settings.redis_url), True


async def crawl_batch(ctx: dict[str, Any] | None = None, trigger: str = "cron") -> dict[str, Any]:
    """Полный цикл обхода. Возвращает статистику прогона."""
    redis, owns_redis = _redis_from_ctx(ctx)
    worker = (ctx or {}).get("worker_name") or default_worker_name()
    bus = EventBus(redis, worker=worker)
    token = uuid4().hex

    acquired = await redis.set(LOCK_KEY, token, nx=True, ex=LOCK_TTL_S)
    if not acquired:
        async with session_scope() as session:
            await bus.publish(
                session,
                None,
                stage="lock",
                level="warning",
                message="Обход пропущен: предыдущий ещё не закончился",
                payload={"trigger": trigger},
            )
        log.warning("crawl.locked", trigger=trigger)
        if owns_redis:
            await redis.aclose()
        return {"skipped": "locked"}

    try:
        return await _run_batch(bus, trigger, worker)
    finally:
        try:
            await redis.eval(_RELEASE_LOCK_LUA, 1, LOCK_KEY, token)
        except Exception as exc:  # лок всё равно протухнет по TTL
            log.warning("crawl.unlock_failed", error=str(exc))
        if owns_redis:
            await redis.aclose()


async def _run_batch(bus: EventBus, trigger: str, worker: str) -> dict[str, Any]:
    stats = _empty_stats()
    started_at = dt.datetime.now(dt.UTC)
    day = dt.datetime.now(settings.tz).date()
    status = "ok"
    error_text: str | None = None
    processed_ids: list[int] = []
    budget = _LlmBudget(settings.llm_budget_seconds)

    async with SessionLocal() as session:
        run = PipelineRun(trigger=trigger, status="running", started_at=started_at)
        session.add(run)
        await session.commit()
        run_id: int = run.id

        await bus.heartbeat(worker, {"run_id": run_id, "stage": "start", "trigger": trigger})

        try:
            await bus.publish(session, run_id, stage="start", message=f"Старт обхода ({trigger})")
            await session.commit()

            async with MetacriticClient() as client:
                selection = await select_batch(session, client, day, settings.batch_size)
                await session.commit()
                stats["selected"] = len(selection.items)

                await bus.publish(
                    session,
                    run_id,
                    stage="select",
                    message=(
                        f"Отобрано игр: {len(selection.items)} "
                        f"(источник: {selection.phase}, "
                        f"дальше {selection.next_phase} с offset {selection.next_offset})"
                    ),
                    payload={
                        "day": str(day),
                        "phase": selection.phase,
                        "next_phase": selection.next_phase,
                        "next_offset": selection.next_offset,
                        "pages_scanned": selection.pages_scanned,
                    },
                )
                await session.commit()

                for index, item in enumerate(selection.items, start=1):
                    await bus.heartbeat(
                        worker,
                        {
                            "run_id": run_id,
                            "stage": "game",
                            "trigger": trigger,
                            "processed": index - 1,
                            "total": len(selection.items),
                            "current": item.slug,
                        },
                    )
                    try:
                        game_id = await _process_game(
                            session, client, bus, run_id, day, item, stats, budget
                        )
                        processed_ids.append(game_id)
                    except Exception as exc:
                        await session.rollback()
                        stats["errors"] += 1
                        await bus.publish(
                            session,
                            run_id,
                            stage="game",
                            level="error",
                            message=f"Игра {item.slug}: {exc}",
                            payload={"slug": item.slug, "error": str(exc)},
                        )
                        await session.commit()
                        log.warning("crawl.game_failed", slug=item.slug, error=str(exc))

            await _embed_batch(session, bus, run_id, processed_ids, stats)
            await _recompute_similar(session, bus, run_id, stats)
            await _run_letsplays(session, bus, run_id, processed_ids, stats, worker, trigger)

        except Exception as exc:  # сбой самого обхода, а не отдельной игры
            await session.rollback()
            status = "error"
            error_text = str(exc)
            stats["errors"] += 1
            log.exception("crawl.failed", trigger=trigger)
            await bus.publish(
                session, run_id, stage="finish", level="error", message=f"Обход упал: {exc}"
            )
            await session.commit()

        stats["summaries_deferred"] = budget.skipped
        if status != "error":
            status = "partial" if stats["errors"] else "ok"

        stats["llm_calls"] = int(
            await session.scalar(
                select(func.count()).select_from(LlmCall).where(LlmCall.ts >= started_at)
            )
            or 0
        )

        run = await session.get(PipelineRun, run_id, populate_existing=True)
        if run is not None:
            run.status = status
            run.finished_at = dt.datetime.now(dt.UTC)
            run.stats = stats
            run.error = error_text
        await bus.publish(
            session,
            run_id,
            stage="finish",
            level="error" if status == "error" else "info",
            message=(
                f"Обход завершён ({status}): обработано {len(processed_ids)} "
                f"из {stats['selected']}, ошибок {stats['errors']}"
            ),
            payload=stats,
        )
        await session.commit()

    await bus.heartbeat(worker, {"run_id": run_id, "stage": "idle", "trigger": trigger})
    log.info("crawl.done", run_id=run_id, status=status, **stats)
    return {"run_id": run_id, "status": status, **stats}


class _LlmBudget:
    """Потолок времени, которое обход тратит на резюме.

    Сбор данных быстрый, а генерация на локальной модели — минуты на игру. Без потолка
    обход из двадцати игр вылезал бы за часовой слот, и следующий заход упирался бы в лок.
    Исчерпав бюджет, прогон дособирает игры без резюме: они видны на витрине сразу, а
    текст догонит на следующих заходах — `input_hash` не даст пересчитать уже готовое.
    """

    def __init__(self, seconds: int) -> None:
        self._deadline = time.monotonic() + seconds if seconds > 0 else None
        self.skipped = 0

    @property
    def exhausted(self) -> bool:
        if self._deadline is None:
            return False
        return time.monotonic() >= self._deadline

    def note_skip(self) -> None:
        self.skipped += 1


async def _process_game(
    session: AsyncSession,
    client: MetacriticClient,
    bus: EventBus,
    run_id: int,
    day: dt.date,
    item: Any,
    stats: dict[str, int],
    budget: _LlmBudget,
) -> int:
    """Одна игра: деталка -> платформы -> отзывы -> резюме -> эмбеддинг -> daily_seen."""
    detail = await client.get_game(item.slug)

    existed = await game_id_by_mc_id(session, detail.mc_id)
    game = await upsert_game(session, detail)
    stats["updated" if existed else "new"] += 1

    userscores: dict[str, Any] = {}
    for platform in detail.platforms:
        try:
            # userscore живёт только на path-эндпоинте платформы; query-фильтр молча
            # вернул бы данные lead-платформы, поэтому запрашиваем каждую отдельно
            userscores[platform.slug] = await client.get_platform_userscore(
                detail.slug, platform.slug
            )
        except Exception as exc:
            log.warning(
                "crawl.userscore_failed", slug=detail.slug, platform=platform.slug, error=str(exc)
            )
    await upsert_platforms(session, game, detail.platforms, userscores)

    critic = await client.get_critic_reviews(detail.slug, settings.mc_critic_reviews_max)
    user = await client.get_user_reviews(detail.slug, settings.mc_user_reviews_max)
    stats["reviews_fetched"] += len(critic) + len(user)
    new_reviews = await upsert_reviews(session, game, list(critic) + list(user))
    await session.commit()

    if budget.exhausted:
        budget.note_skip()
        log.info("crawl.summary_deferred", slug=game.slug)
    else:
        for kind in ("critic", "user"):
            if await _safe_summary(session, game, kind, stats):
                stats["summaries_written"] += 1

    # Эмбеддинг считается не здесь, а одной пачкой после цикла: см. _embed_batch
    await mark_seen(session, day, item.mc_id, game_id=game.id, run_id=run_id)
    await bump_processed(session, day, 1)
    await bus.publish(
        session,
        run_id,
        stage="game",
        game_id=game.id,
        message=f"Готово: {game.title}",
        payload={
            "slug": game.slug,
            "new": not existed,
            "reviews_new": new_reviews,
            "platforms": len(detail.platforms),
        },
    )
    await session.commit()
    return game.id


async def _safe_summary(
    session: AsyncSession, game: Game, kind: str, stats: dict[str, int]
) -> bool:
    """Резюме не должно ронять игру: без ключа LLM карточка всё равно наполняется."""
    try:
        summary = await summarize_game(session, game, kind)
    except LlmDisabled:
        return False
    except Exception as exc:
        stats["errors"] += 1
        log.warning("crawl.summary_failed", slug=game.slug, kind=kind, error=str(exc))
        await session.rollback()
        return False
    await session.commit()
    return summary is not None


async def _embed_batch(
    session: AsyncSession, bus: EventBus, run_id: int, game_ids: list[int], stats: dict[str, int]
) -> None:
    """Эмбеддинги всего батча одним вызовом — иначе локальная модель тратит время
    на выгрузку и загрузку весов между каждым резюме и каждым эмбеддингом."""
    if not game_ids:
        return
    try:
        games = list((await session.scalars(select(Game).where(Game.id.in_(game_ids)))).all())
        count = await embed_games(session, games)
        await session.commit()
        if count:
            await bus.publish(
                session, run_id, stage="embedding", message=f"Эмбеддинги посчитаны: {count}"
            )
            await session.commit()
    except LlmDisabled:
        pass
    except Exception as exc:
        await session.rollback()
        stats["errors"] += 1
        log.warning("crawl.embeddings_failed", error=str(exc))


async def _recompute_similar(
    session: AsyncSession, bus: EventBus, run_id: int, stats: dict[str, int]
) -> None:
    try:
        count = await recompute_similar(session)
        await session.commit()
        await bus.publish(
            session, run_id, stage="similar", message=f"Похожие игры пересчитаны: {count}"
        )
        await session.commit()
    except Exception as exc:
        await session.rollback()
        stats["errors"] += 1
        log.warning("crawl.similar_failed", error=str(exc))
        await bus.publish(
            session, run_id, stage="similar", level="error", message=f"Похожие игры: {exc}"
        )
        await session.commit()


async def _run_letsplays(
    session: AsyncSession,
    bus: EventBus,
    run_id: int,
    game_ids: list[int],
    stats: dict[str, int],
    worker: str,
    trigger: str,
) -> None:
    if not settings.youtube_enabled or not game_ids:
        return

    # Разбор летсплея — это транскрипт плюс генерация, то есть минуты на игру. Свой
    # потолок нужен отдельно от резюме: иначе на двадцати играх прогон упирался в
    # job_timeout и умирал, не дойдя до конца.
    budget = _LlmBudget(settings.letsplay_budget_seconds)

    for index, game_id in enumerate(game_ids, start=1):
        if budget.exhausted:
            stats["letsplays_deferred"] = len(game_ids) - index + 1
            log.info("crawl.letsplays_deferred", remaining=stats["letsplays_deferred"])
            break
        await bus.heartbeat(
            worker,
            {
                "run_id": run_id,
                "stage": "letsplay",
                "trigger": trigger,
                "processed": index - 1,
                "total": len(game_ids),
            },
        )
        game = await session.get(Game, game_id)
        if game is None:
            continue
        try:
            row = await process_letsplay(session, game)
            await session.commit()
            # Считаем только доведённые до заключения: без ключа YouTube строка тоже
            # создаётся (с error), и такой счётчик в мониторинге вводил бы в заблуждение
            if row is not None and row.conclusion:
                stats["letsplays"] += 1
        except Exception as exc:
            await session.rollback()
            stats["errors"] += 1
            log.warning("crawl.letsplay_failed", game_id=game_id, error=str(exc))
            await bus.publish(
                session,
                run_id,
                stage="letsplay",
                level="error",
                game_id=game_id,
                message=f"Летсплей для {game.slug}: {exc}",
            )
            await session.commit()


async def startup(ctx: dict[str, Any]) -> None:
    """Догоняющий запуск: если слот пропущен (простой контейнера), не ждём следующего часа."""
    ctx["worker_name"] = default_worker_name()
    bus = EventBus(ctx.get("redis"), worker=ctx["worker_name"])
    ctx["event_bus"] = bus
    await bus.heartbeat(ctx["worker_name"], {"stage": "starting"})

    await _close_orphan_runs()

    if not settings.catch_up_on_start:
        return

    since = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)
    async with session_scope() as session:
        recent = await session.scalar(
            select(func.count())
            .select_from(PipelineRun)
            .where(PipelineRun.started_at >= since, PipelineRun.status != "error")
        )
    if recent:
        log.info("startup.catchup_skipped", recent_runs=int(recent))
        return

    redis = ctx.get("redis")
    if redis is None:  # pragma: no cover — arq всегда кладёт соединение в ctx
        log.warning("startup.no_redis")
        return
    now = dt.datetime.now(dt.UTC)
    await redis.enqueue_job("crawl_batch", "catchup", _job_id=f"catchup:{now:%Y%m%d%H}")
    log.info("startup.catchup_enqueued")


async def _close_orphan_runs() -> None:
    """Закрыть прогоны, оборванные падением или рестартом контейнера.

    Такой прогон навсегда остался бы в статусе running и висел бы в мониторинге как
    «идёт обработка», хотя выполнять его уже некому: при старте воркер — единственный,
    кто мог его вести.
    """
    async with session_scope() as session:
        orphans = list(
            (
                await session.scalars(
                    select(PipelineRun).where(PipelineRun.status == "running")
                )
            ).all()
        )
        for run in orphans:
            run.status = "interrupted"
            run.finished_at = dt.datetime.now(dt.UTC)
            run.error = "Прогон оборван: воркер перезапустился"
    if orphans:
        log.info("startup.orphan_runs_closed", count=len(orphans))


async def shutdown(ctx: dict[str, Any]) -> None:
    bus = ctx.get("event_bus")
    if bus is not None:
        await bus.close()
    log.info("worker.shutdown")
