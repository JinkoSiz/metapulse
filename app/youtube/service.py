"""Оркестрация доп. части 1: летсплей -> транскрипт -> заключение LLM.

Контракт (docs/CONTRACTS.md, п.3): любой сбой на любом шаге приводит к строке в
`letsplays` с заполненным `error` и `conclusion=None`, но НЕ к исключению наружу —
карточка игры обязана рендериться и без летсплея.
"""

from __future__ import annotations

import datetime as dt
import inspect
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Game, LetsPlay
from app.youtube.search import VideoCandidate, pick_best_letsplay
from app.youtube.transcript import fetch_transcript

log = structlog.get_logger(__name__)

MAX_ERROR_CHARS = 1000
# Через сколько повторять неудавшийся разбор: достаточно часто, чтобы починка
# конфигурации подхватилась в тот же день, и достаточно редко, чтобы не долбить
# YouTube на каждом часовом обходе.
RETRY_FAILED_AFTER_HOURS = 6

# Запасные промпт и схема на случай, если app/llm/prompts.py не отдаёт LETSPLAY_*.
FALLBACK_LETSPLAY_SYSTEM = (
    "Ты игровой обозреватель. По транскрипту летсплея опиши, каково играть в эту игру "
    "на практике: темп, ощущения от управления и боя, качество подачи, что раздражает. "
    "Опирайся только на транскрипт, не выдумывай факты и оценки. Если транскрипт "
    "автоматический и местами бессвязный — игнорируй шум. Отвечай по-русски, "
    "3-5 предложений, без маркированных списков и без пересказа сюжета."
)

FALLBACK_LETSPLAY_USER_TEMPLATE = (
    "Игра: {game_title}\n"
    "Ролик: {video_title}\n"
    "Канал: {channel}\n\n"
    "Транскрипт летсплея:\n{transcript}"
)

FALLBACK_LETSPLAY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "conclusion": {
            "type": "string",
            "description": "Заключение о геймплее на русском языке, 3-5 предложений.",
        }
    },
    "required": ["conclusion"],
    "additionalProperties": False,
}

_CONCLUSION_KEYS = ("conclusion", "tl_dr", "summary", "text")


class LetsPlayUnavailable(Exception):
    """Ожидаемый сбой шага: пишется в letsplays.error, наружу не выходит."""


async def process_letsplay(session: AsyncSession, game: Game) -> LetsPlay:
    """Полный цикл летсплея для игры; см. docs/CONTRACTS.md, п.3.

    Кэш: если `letsplays.fetched_at` моложе `settings.letsplay_ttl_days`, ничего
    не делаем и возвращаем существующую строку — иначе на 20 играх в час сгорит
    суточная квота search.list (100 вызовов).

    Строку только флашит; коммит — на вызывающем коде (pipeline работает в session_scope).
    """
    row = await _load(session, game.id)
    if row is not None and _is_fresh(row):
        log.info("letsplay.cache_hit", game_id=game.id, video_id=row.video_id)
        return row

    fields: dict[str, Any] = {
        "video_id": None,
        "video_url": None,
        "video_title": None,
        "channel": None,
        "view_count": None,
        "duration_sec": None,
        "transcript_source": "none",
        "transcript_len": None,
        "conclusion": None,
        "model": None,
        "error": None,
    }

    try:
        candidate = await _find_video(game)
        fields.update(
            video_id=candidate.video_id,
            video_url=candidate.url,
            video_title=candidate.title,
            channel=candidate.channel,
            view_count=candidate.view_count,
            duration_sec=candidate.duration_sec,
        )

        transcript = await fetch_transcript(candidate.video_id)
        if transcript is None:
            raise LetsPlayUnavailable(
                "Транскрипт недоступен: субтитров нет, yt-dlp тоже ничего не вернул"
            )
        text, source = transcript
        fields["transcript_source"] = source
        fields["transcript_len"] = len(text)

        fields["conclusion"] = await _make_conclusion(session, game, candidate, text)
        fields["model"] = settings.llm_model
        log.info(
            "letsplay.done",
            game_id=game.id,
            video_id=candidate.video_id,
            transcript_source=source,
            transcript_len=len(text),
        )
    except LetsPlayUnavailable as exc:
        fields["error"] = str(exc)[:MAX_ERROR_CHARS]
        log.warning("letsplay.skipped", game_id=game.id, reason=str(exc))
    except Exception as exc:  # noqa: BLE001 — сбой летсплея не должен ронять обход игры
        fields["error"] = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
        log.warning(
            "letsplay.failed",
            game_id=game.id,
            error=f"{type(exc).__name__}: {exc}",
            exc_info=True,
        )

    return await _upsert(session, game.id, fields)


async def _find_video(game: Game) -> VideoCandidate:
    if not settings.youtube_enabled:
        raise LetsPlayUnavailable("Модуль YouTube отключён настройкой youtube_enabled")
    # Ключ нужен только официальному Data API; бэкенд yt-dlp работает без него
    if settings.youtube_search_backend != "yt-dlp" and not (settings.youtube_api_key or "").strip():
        raise LetsPlayUnavailable("Не задан YOUTUBE_API_KEY — поиск летсплея пропущен")

    candidate = await pick_best_letsplay(game.title)
    if candidate is None:
        raise LetsPlayUnavailable(
            "Подходящий летсплей не найден: выдача пуста или все ролики "
            "короче 8 минут либо опознаны как трейлер/обзор"
        )
    return candidate


def _fit_transcript(transcript: str) -> str:
    """Подгоняет транскрипт под возможности активной модели.

    Четырёхчасовое прохождение даёт сотню тысяч символов. Облачная модель прочитает
    их за секунды, локальная на CPU — за четверть часа, поэтому ей достаётся начало
    ролика: там блогер рассказывает о впечатлениях, а дальше идёт игровой процесс.
    """
    if settings.llm_provider != "ollama":
        return transcript
    limit = settings.ollama_transcript_chars
    if len(transcript) <= limit:
        return transcript
    return transcript[:limit].rsplit(" ", 1)[0]


async def _make_conclusion(
    session: AsyncSession,
    game: Game,
    candidate: VideoCandidate,
    transcript: str,
) -> str:
    llm_client_cls, llm_disabled_cls = _import_llm()
    system, user_template, schema = _letsplay_prompt()
    user = user_template.format(
        game_title=game.title,
        video_title=candidate.title,
        channel=candidate.channel,
        transcript=_fit_transcript(transcript),
    )

    client = _build_llm_client(llm_client_cls, session)
    try:
        if hasattr(client, "__aenter__"):
            async with client as opened:
                payload = await opened.complete_structured(
                    purpose="letsplay_conclusion",
                    system=system,
                    user=user,
                    schema=schema,
                    game_id=game.id,
                )
        else:
            payload = await client.complete_structured(
                purpose="letsplay_conclusion",
                system=system,
                user=user,
                schema=schema,
                game_id=game.id,
            )
    except llm_disabled_cls as exc:
        raise LetsPlayUnavailable(f"LLM отключена: {exc}") from exc

    conclusion = _extract_conclusion(payload)
    if not conclusion:
        raise LetsPlayUnavailable("LLM вернула пустое заключение по летсплею")
    return conclusion


def _import_llm() -> tuple[Any, type[BaseException]]:
    """Ленивый импорт LLM-контура: он единственная точка вызова Anthropic во всём коде."""
    try:
        from app.llm.client import LlmClient, LlmDisabled
    except ImportError as exc:
        raise LetsPlayUnavailable(f"LLM-контур недоступен: {exc}") from exc
    return LlmClient, LlmDisabled


def _build_llm_client(llm_client_cls: Any, session: AsyncSession) -> Any:
    """Сессия нужна LlmClient только для индекса llm_calls и передаётся, если он её принимает."""
    try:
        params = inspect.signature(llm_client_cls).parameters
    except (TypeError, ValueError):
        params = {}
    if "session" in params:
        return llm_client_cls(session=session)
    return llm_client_cls()


def _letsplay_prompt() -> tuple[str, str, dict[str, Any]]:
    """Промпт и схема из app/llm/prompts.py, иначе локальные запасные константы."""
    try:
        from app.llm import prompts
    except ImportError:
        return (
            FALLBACK_LETSPLAY_SYSTEM,
            FALLBACK_LETSPLAY_USER_TEMPLATE,
            FALLBACK_LETSPLAY_SCHEMA,
        )

    system = getattr(prompts, "LETSPLAY_SYSTEM", FALLBACK_LETSPLAY_SYSTEM)
    user_template = (
        getattr(prompts, "LETSPLAY_USER_TEMPLATE", None)
        or getattr(prompts, "LETSPLAY_USER", None)
        or FALLBACK_LETSPLAY_USER_TEMPLATE
    )
    schema = getattr(prompts, "LETSPLAY_SCHEMA", FALLBACK_LETSPLAY_SCHEMA)
    return system, user_template, schema


def _extract_conclusion(payload: Any) -> str | None:
    if isinstance(payload, str):
        return payload.strip() or None
    if not isinstance(payload, dict):
        return None
    for key in _CONCLUSION_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in payload.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


async def _load(session: AsyncSession, game_id: int) -> LetsPlay | None:
    result = await session.execute(select(LetsPlay).where(LetsPlay.game_id == game_id))
    return result.scalar_one_or_none()


def _is_fresh(row: LetsPlay, *, now: dt.datetime | None = None) -> bool:
    """Кэшируется только удавшийся разбор.

    Неудачу кэшировать на неделю нельзя: причина обычно внешняя и устранимая — не
    настроен прокси, не было ключа, YouTube отдал ошибку. Иначе после починки
    конфигурации летсплеи не появились бы ещё несколько дней.
    """
    if row.fetched_at is None:
        return False
    now = now or dt.datetime.now(dt.UTC)
    fetched = row.fetched_at
    if fetched.tzinfo is None:  # на случай naive-значения из старой записи
        fetched = fetched.replace(tzinfo=dt.UTC)
    age = now - fetched
    if not row.conclusion:
        return age < dt.timedelta(hours=RETRY_FAILED_AFTER_HOURS)
    return age < dt.timedelta(days=settings.letsplay_ttl_days)


async def _upsert(session: AsyncSession, game_id: int, fields: dict[str, Any]) -> LetsPlay:
    row = await _load(session, game_id)
    if row is None:
        row = LetsPlay(game_id=game_id)
        session.add(row)
    for key, value in fields.items():
        setattr(row, key, value)
    row.fetched_at = dt.datetime.now(dt.UTC)
    await session.flush()
    return row
