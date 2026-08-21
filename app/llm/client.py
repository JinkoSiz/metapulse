"""Единственная точка обращения к Anthropic и Voyage во всём проекте.

Требование задания «вся переписка с нейросетью, можно raw JSONL-файлами» закрыто
архитектурно: любой сетевой вызов к LLM проходит через `LlmClient`, а каждый вызов
(успех, ошибка, отказ модели, эмбеддинги) оставляет ровно одну строку в
`{settings.llm_log_dir}/YYYY-MM-DD.jsonl`. Ни один другой модуль не импортирует
`anthropic` и не ходит на api.voyageai.com напрямую.

Иерархия исключений (расширение контракта, в котором назван только `LlmDisabled`):

* `LlmError` — базовый класс всех сбоев контура; ловите его, если не важна причина;
* `LlmDisabled` — ключа нет либо `llm_enabled=false`: сети не было, JSONL не пишется;
* `LlmRefusal` — провайдер вернул `stop_reason="refusal"`; строка в JSONL пишется
  со `status="refusal"`, чтобы отказ остался в переписке.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import structlog
from anthropic import AsyncAnthropic
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.db.models import LlmCall

log = structlog.get_logger(__name__)

PURPOSES = ("critic_summary", "user_summary", "letsplay_conclusion", "embedding")

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"

ANTHROPIC_TIMEOUT_S = 120.0
VOYAGE_TIMEOUT_S = 60.0

# Одна строка на вызов: пишем под общим локом, чтобы параллельные задачи воркера
# не перемешали свои строки внутри одного файла.
_file_lock = asyncio.Lock()


class LlmError(Exception):
    """Базовая ошибка LLM-контура."""


class LlmDisabled(LlmError):
    """Нет ключа или контур выключен настройками — вызывающий обязан это пережить."""


class LlmRefusal(LlmError):
    """Провайдер отказался отвечать (`stop_reason="refusal"`)."""


def jsonl_path(moment: dt.datetime | None = None) -> Path:
    """Файл переписки за сутки. Дата берётся в таймзоне приложения."""
    moment = moment or dt.datetime.now(settings.tz)
    return Path(settings.llm_log_dir) / f"{moment.date().isoformat()}.jsonl"


async def append_jsonl(entry: dict[str, Any], *, moment: dt.datetime | None = None) -> Path:
    """Дописывает ровно одну строку в суточный JSONL и возвращает путь к файлу."""
    path = jsonl_path(moment)
    line = json.dumps(entry, ensure_ascii=False, default=str)
    async with _file_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
    return path


async def record_llm_call(
    session: AsyncSession | None,
    entry: dict[str, Any],
    *,
    jsonl_file: str | None = None,
) -> None:
    """Индексная строка в `llm_calls` для UI.

    Первичный артефакт — сам JSONL-файл, поэтому падение вставки никогда не роняет
    вызывающий код: без сессии просто выходим, при ошибке БД пишем warning.
    """
    if session is None:
        return
    usage = entry.get("usage") or {}
    try:
        session.add(
            LlmCall(
                id=uuid.UUID(entry["id"]),
                provider=entry["provider"],
                model=entry["model"],
                purpose=entry["purpose"],
                game_id=entry.get("game_id"),
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                latency_ms=entry.get("latency_ms"),
                status=entry.get("status", "ok"),
                error=entry.get("error"),
                jsonl_file=jsonl_file,
            )
        )
        await session.flush()
    except Exception as exc:  # индекс не критичен — файл уже на диске
        log.warning("llm_call_index_failed", error=str(exc), call_id=entry.get("id"))


class LlmClient:
    """Асинхронный фасад над Anthropic (резюме) и Voyage (эмбеддинги).

    `session` нужна только для индексной таблицы `llm_calls`; транзакцией владеет
    вызывающий, коммита здесь нет. После каждого вызова в `last_call_id` лежит uuid
    строки JSONL — его кладут в `summaries.llm_call_id` (контракт не даёт вернуть
    id через возвращаемое значение, поэтому один экземпляр клиента = одна задача).
    """

    def __init__(self, session: AsyncSession | None = None) -> None:
        self._session = session
        self._anthropic: AsyncAnthropic | None = None
        self.last_call_id: uuid.UUID | None = None

    async def __aenter__(self) -> LlmClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._anthropic is not None:
            await self._anthropic.close()
            self._anthropic = None

    # --- внутреннее ---------------------------------------------------------

    def _messages_client(self) -> AsyncAnthropic:
        if self._anthropic is None:
            self._anthropic = AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                timeout=ANTHROPIC_TIMEOUT_S,
                max_retries=3,
            )
        return self._anthropic

    async def _emit(
        self,
        *,
        provider: str,
        model: str,
        purpose: str,
        game_id: int | None,
        request: dict[str, Any],
        response: Any,
        usage: dict[str, Any] | None,
        latency_ms: int,
        status: str,
        error: str | None,
    ) -> uuid.UUID:
        call_id = uuid.uuid4()
        now = dt.datetime.now(settings.tz)
        entry = {
            "id": str(call_id),
            "ts": now.isoformat(),
            "provider": provider,
            "model": model,
            "purpose": purpose,
            "game_id": game_id,
            "request": request,
            "response": response,
            "usage": usage,
            "latency_ms": latency_ms,
            "status": status,
            "error": error,
        }
        path = await append_jsonl(entry, moment=now)
        await record_llm_call(self._session, entry, jsonl_file=path.name)
        self.last_call_id = call_id
        return call_id

    # --- Anthropic ----------------------------------------------------------

    async def complete_structured(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        schema: dict,
        game_id: int | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Структурированный ответ активного бэкенда. Возвращает распарсенный JSON."""
        if purpose not in PURPOSES:
            log.warning("llm_unknown_purpose", purpose=purpose)
        if not settings.llm_enabled:
            raise LlmDisabled("LLM выключен настройкой llm_enabled=false")

        if settings.llm_provider == "ollama":
            return await self._complete_ollama(
                purpose=purpose,
                system=system,
                user=user,
                schema=schema,
                game_id=game_id,
                max_tokens=max_tokens,
            )
        return await self._complete_anthropic(
            purpose=purpose,
            system=system,
            user=user,
            schema=schema,
            game_id=game_id,
            max_tokens=max_tokens,
        )

    async def _complete_anthropic(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        schema: dict,
        game_id: int | None,
        max_tokens: int | None,
    ) -> dict:
        """Claude со structured outputs (GA, без beta-заголовков)."""
        if not settings.anthropic_api_key:
            # Сети не было — писать в «переписку с нейросетью» нечего.
            raise LlmDisabled("LLM выключен: нет ANTHROPIC_API_KEY")

        model = settings.anthropic_model
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }

        started = time.perf_counter()
        try:
            response = await self._messages_client().messages.create(**request)
        except Exception as exc:
            await self._emit(
                provider="anthropic",
                model=model,
                purpose=purpose,
                game_id=game_id,
                request=request,
                response=None,
                usage=None,
                latency_ms=int((time.perf_counter() - started) * 1000),
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise LlmError(f"Anthropic не ответил: {exc}") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        payload = _model_dump(response)
        usage = _model_dump(getattr(response, "usage", None))

        if getattr(response, "stop_reason", None) == "refusal":
            await self._emit(
                provider="anthropic",
                model=model,
                purpose=purpose,
                game_id=game_id,
                request=request,
                response=payload,
                usage=usage,
                latency_ms=latency_ms,
                status="refusal",
                error="stop_reason=refusal",
            )
            raise LlmRefusal("Модель отказалась отвечать на этот запрос")

        text = "".join(
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        )
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("ожидался JSON-объект")
        except Exception as exc:
            await self._emit(
                provider="anthropic",
                model=model,
                purpose=purpose,
                game_id=game_id,
                request=request,
                response=payload,
                usage=usage,
                latency_ms=latency_ms,
                status="error",
                error=f"не удалось разобрать ответ: {exc}",
            )
            raise LlmError(f"Не удалось разобрать ответ модели: {exc}") from exc

        await self._emit(
            provider="anthropic",
            model=model,
            purpose=purpose,
            game_id=game_id,
            request=request,
            response=payload,
            usage=usage,
            latency_ms=latency_ms,
            status="ok",
            error=None,
        )
        return data

    # --- Ollama -------------------------------------------------------------

    async def _complete_ollama(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        schema: dict,
        game_id: int | None,
        max_tokens: int | None,
    ) -> dict:
        """Локальная модель. `format` принимает JSON Schema и гарантирует форму ответа.

        Таймаут щедрый: на CPU без GPU чтение промпта идёт порядка 20 токенов в секунду,
        и одно резюме занимает минуты — это нормальный режим, а не зависание.
        """
        model = settings.ollama_model
        request: dict[str, Any] = {
            "model": model,
            "system": system,
            "prompt": user,
            "stream": False,
            "format": schema,
            "options": {"num_predict": max_tokens or settings.llm_max_tokens},
        }

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=settings.ollama_timeout_s) as client:
                response = await client.post(
                    f"{settings.ollama_url.rstrip('/')}/api/generate", json=request
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            await self._emit(
                provider="ollama",
                model=model,
                purpose=purpose,
                game_id=game_id,
                request=request,
                response=None,
                usage=None,
                latency_ms=int((time.perf_counter() - started) * 1000),
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise LlmError(f"Ollama не ответила: {exc}") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = {
            "input_tokens": payload.get("prompt_eval_count"),
            "output_tokens": payload.get("eval_count"),
            "prompt_eval_ms": _ns_to_ms(payload.get("prompt_eval_duration")),
            "eval_ms": _ns_to_ms(payload.get("eval_duration")),
        }
        text = payload.get("response") or ""
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("ожидался JSON-объект")
        except Exception as exc:
            await self._emit(
                provider="ollama",
                model=model,
                purpose=purpose,
                game_id=game_id,
                request=request,
                response=payload,
                usage=usage,
                latency_ms=latency_ms,
                status="error",
                error=f"не удалось разобрать ответ: {exc}",
            )
            raise LlmError(f"Не удалось разобрать ответ Ollama: {exc}") from exc

        await self._emit(
            provider="ollama",
            model=model,
            purpose=purpose,
            game_id=game_id,
            request=request,
            response=payload,
            usage=usage,
            latency_ms=latency_ms,
            status="ok",
            error=None,
        )
        return data

    # --- эмбеддинги ---------------------------------------------------------

    async def embed(
        self,
        texts: list[str],
        *,
        game_id: int | None = None,
        input_type: str = "document",
    ) -> list[list[float]]:
        """Эмбеддинги активного бэкенда. Недоступность — `LlmDisabled`, пайплайн это переживает."""
        if not texts:
            return []
        if settings.embedding_provider == "none":
            raise LlmDisabled("Эмбеддинги выключены: embedding_provider=none")
        if settings.embedding_provider == "ollama":
            return await self._embed_ollama(texts, game_id=game_id)
        if not settings.voyage_api_key:
            raise LlmDisabled("Эмбеддинги выключены: нет VOYAGE_API_KEY")

        model = settings.voyage_model
        request: dict[str, Any] = {
            "input": texts,
            "model": model,
            "input_type": input_type,
        }

        started = time.perf_counter()
        try:
            payload = await _voyage_post(request)
        except Exception as exc:
            await self._emit(
                provider="voyage",
                model=model,
                purpose="embedding",
                game_id=game_id,
                request=request,
                response=None,
                usage=None,
                latency_ms=int((time.perf_counter() - started) * 1000),
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise LlmError(f"Voyage не ответил: {exc}") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        try:
            items = sorted(payload["data"], key=lambda row: row.get("index", 0))
            vectors = [[float(x) for x in row["embedding"]] for row in items]
        except Exception as exc:
            await self._emit(
                provider="voyage",
                model=model,
                purpose="embedding",
                game_id=game_id,
                request=request,
                response={"raw_keys": sorted(payload)} if isinstance(payload, dict) else None,
                usage=None,
                latency_ms=latency_ms,
                status="error",
                error=f"неожиданный формат ответа: {exc}",
            )
            raise LlmError(f"Неожиданный формат ответа Voyage: {exc}") from exc

        usage = payload.get("usage") or {}
        # Осознанное решение: полный запрос сохраняем, а векторы — нет. 1024 float на
        # каждый текст превратили бы суточный JSONL в мегабайты нечитаемого мусора,
        # поэтому в response кладём форму ответа и «голову» первого вектора — этого
        # хватает, чтобы убедиться, что эмбеддинг реальный, а не заглушка.
        await self._emit(
            provider="voyage",
            model=model,
            purpose="embedding",
            game_id=game_id,
            request=request,
            response={
                "vectors": len(vectors),
                "dimensions": len(vectors[0]) if vectors else 0,
                "first_vector_head": vectors[0][:8] if vectors else [],
                "note": "векторы целиком не логируются (см. комментарий в app/llm/client.py)",
            },
            usage={"input_tokens": usage.get("total_tokens")} | dict(usage),
            latency_ms=latency_ms,
            status="ok",
            error=None,
        )
        return vectors


    async def _embed_ollama(
        self, texts: list[str], *, game_id: int | None = None
    ) -> list[list[float]]:
        """Локальные эмбеддинги. Чтение текста несопоставимо дешевле генерации,
        поэтому здесь слабый процессор сервера не помеха."""
        model = settings.ollama_embedding_model
        request: dict[str, Any] = {"model": model, "input": texts}

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=settings.ollama_timeout_s) as client:
                response = await client.post(
                    f"{settings.ollama_url.rstrip('/')}/api/embed", json=request
                )
                response.raise_for_status()
                payload = response.json()
            vectors = [[float(x) for x in row] for row in payload["embeddings"]]
        except Exception as exc:
            await self._emit(
                provider="ollama",
                model=model,
                purpose="embedding",
                game_id=game_id,
                request=request,
                response=None,
                usage=None,
                latency_ms=int((time.perf_counter() - started) * 1000),
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise LlmError(f"Ollama не отдала эмбеддинги: {exc}") from exc

        await self._emit(
            provider="ollama",
            model=model,
            purpose="embedding",
            game_id=game_id,
            request=request,
            response={
                "vectors": len(vectors),
                "dimensions": len(vectors[0]) if vectors else 0,
                "first_vector_head": vectors[0][:8] if vectors else [],
                "note": "векторы целиком не логируются (см. комментарий в app/llm/client.py)",
            },
            usage={"input_tokens": payload.get("prompt_eval_count")},
            latency_ms=int((time.perf_counter() - started) * 1000),
            status="ok",
            error=None,
        )
        return vectors


def _ns_to_ms(value: Any) -> int | None:
    """Ollama отдаёт длительности в наносекундах."""
    try:
        return int(value) // 1_000_000
    except (TypeError, ValueError):
        return None


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
)
async def _voyage_post(body: dict[str, Any]) -> dict[str, Any]:
    """POST в Voyage с ретраями. 4xx кроме 429 не ретраим — это ошибка запроса."""
    async with httpx.AsyncClient(timeout=VOYAGE_TIMEOUT_S) as client:
        response = await client.post(
            VOYAGE_URL,
            headers={
                "Authorization": f"Bearer {settings.voyage_api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        if response.status_code >= 400:
            detail = response.text[:500]
            if response.status_code == 429 or response.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"Voyage {response.status_code}: {detail}",
                    request=response.request,
                    response=response,
                )
            raise LlmError(f"Voyage {response.status_code}: {detail}")
        return response.json()


def _model_dump(obj: Any) -> Any:
    """pydantic-модели SDK -> dict; всё остальное отдаём как есть."""
    if obj is None:
        return None
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except TypeError:
            return dump()
    return obj
