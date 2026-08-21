"""JSONL-логирование переписки с нейросетью — сдаваемый артефакт задания.

Сети здесь нет: подменяется единственный метод, который ходит наружу.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.config import settings
from app.llm.client import LlmClient, LlmDisabled, LlmError

SCHEMA = {
    "type": "object",
    "properties": {
        "likes": {"type": "array", "items": {"type": "string"}},
        "dislikes": {"type": "array", "items": {"type": "string"}},
        "tl_dr": {"type": "string"},
    },
    "required": ["likes", "dislikes", "tl_dr"],
}

ANSWER = {
    "likes": ["боевая система", "музыка"],
    "dislikes": ["оптимизация на PC"],
    "tl_dr": "Игрокам нравится бой, ругают производительность.",
}


class _Block:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Usage:
    def model_dump(self, **_: Any) -> dict[str, int]:
        return {"input_tokens": 1234, "output_tokens": 56}


class _Response:
    stop_reason = "end_turn"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.content = [_Block(json.dumps(payload, ensure_ascii=False))]
        self.usage = _Usage()

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {"stop_reason": self.stop_reason, "content": [{"type": "text"}]}


class _Messages:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class _FakeAnthropic:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.messages = _Messages(response, error)

    async def close(self) -> None:
        return None


@pytest.fixture
def log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Логи пишем во временный каталог; бэкенд фиксируем облачный."""
    target = tmp_path / "llm"
    monkeypatch.setattr(settings, "llm_log_dir", target)
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "embedding_provider", "voyage")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_enabled", True)
    return target


@pytest.fixture
def ollama_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """То же, но для локального бэкенда."""
    target = tmp_path / "llm-ollama"
    monkeypatch.setattr(settings, "llm_log_dir", target)
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "embedding_provider", "ollama")
    monkeypatch.setattr(settings, "llm_enabled", True)
    return target


def read_lines(log_dir: Path) -> list[dict[str, Any]]:
    files = sorted(log_dir.glob("*.jsonl"))
    assert files, "JSONL-файл не создан"
    raw = files[0].read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


async def test_successful_call_writes_one_line(
    log_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeAnthropic(_Response(ANSWER))
    monkeypatch.setattr(LlmClient, "_messages_client", lambda self: fake)

    client = LlmClient()
    result = await client.complete_structured(
        purpose="critic_summary",
        system="Отвечай по-русски",
        user="Отзывы критиков…",
        schema=SCHEMA,
        game_id=42,
    )

    assert result == ANSWER

    entries = read_lines(log_dir)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["status"] == "ok"
    assert entry["provider"] == "anthropic"
    assert entry["purpose"] == "critic_summary"
    assert entry["game_id"] == 42
    assert entry["model"] == settings.llm_model
    assert entry["usage"]["input_tokens"] == 1234
    assert isinstance(entry["latency_ms"], int)
    # В переписке должен быть виден и запрос целиком, включая схему структурированного вывода
    assert entry["request"]["messages"][0]["content"] == "Отзывы критиков…"
    assert entry["request"]["output_config"]["format"]["schema"] == SCHEMA
    assert client.last_call_id is not None


async def test_cyrillic_is_not_escaped(log_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Логи читает человек: \\uXXXX вместо русского текста сделал бы их бесполезными."""
    fake = _FakeAnthropic(_Response(ANSWER))
    monkeypatch.setattr(LlmClient, "_messages_client", lambda self: fake)

    await LlmClient().complete_structured(
        purpose="user_summary",
        system="система",
        user="Игрокам нравится боёвка",
        schema=SCHEMA,
    )

    raw = sorted(log_dir.glob("*.jsonl"))[0].read_text(encoding="utf-8")
    assert "Игрокам нравится боёвка" in raw
    assert "\\u04" not in raw


async def test_provider_error_is_logged(log_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Неудачный вызов — тоже часть переписки, он обязан попасть в файл."""
    fake = _FakeAnthropic(error=RuntimeError("overloaded_error"))
    monkeypatch.setattr(LlmClient, "_messages_client", lambda self: fake)

    with pytest.raises(LlmError):
        await LlmClient().complete_structured(
            purpose="critic_summary", system="s", user="u", schema=SCHEMA, game_id=7
        )

    entry = read_lines(log_dir)[0]
    assert entry["status"] == "error"
    assert "overloaded_error" in entry["error"]
    assert entry["response"] is None
    assert entry["game_id"] == 7


async def test_ollama_call_is_logged(
    ollama_log_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Локальный бэкенд пишет ту же переписку: провайдер и токены видны в JSONL."""
    captured: dict[str, Any] = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "response": json.dumps(ANSWER, ensure_ascii=False),
                "prompt_eval_count": 2066,
                "prompt_eval_duration": 92_500_000_000,
                "eval_count": 160,
                "eval_duration": 42_600_000_000,
            }

    class _Client:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def post(self, url: str, json: dict[str, Any]) -> _Resp:  # noqa: A002
            captured["url"] = url
            captured["body"] = json
            return _Resp()

    monkeypatch.setattr("app.llm.client.httpx.AsyncClient", _Client)

    result = await LlmClient().complete_structured(
        purpose="user_summary", system="Отвечай по-русски", user="Отзывы игроков…",
        schema=SCHEMA, game_id=11,
    )

    assert result == ANSWER
    assert captured["url"].endswith("/api/generate")
    # Схема уходит в поле format — именно оно гарантирует форму ответа у Ollama
    assert captured["body"]["format"] == SCHEMA
    assert captured["body"]["stream"] is False

    entry = read_lines(ollama_log_dir)[0]
    assert entry["provider"] == "ollama"
    assert entry["model"] == settings.ollama_model
    assert entry["usage"]["input_tokens"] == 2066
    assert entry["usage"]["output_tokens"] == 160
    assert entry["usage"]["prompt_eval_ms"] == 92_500
    assert entry["status"] == "ok"


async def test_disabled_llm_raises_and_writes_nothing(
    log_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Без ключа сети не было — писать в переписку нечего, но пайплайн это переживает."""
    monkeypatch.setattr(settings, "anthropic_api_key", None)

    with pytest.raises(LlmDisabled):
        await LlmClient().complete_structured(
            purpose="critic_summary", system="s", user="u", schema=SCHEMA
        )

    assert not list(log_dir.glob("*.jsonl"))


async def test_embeddings_are_logged_without_vector_dump(
    log_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Эмбеддинги — тоже обращение к нейросети, но векторы в лог не выгружаются."""
    monkeypatch.setattr(settings, "voyage_api_key", "test-key")

    async def fake_post(body: dict[str, Any]) -> dict[str, Any]:
        return {
            "data": [{"index": 0, "embedding": [0.1] * 1024}],
            "usage": {"total_tokens": 21},
        }

    monkeypatch.setattr("app.llm.client._voyage_post", fake_post)

    vectors = await LlmClient().embed(["Clair Obscur: Expedition 33"], game_id=5)
    assert len(vectors) == 1 and len(vectors[0]) == 1024

    entry = read_lines(log_dir)[0]
    assert entry["provider"] == "voyage"
    assert entry["purpose"] == "embedding"
    assert entry["status"] == "ok"
    assert entry["response"]["dimensions"] == 1024
    assert len(entry["response"]["first_vector_head"]) == 8
