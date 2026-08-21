"""Получение текста летсплея: субтитры YouTube с фолбэком на yt-dlp.

Две ветки:

1. `youtube-transcript-api` (v1.x, инстансный API) — сначала ручные субтитры, потом
   автоматические. YouTube блокирует датацентровые IP (RequestBlocked/IpBlocked),
   поэтому обе ветки уважают `settings.youtube_proxy` — без резидентного прокси
   в Docker эта ветка почти всегда падает и работу берёт на себя фолбэк.
2. `yt-dlp` со `--skip-download --write-auto-subs` — качает VTT и парсит его в текст.

Точка расширения (в коде не реализована): третьим фолбэком напрашивается STT —
скачать аудиодорожку тем же yt-dlp и прогнать через Whisper, дав
`transcript_source='stt'` (значение уже заложено в схему БД). Это стоит отдельного
GPU-воркера, поэтому здесь только помечено местом в `fetch_transcript`.

Все вызовы библиотек синхронные и блокирующие — уходят в `asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import html
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path

import structlog

from app.config import settings

log = structlog.get_logger(__name__)

# Порядок важен: сначала оригинальная английская дорожка, потом русская.
LANGUAGES: tuple[str, ...] = ("en", "en-US", "ru")

# Для yt-dlp языки задаются регулярками: en.* ловит en, en-US, en-orig.
YTDLP_SUB_LANGS: list[str] = ["en.*", "ru"]

SUBTITLE_SUFFIXES = (".vtt", ".srt")

_TAG_RE = re.compile(r"<[^>]*>")
_WS_RE = re.compile(r"\s+")
_CUE_INDEX_RE = re.compile(r"^\d+$")
_VTT_META_PREFIXES = ("WEBVTT", "NOTE", "STYLE", "REGION", "Kind:", "Language:", "::cue")


async def fetch_transcript(video_id: str) -> tuple[str, str] | None:
    """Текст летсплея и источник: ('captions' | 'ytdlp').

    Возвращает None, если ни одна ветка не дала текста. Исключений не бросает.
    """
    text = await asyncio.to_thread(_fetch_via_captions, video_id)
    if text:
        return _truncate(text), "captions"

    text = await asyncio.to_thread(_fetch_via_ytdlp, video_id)
    if text:
        return _truncate(text), "ytdlp"

    # Здесь встраивается STT-фолбэк (см. docstring модуля): скачать аудио и
    # распознать его, вернув (text, "stt").
    log.warning("youtube.transcript.unavailable", video_id=video_id)
    return None


def _truncate(text: str) -> str:
    limit = settings.youtube_transcript_max_chars
    if limit and len(text) > limit:
        return text[:limit].rstrip()
    return text


# --- ветка 1: youtube-transcript-api -------------------------------------------------


def _proxy_config():
    """GenericProxyConfig из settings.youtube_proxy либо None."""
    proxy = (settings.youtube_proxy or "").strip()
    if not proxy:
        return None
    from youtube_transcript_api.proxies import GenericProxyConfig

    return GenericProxyConfig(http_url=proxy, https_url=proxy)


def _fetch_via_captions(video_id: str) -> str | None:
    """Субтитры через youtube-transcript-api. Блокирующая, вызывается в to_thread."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi(proxy_config=_proxy_config())
        transcript_list = api.list(video_id)

        transcript = None
        # сначала ручные (они точнее и с пунктуацией), потом автоматические
        for finder in (
            transcript_list.find_manually_created_transcript,
            transcript_list.find_generated_transcript,
        ):
            try:
                transcript = finder(LANGUAGES)
                break
            except Exception:  # noqa: BLE001 — нужного языка нет, пробуем следующий источник
                continue
        if transcript is None:
            transcript = transcript_list.find_transcript(LANGUAGES)

        fetched = transcript.fetch()
        text = _clean_lines(_snippet_texts(fetched))
        if not text:
            return None
        log.info(
            "youtube.transcript.captions_ok",
            video_id=video_id,
            language=getattr(transcript, "language_code", None),
            generated=getattr(transcript, "is_generated", None),
            chars=len(text),
        )
        return text
    except Exception as exc:  # noqa: BLE001 — любая проблема означает переход на yt-dlp
        log.info(
            "youtube.transcript.captions_failed",
            video_id=video_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return None


def _snippet_texts(fetched) -> list[str]:
    """Текст из FetchedTranscript; поддерживает и старый формат list[dict]."""
    texts: list[str] = []
    for snippet in fetched:
        value = getattr(snippet, "text", None)
        if value is None and isinstance(snippet, dict):
            value = snippet.get("text")
        if value:
            texts.append(str(value))
    return texts


# --- ветка 2: yt-dlp -----------------------------------------------------------------


def _fetch_via_ytdlp(video_id: str) -> str | None:
    """Автосубтитры через yt-dlp. Блокирующая, вызывается в to_thread."""
    try:
        from yt_dlp import YoutubeDL

        with tempfile.TemporaryDirectory(prefix="metapulse-subs-") as tmp_dir:
            tmp = Path(tmp_dir)
            options = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": YTDLP_SUB_LANGS,
                "subtitlesformat": "vtt",
                "outtmpl": str(tmp / "%(id)s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "noprogress": True,
                "socket_timeout": 30,
                "retries": 2,
            }
            proxy = (settings.youtube_proxy or "").strip()
            if proxy:
                options["proxy"] = proxy

            with YoutubeDL(options) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

            for path in _subtitle_files(tmp):
                text = parse_vtt(path.read_text(encoding="utf-8", errors="replace"))
                if text:
                    log.info(
                        "youtube.transcript.ytdlp_ok",
                        video_id=video_id,
                        file=path.name,
                        chars=len(text),
                    )
                    return text
        return None
    except Exception as exc:  # noqa: BLE001 — фолбэк не имеет права ронять пайплайн
        log.warning(
            "youtube.transcript.ytdlp_failed",
            video_id=video_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return None


def _subtitle_files(directory: Path) -> list[Path]:
    """Файлы субтитров, английские первыми."""
    files = [p for p in directory.iterdir() if p.suffix.lower() in SUBTITLE_SUFFIXES]
    return sorted(files, key=lambda p: (0 if ".en" in p.name.lower() else 1, p.name))


def parse_vtt(content: str) -> str:
    """VTT/SRT -> сплошной текст: без таймкодов, тегов <c> и повторов роллинг-субтитров."""
    payload: list[str] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "-->" in line:
            continue
        if _CUE_INDEX_RE.match(line):
            continue
        if line.startswith(_VTT_META_PREFIXES):
            continue
        payload.append(line)
    return _clean_lines(payload)


def _clean_lines(raw_lines: Iterable[str]) -> str:
    """Чистит теги/сущности и схлопывает повторы.

    Автосубтитры идут «бегущей строкой»: одна и та же фраза приезжает несколько раз,
    дорастая по словам. Без схлопывания транскрипт раздувается втрое и упирается
    в лимит символов на пустом месте.
    """
    result: list[str] = []
    for raw in raw_lines:
        line = _WS_RE.sub(" ", html.unescape(_TAG_RE.sub("", raw or ""))).strip()
        if not line:
            continue
        if result:
            previous = result[-1]
            if line == previous:
                continue
            if line.startswith(previous):
                result[-1] = line
                continue
            if previous.startswith(line):
                continue
        if line in result[-3:]:
            continue
        result.append(line)
    return " ".join(result)
