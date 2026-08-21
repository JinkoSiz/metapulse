"""Поиск самого популярного летсплея. Два сменных бэкенда.

`yt-dlp` (по умолчанию) — без ключа и без квоты, выдача сразу содержит просмотры и
длительность. `official` — YouTube Data API v3: `search.list` стоит 100 юнитов из
суточных 10000, то есть не больше 100 поисков в сутки, и не отдаёт ни просмотров, ни
длительности — их приходится добирать вторым запросом `videos.list`.

Data API работает по REST через httpx, а не через google-api-python-client: последний
синхронный и тянет в образ пол-Google Cloud ради двух GET-запросов.

Из России домен youtube.com напрямую недоступен, поэтому у yt-dlp есть настройка
прокси (`youtube_proxy`). Любопытно, что `googleapis.com` при этом отвечает напрямую —
у двух бэкендов разные требования к сети.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import httpx
import structlog
from pydantic import BaseModel, ConfigDict
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

log = structlog.get_logger(__name__)

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

SEARCH_MAX_RESULTS = 10
MIN_DURATION_SEC = 8 * 60
REQUEST_TIMEOUT_S = 20.0
MAX_ATTEMPTS = 3

# Выдача по order=viewCount почти всегда возглавляется трейлером или обзором: у них
# просмотров на порядок больше, чем у любого летсплея, и без этого фильтра «самым
# популярным летсплеем» стабильно оказывается двухминутный релизный трейлер.
_EXCLUDE_PATTERNS = (
    r"\btrailers?\b",
    r"\breviews?\b",
    r"\bобзор\w*",
    r"\bost\b",
    r"\bsoundtracks?\b",
    r"before you buy",
    r"first impressions?",
    r"is it worth",
    r"\bvs\b",
    r"\btier list\b",
    r"\bstoit li\b",
    r"стоит ли",
)
_EXCLUDE_RE = re.compile("|".join(_EXCLUDE_PATTERNS), re.IGNORECASE)

# Признаки собственно летсплея. Нужны потому, что по просмотрам обзор почти всегда
# обгоняет прохождение: у «Before You Buy» миллион просмотров против сотни тысяч у
# двухчасового walkthrough, а пересказывать надо именно игровой процесс.
_LETSPLAY_PATTERNS = (
    r"let'?s play",
    r"\bwalkthrough\b",
    r"\bplaythrough\b",
    r"\bgameplay\b",
    r"\bfull game\b",
    r"\bpart\s*\d+",
    r"прохождени\w*",
    r"летсплей",
)
_LETSPLAY_RE = re.compile("|".join(_LETSPLAY_PATTERNS), re.IGNORECASE)


def looks_like_letsplay(title: str) -> bool:
    return bool(_LETSPLAY_RE.search(title or ""))

_ISO_DURATION_RE = re.compile(
    r"^P(?:(?P<weeks>\d+)W)?(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
)


class YouTubeApiError(RuntimeError):
    """Ошибка обращения к YouTube Data API: неверный ключ, исчерпанная квота, сеть."""


class _TransientApiError(YouTubeApiError):
    """Временная ошибка (429/5xx) — есть смысл повторить запрос."""


class VideoCandidate(BaseModel):
    """Ролик-кандидат со всеми полями, нужными для выбора и для карточки игры."""

    model_config = ConfigDict(extra="ignore")

    video_id: str
    url: str
    title: str
    channel: str
    view_count: int
    duration_sec: int


def video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def parse_iso8601_duration(value: str | None) -> int | None:
    """ISO-8601 длительность из contentDetails.duration в секунды.

    Возвращает None, если строку не удалось разобрать. Для идущих прямых эфиров
    YouTube отдаёт "P0D" — это корректно разбирается в 0 и отсеивается фильтром длины.
    """
    if not value:
        return None
    match = _ISO_DURATION_RE.match(value.strip())
    if match is None:
        return None
    parts = match.groupdict()
    total = 0.0
    total += int(parts["weeks"] or 0) * 7 * 86400
    total += int(parts["days"] or 0) * 86400
    total += int(parts["hours"] or 0) * 3600
    total += int(parts["minutes"] or 0) * 60
    total += float(parts["seconds"] or 0)
    return int(total)


def is_excluded_title(title: str) -> bool:
    """Заголовок явно не летсплей (трейлер, обзор, саундтрек)."""
    return bool(_EXCLUDE_RE.search(title or ""))


def pick_best(
    candidates: Iterable[VideoCandidate],
    *,
    min_duration_sec: int = MIN_DURATION_SEC,
) -> VideoCandidate | None:
    """Самый просматриваемый летсплей достаточной длины.

    Ролики с признаками прохождения имеют приоритет над всеми остальными, даже если
    просмотров у них меньше: иначе «самым популярным летсплеем» стабильно оказывается
    обзор или разбор, где о самой игре говорят пару минут.
    """
    suitable = [
        c
        for c in candidates
        if c.duration_sec >= min_duration_sec and not is_excluded_title(c.title)
    ]
    if not suitable:
        return None
    letsplays = [c for c in suitable if looks_like_letsplay(c.title)]
    pool = letsplays or suitable
    # доп. ключи сортировки нужны только чтобы выбор был детерминированным при равных просмотрах
    return max(pool, key=lambda c: (c.view_count, c.duration_sec, c.video_id))


def build_candidates(
    snippets: dict[str, dict[str, str]],
    videos_payload: dict,
) -> list[VideoCandidate]:
    """Склеивает snippet-данные из search.list со статистикой из videos.list."""
    candidates: list[VideoCandidate] = []
    for item in videos_payload.get("items") or []:
        video_id = item.get("id")
        if not video_id:
            continue
        duration_sec = parse_iso8601_duration((item.get("contentDetails") or {}).get("duration"))
        if duration_sec is None:
            continue
        raw_views = (item.get("statistics") or {}).get("viewCount")
        try:
            view_count = int(raw_views)
        except (TypeError, ValueError):
            # у роликов со скрытой статистикой viewCount отсутствует — сравнивать нечем
            continue
        snippet = snippets.get(video_id) or {}
        fallback = item.get("snippet") or {}
        candidates.append(
            VideoCandidate(
                video_id=video_id,
                url=video_url(video_id),
                title=snippet.get("title") or fallback.get("title") or "",
                channel=snippet.get("channel") or fallback.get("channelTitle") or "",
                view_count=view_count,
                duration_sec=duration_sec,
            )
        )
    return candidates


def _index_search_items(payload: dict) -> dict[str, dict[str, str]]:
    """video_id -> {title, channel}; порядок выдачи сохраняется."""
    indexed: dict[str, dict[str, str]] = {}
    for item in payload.get("items") or []:
        video_id = (item.get("id") or {}).get("videoId")
        if not video_id:
            continue
        snippet = item.get("snippet") or {}
        indexed[video_id] = {
            "title": snippet.get("title") or "",
            "channel": snippet.get("channelTitle") or "",
        }
    return indexed


@retry(
    retry=retry_if_exception_type((_TransientApiError, httpx.TransportError)),
    wait=wait_exponential(multiplier=0.5, max=8),
    stop=stop_after_attempt(MAX_ATTEMPTS),
    reraise=True,
)
async def _get_json(client: httpx.AsyncClient, path: str, params: dict) -> dict:
    try:
        response = await client.get(f"{YOUTUBE_API_BASE}/{path}", params=params)
    except httpx.HTTPError as exc:
        raise _TransientApiError(f"сеть YouTube API ({path}): {exc}") from exc

    if response.status_code == 429 or response.status_code >= 500:
        raise _TransientApiError(f"YouTube API {path}: HTTP {response.status_code}")
    if response.status_code >= 400:
        raise YouTubeApiError(
            f"YouTube API {path}: HTTP {response.status_code} {_error_reason(response)}"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise YouTubeApiError(f"YouTube API {path}: ответ не является JSON") from exc


def _error_reason(response: httpx.Response) -> str:
    """Достаёт reason из тела ошибки (quotaExceeded, keyInvalid и т.п.)."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    error = payload.get("error") or {}
    errors = error.get("errors") or []
    reason = errors[0].get("reason") if errors else None
    return reason or error.get("message") or ""


def _ytdlp_search_options() -> dict[str, object]:
    options: dict[str, object] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,  # только метаданные выдачи: просмотры приходят сразу
        "socket_timeout": 30,
    }
    if settings.youtube_proxy:
        options["proxy"] = settings.youtube_proxy
    return options


def _ytdlp_candidates(entries: Iterable[dict]) -> list[VideoCandidate]:
    candidates: list[VideoCandidate] = []
    for entry in entries:
        video_id = entry.get("id")
        views = entry.get("view_count")
        duration = entry.get("duration")
        # Ролики без просмотров или длительности сравнивать не с чем
        if not video_id or views is None or duration is None:
            continue
        candidates.append(
            VideoCandidate(
                video_id=str(video_id),
                url=video_url(str(video_id)),
                title=str(entry.get("title") or ""),
                channel=str(entry.get("channel") or entry.get("uploader") or ""),
                view_count=int(views),
                duration_sec=int(duration),
            )
        )
    return candidates


async def search_via_ytdlp(title: str) -> list[VideoCandidate]:
    """Поиск через yt-dlp: без ключа и без суточной квоты в 100 запросов.

    В отличие от Data API, выдача сразу содержит просмотры и длительность,
    поэтому второй запрос за статистикой не нужен.
    """
    import asyncio

    from yt_dlp import YoutubeDL

    query = f"ytsearch{settings.youtube_search_results}:{title} let's play"

    def _run() -> list[VideoCandidate]:
        with YoutubeDL(_ytdlp_search_options()) as ydl:
            info = ydl.extract_info(query, download=False)
        entries = (info or {}).get("entries") or []
        return _ytdlp_candidates(e for e in entries if isinstance(e, dict))

    return await asyncio.to_thread(_run)


async def pick_best_letsplay(
    title: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> VideoCandidate | None:
    """Ищет самый популярный летсплей игры.

    Бэкенд выбирается настройкой `youtube_search_backend`: `yt-dlp` работает без ключа,
    `official` — через Data API v3 (нужен ключ, лимит 100 поисков в сутки).
    Возвращает None, если бэкенд недоступен или ни один ролик не прошёл фильтры.
    """
    if settings.youtube_search_backend == "yt-dlp":
        candidates = await search_via_ytdlp(title)
        best = pick_best(candidates)
        log.info(
            "youtube.search.ytdlp",
            title=title,
            considered=len(candidates),
            picked=best.video_id if best else None,
        )
        return best

    api_key = (settings.youtube_api_key or "").strip()
    if not api_key:
        log.warning("youtube.search.no_api_key", title=title)
        return None

    query = f"{title} let's play"
    own_client = client is None
    http = client or httpx.AsyncClient(
        timeout=httpx.Timeout(REQUEST_TIMEOUT_S),
        headers={"Accept": "application/json"},
    )
    try:
        search_payload = await _get_json(
            http,
            "search",
            {
                "part": "snippet",
                "q": query,
                "type": "video",
                "order": "viewCount",
                "maxResults": SEARCH_MAX_RESULTS,
                "key": api_key,
            },
        )
        snippets = _index_search_items(search_payload)
        if not snippets:
            log.info("youtube.search.empty", title=title, query=query)
            return None

        videos_payload = await _get_json(
            http,
            "videos",
            {
                "part": "statistics,contentDetails",
                "id": ",".join(snippets),
                "key": api_key,
            },
        )
        candidates = build_candidates(snippets, videos_payload)
        best = pick_best(candidates)
        if best is None:
            log.info(
                "youtube.search.all_filtered",
                title=title,
                candidates=len(candidates),
                min_duration_sec=MIN_DURATION_SEC,
            )
            return None
        log.info(
            "youtube.search.picked",
            title=title,
            video_id=best.video_id,
            view_count=best.view_count,
            duration_sec=best.duration_sec,
            considered=len(candidates),
        )
        return best
    finally:
        if own_client:
            await http.aclose()
