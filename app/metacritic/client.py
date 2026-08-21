"""Асинхронный клиент backend.metacritic.com.

Публичное API — методы `MetacriticClient`; разбор ответов вынесен в `parsers.py`.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.config import settings
from app.metacritic.parsers import (
    parse_critic_review,
    parse_finder_item,
    parse_game_detail,
    parse_score_stats,
    parse_user_review,
)
from app.metacritic.schemas import FinderItem, GameDetail, ReviewItem, ScoreStats

log = structlog.get_logger(__name__)

BACKEND_BASE = "https://backend.metacritic.com"
SITE_BASE = "https://www.metacritic.com"

FINDER_PATH = "/finder/metacritic/web"
GAME_PATH = "/games/metacritic/{slug}/web"
CRITIC_REVIEWS_PATH = "/reviews/metacritic/critic/games/{slug}/web"
USER_REVIEWS_PATH = "/reviews/metacritic/user/games/{slug}/web"
USER_STATS_PATH = "/reviews/metacritic/user/games/{slug}/platform/{platform}/stats/web"

# limit finder-а: 50 отдаётся, 60 и больше -> HTTP 400 (проверено запросами).
FINDER_MAX_LIMIT = 50
# Эндпоинт отзывов критиков игнорирует limit и всегда отдаёт ровно 10 элементов,
# поэтому листать можно только шагом 10 — иначе часть отзывов молча теряется.
CRITIC_PAGE_SIZE = 10
USER_PAGE_MAX_LIMIT = 200

_AUTH_STATUS = frozenset({401, 403})
_RETRY_STATUS = frozenset({403, 408, 425, 429, 500, 502, 503, 504})
_API_KEY_RE = re.compile(r"apiKey=([A-Za-z0-9]+)")

_CAROUSEL_PARAMS: dict[str, Any] = {
    "componentName": "new-releases-carousel",
    "componentDisplayName": "Newly Released",
    "componentType": "ProductList",
    "sortBy": "-releaseDate",
    "metaScoreMin": 1,
    "mcoTypeId": 13,
}


class MetacriticError(Exception):
    """Любая невосстановимая ошибка обращения к Metacritic."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class MetacriticNotFound(MetacriticError):
    """404: игры/платформы нет. Ретраить бессмысленно."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=404)


class _RetryableError(MetacriticError):
    """Временный сбой (429/5xx/403/битый JSON) — повторяем с backoff."""


def build_userscore_path(slug: str, platform_slug: str) -> str:
    """Путь per-platform userscore.

    Платформа обязана быть ИМЕННО path-сегментом: query-параметр
    (`?platform=` / `?filterByPlatform=`) сервер молча игнорирует и отдаёт
    цифры lead-платформы — тихая порча данных, которую видно только сверкой.
    """
    return USER_STATS_PATH.format(slug=quote(slug, safe=""), platform=quote(platform_slug, safe=""))


class _RateLimiter:
    """Пейсер запросов: не чаще `rps` штук в секунду на клиента."""

    def __init__(self, rps: float) -> None:
        self._interval = 1.0 / rps if rps and rps > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        if not self._interval:
            return
        async with self._lock:
            now = time.monotonic()
            wait_for = self._next_at - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
                now = time.monotonic()
            self._next_at = now + self._interval


class MetacriticClient:
    """Клиент Metacritic: троттлинг, ретраи с джиттером, перевыдача apiKey.

    Используется как асинхронный контекст-менеджер; переданный извне
    `httpx.AsyncClient` не закрывается (владелец — вызывающий код).
    """

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        self._client = client if client is not None else self._build_client()
        self._api_key = settings.mc_api_key
        self._limiter = _RateLimiter(settings.mc_rate_limit_rps)
        self._key_lock = asyncio.Lock()
        self._key_refreshed = False

    # --- жизненный цикл -------------------------------------------------

    @staticmethod
    def _build_client() -> httpx.AsyncClient:
        kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(settings.mc_timeout_s),
            "follow_redirects": True,
            "headers": {
                "User-Agent": settings.mc_user_agent,
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": f"{SITE_BASE}/",
                "Origin": SITE_BASE,
            },
        }
        if settings.mc_proxy:
            kwargs["proxy"] = settings.mc_proxy
        return httpx.AsyncClient(**kwargs)

    async def __aenter__(self) -> MetacriticClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # --- транспорт ------------------------------------------------------

    async def _send(self, path: str, params: dict[str, Any] | None) -> httpx.Response:
        query = {k: v for k, v in (params or {}).items() if v is not None}
        query["apiKey"] = self._api_key
        await self._limiter.acquire()
        return await self._client.get(f"{BACKEND_BASE}{path}", params=query)

    async def _refresh_api_key(self, slug: str | None) -> bool:
        """Выдирает свежий apiKey со страницы игры. Делается не более одного раза на клиента."""
        async with self._key_lock:
            if self._key_refreshed:
                return False
            self._key_refreshed = True
            page = f"{SITE_BASE}/game/{quote(slug, safe='')}/" if slug else f"{SITE_BASE}/game/"
            try:
                await self._limiter.acquire()
                resp = await self._client.get(page, headers={"Accept": "text/html,*/*"})
            except httpx.HTTPError as exc:
                log.warning("mc.api_key_refresh_failed", error=str(exc))
                return False
            if resp.status_code >= 400:
                log.warning("mc.api_key_refresh_failed", status=resp.status_code)
                return False
            match = _API_KEY_RE.search(resp.text)
            if not match:
                log.warning("mc.api_key_not_found", page=page)
                return False
            new_key = match.group(1)
            changed = new_key != self._api_key
            self._api_key = new_key
            log.info("mc.api_key_refreshed", changed=changed)
            return True

    async def _attempt(
        self, path: str, params: dict[str, Any] | None, *, slug_for_key: str | None
    ) -> dict[str, Any]:
        resp = await self._send(path, params)
        if resp.status_code in _AUTH_STATUS and await self._refresh_api_key(slug_for_key):
            resp = await self._send(path, params)

        status = resp.status_code
        if status == 404:
            raise MetacriticNotFound(f"Metacritic: ресурс не найден ({path})")
        if status in _RETRY_STATUS:
            raise _RetryableError(
                f"Metacritic: временная ошибка {status} ({path})", status_code=status
            )
        if status >= 400:
            raise MetacriticError(f"Metacritic: HTTP {status} ({path})", status_code=status)
        try:
            payload = resp.json()
        except ValueError as exc:  # вместо JSON прилетела HTML-заглушка защиты
            raise _RetryableError(f"Metacritic: ответ не является JSON ({path}): {exc}") from exc
        if not isinstance(payload, dict):
            raise MetacriticError(f"Metacritic: неожиданный формат ответа ({path})")
        return payload

    async def _get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        slug_for_key: str | None = None,
    ) -> dict[str, Any]:
        retryer = AsyncRetrying(
            stop=stop_after_attempt(max(1, settings.mc_max_retries)),
            wait=wait_random_exponential(multiplier=0.7, max=15),
            retry=retry_if_exception_type((_RetryableError, httpx.TransportError)),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                return await self._attempt(path, params, slug_for_key=slug_for_key)
        raise MetacriticError(f"Metacritic: запрос не удался ({path})")  # недостижимо

    @staticmethod
    def _items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        data = payload.get("data") or {}
        items = data.get("items")
        return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []

    @staticmethod
    def _total(payload: dict[str, Any]) -> int:
        data = payload.get("data") or {}
        try:
            return int(data.get("totalResults") or 0)
        except (TypeError, ValueError):
            return 0

    # --- листинги -------------------------------------------------------

    async def list_new_releases(self, limit: int = 20) -> list[FinderItem]:
        """Карусель «Newly Released» — свежие релизы с уже проставленным метаскором."""
        params = dict(_CAROUSEL_PARAMS)
        params["offset"] = 0
        params["limit"] = max(1, min(int(limit), FINDER_MAX_LIMIT))
        payload = await self._get_json(FINDER_PATH, params)
        items = [parse_finder_item(item) for item in self._items(payload)]
        log.debug("mc.new_releases", count=len(items))
        return items

    async def list_browse(
        self, offset: int, limit: int = FINDER_MAX_LIMIT
    ) -> tuple[list[FinderItem], int]:
        """Постраничный обход всего каталога игр. Возвращает (items, total_results)."""
        params = {
            "sortBy": "-releaseDate",
            "productType": "games",
            "offset": max(0, int(offset)),
            # limit > 50 сервер отвергает с HTTP 400, поэтому обрезаем молча
            "limit": max(1, min(int(limit), FINDER_MAX_LIMIT)),
        }
        payload = await self._get_json(FINDER_PATH, params)
        items = [parse_finder_item(item) for item in self._items(payload)]
        total = self._total(payload)
        log.debug("mc.browse", offset=offset, count=len(items), total=total)
        return items, total

    # --- игра -----------------------------------------------------------

    async def get_game(self, slug: str) -> GameDetail:
        payload = await self._get_json(
            GAME_PATH.format(slug=quote(slug, safe="")), slug_for_key=slug
        )
        item = (payload.get("data") or {}).get("item")
        if not isinstance(item, dict):
            raise MetacriticError(f"Metacritic: пустая деталка игры {slug}")
        return parse_game_detail(item)

    async def get_platform_userscore(self, slug: str, platform_slug: str) -> ScoreStats | None:
        """Userscore конкретной платформы. None, если у платформы нет пользовательских оценок."""
        try:
            payload = await self._get_json(
                build_userscore_path(slug, platform_slug), slug_for_key=slug
            )
        except MetacriticNotFound:
            log.debug("mc.userscore_missing", slug=slug, platform=platform_slug)
            return None
        return parse_score_stats((payload.get("data") or {}).get("item"))

    # --- отзывы ---------------------------------------------------------

    async def get_critic_reviews(
        self, slug: str, max_items: int = settings.mc_critic_reviews_max
    ) -> list[ReviewItem]:
        """Отзывы критиков. Пагинация обязательна: сервер игнорирует limit и даёт по 10 штук."""
        return await self._collect_reviews(
            path=CRITIC_REVIEWS_PATH.format(slug=quote(slug, safe="")),
            slug=slug,
            max_items=max_items,
            page_size=CRITIC_PAGE_SIZE,
            parse=parse_critic_review,
        )

    async def get_user_reviews(
        self, slug: str, max_items: int = settings.mc_user_reviews_max
    ) -> list[ReviewItem]:
        """Отзывы пользователей: здесь limit уважается (проверено до 200) — обычно хватает
        одного запроса."""
        return await self._collect_reviews(
            path=USER_REVIEWS_PATH.format(slug=quote(slug, safe="")),
            slug=slug,
            max_items=max_items,
            page_size=min(max(1, int(max_items)), USER_PAGE_MAX_LIMIT),
            parse=parse_user_review,
        )

    async def _collect_reviews(
        self,
        *,
        path: str,
        slug: str,
        max_items: int,
        page_size: int,
        parse: Callable[[dict[str, Any]], ReviewItem],
    ) -> list[ReviewItem]:
        wanted = max(0, int(max_items))
        collected: list[ReviewItem] = []
        seen: set[str] = set()
        offset = 0
        total: int | None = None

        while len(collected) < wanted:
            try:
                payload = await self._get_json(
                    path, {"offset": offset, "limit": page_size}, slug_for_key=slug
                )
            except MetacriticNotFound:
                break
            raw_items = self._items(payload)
            if total is None:
                total = self._total(payload)
            if not raw_items:
                break
            for raw in raw_items:
                review = parse(raw)
                if review.source_key in seen:
                    continue
                seen.add(review.source_key)
                collected.append(review)
                if len(collected) >= wanted:
                    break
            # шаг делаем по фактически полученной странице: критики отдают 10 при любом limit
            offset += len(raw_items)
            if total and offset >= total:
                break

        log.debug("mc.reviews", slug=slug, count=len(collected), total=total)
        return collected[:wanted]
