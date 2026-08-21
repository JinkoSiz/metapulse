"""Чистые парсеры ответов Metacritic: JSON -> модели. Без сети и без БД.

Все функции терпимы к «дырам» в выдаче: Metacritic регулярно отдаёт `null`,
пустые строки и отсутствующие блоки вместо честного отсутствия поля.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from app.metacritic.schemas import (
    FinderItem,
    GameDetail,
    PlatformInfo,
    ReviewItem,
    ScoreStats,
)

IMAGE_BASE = "https://www.metacritic.com/a/img"
SITE_BASE = "https://www.metacritic.com"
JWPLAYER_EMBED = "https://cdn.jwplayer.com/players/{jw_id}.html"

# Порядок предпочтения обложки в деталке: cardImage — вертикальный постер, mainImage — кадр.
_COVER_PRIORITY = ("cardImage", "mainImage")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str | None) -> str | None:
    """Имя платформы -> слаг Metacritic ("PlayStation 5" -> "playstation-5")."""
    if not value:
        return None
    slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    return slug or None


def parse_date(value: Any) -> dt.date | None:
    """Дата из выдачи: 'YYYY-MM-DD', ISO-8601 с временем, пустая строка, null или вложенный dict."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, dict):  # displayDatePublished и подобные обёртки
        return parse_date(value.get("date"))
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        return dt.date.fromisoformat(raw[:10])
    except ValueError:
        pass
    try:
        return dt.datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean(value: Any) -> str | None:
    """Пустая строка в выдаче означает «нет значения» — приводим её к None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def build_image_url(image: dict[str, Any] | None) -> str | None:
    """Абсолютный URL картинки из пары bucketType + bucketPath.

    Готового URL в выдаче нет: `imageUrl` всегда null, склейка обязательна.
    """
    if not isinstance(image, dict):
        return None
    direct = _clean(image.get("imageUrl"))
    if direct:
        return direct if direct.startswith("http") else f"{SITE_BASE}{direct}"
    bucket_type = _clean(image.get("bucketType"))
    bucket_path = _clean(image.get("bucketPath"))
    if not bucket_type or not bucket_path:
        return None
    if not bucket_path.startswith("/"):
        bucket_path = f"/{bucket_path}"
    return f"{IMAGE_BASE}/{bucket_type}{bucket_path}"


def _pick_cover(item: dict[str, Any]) -> str | None:
    """Обложка деталки: сначала постер (cardImage), потом кадр, потом что нашлось."""
    images = item.get("images")
    if isinstance(images, list) and images:
        by_type = {_clean(img.get("typeName")): img for img in images if isinstance(img, dict)}
        for type_name in _COVER_PRIORITY:
            url = build_image_url(by_type.get(type_name))
            if url:
                return url
        for img in images:
            url = build_image_url(img if isinstance(img, dict) else None)
            if url:
                return url
    return build_image_url(item.get("image"))


def _genre_names(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for genre in raw:
        name = _clean(genre.get("name")) if isinstance(genre, dict) else _clean(genre)
        if name and name not in names:
            names.append(name)
    return names


def _company(production: Any, type_name: str) -> str | None:
    """Разработчик/издатель из production.companies по typeName.

    Компаний одного типа бывает несколько — берём первую.
    """
    if not isinstance(production, dict):
        return None
    companies = production.get("companies")
    if not isinstance(companies, list):
        return None
    for company in companies:
        if not isinstance(company, dict):
            continue
        if (company.get("typeName") or "").strip().lower() == type_name.lower():
            name = _clean(company.get("name"))
            if name:
                return name
    return None


def parse_score_stats(raw: Any) -> ScoreStats | None:
    """ScoreStats из criticScoreSummary или из data.item эндпоинта user-stats."""
    if isinstance(raw, dict) and "item" in raw:  # допускаем передачу целого конверта
        raw = raw["item"]
    if isinstance(raw, dict) and "data" in raw:
        raw = (raw.get("data") or {}).get("item")
    if not isinstance(raw, dict):
        return None
    score = _as_float(raw.get("score"))
    review_count = _as_int(raw.get("reviewCount"))
    sentiment = _clean(raw.get("sentiment"))
    if score is None and review_count is None and sentiment is None:
        return None
    return ScoreStats(score=score, review_count=review_count, sentiment=sentiment)


def _unwrap_item(raw: dict[str, Any]) -> dict[str, Any]:
    """Позволяет передавать как сам объект, так и конверт {"data": {"item": ...}}."""
    data = raw.get("data")
    if isinstance(data, dict) and isinstance(data.get("item"), dict):
        return data["item"]
    item = raw.get("item")
    if isinstance(item, dict):
        return item
    return raw


def parse_finder_item(raw: dict[str, Any]) -> FinderItem:
    """Элемент listing-выдачи finder-а (карусель и browse отдают одинаковую форму)."""
    critic = raw.get("criticScoreSummary")
    metascore = _as_int(critic.get("score")) if isinstance(critic, dict) else None
    return FinderItem(
        mc_id=int(raw["id"]),
        slug=str(raw["slug"]),
        title=_clean(raw.get("title")) or str(raw["slug"]),
        release_date=parse_date(raw.get("releaseDate")),
        description=_clean(raw.get("description")),
        cover_url=build_image_url(raw.get("image")),
        metascore=metascore,
        genres=_genre_names(raw.get("genres")),
    )


def parse_platform(raw: dict[str, Any]) -> PlatformInfo:
    name = _clean(raw.get("name")) or "Unknown"
    slug = _clean(raw.get("slug")) or slugify(name) or "unknown"
    return PlatformInfo(
        mc_id=_as_int(raw.get("id")),
        name=name,
        slug=slug,
        is_lead=bool(raw.get("isLeadPlatform")),
        release_date=parse_date(raw.get("releaseDate")),
        metascore=parse_score_stats(raw.get("criticScoreSummary")),
    )


def parse_game_detail(raw: dict[str, Any]) -> GameDetail:
    """Деталка игры. Принимает как `data.item`, так и весь ответ целиком."""
    item = _unwrap_item(raw)
    platforms = [
        parse_platform(platform)
        for platform in (item.get("platforms") or [])
        if isinstance(platform, dict)
    ]

    video = item.get("video") if isinstance(item.get("video"), dict) else {}
    trailer_url = _clean(video.get("embedUrl"))
    if not trailer_url:
        jw_id = _clean(video.get("jwPlayerId"))
        # embedUrl иногда пустой, хотя проигрыватель есть — собираем ссылку сами
        trailer_url = JWPLAYER_EMBED.format(jw_id=jw_id) if jw_id else None

    lead_metascore = None
    top_summary = parse_score_stats(item.get("criticScoreSummary"))
    if top_summary is not None:
        lead_metascore = _as_int(top_summary.score)
    if lead_metascore is None:
        for platform in platforms:
            if platform.is_lead and platform.metascore is not None:
                lead_metascore = _as_int(platform.metascore.score)
                break

    return GameDetail(
        mc_id=int(item["id"]),
        slug=str(item["slug"]),
        title=_clean(item.get("title")) or str(item["slug"]),
        description=_clean(item.get("description")),
        developer=_company(item.get("production"), "Developer"),
        publisher=_company(item.get("production"), "Publisher"),
        release_date=parse_date(item.get("releaseDate")) or parse_date(item.get("releaseDateText")),
        esrb_rating=_clean(item.get("rating")),
        genres=_genre_names(item.get("genres")),
        cover_url=_pick_cover(item),
        trailer_embed_url=trailer_url,
        trailer_title=_clean(video.get("videoTitle")) or _clean(video.get("title")),
        platforms=platforms,
        lead_metascore=lead_metascore,
    )


def _review_platform_slug(raw: dict[str, Any]) -> str | None:
    """Платформа отзыва: строкой в `platform`, либо в reviewedProduct.platform.name."""
    slug = slugify(_clean(raw.get("platform")))
    if slug:
        return slug
    product = raw.get("reviewedProduct")
    if isinstance(product, dict):
        platform = product.get("platform")
        if isinstance(platform, dict):
            return slugify(_clean(platform.get("name")))
    return None


def parse_critic_review(raw: dict[str, Any]) -> ReviewItem:
    """Отзыв критика. Своего `id` у него нет — ключ собираем из издания, автора и даты."""
    publication_slug = (
        _clean(raw.get("publicationSlug")) or slugify(raw.get("publicationName")) or ""
    )
    author_raw = raw.get("author") or ""
    date_raw = raw.get("date") or ""
    source_key = f"{publication_slug}:{author_raw}:{date_raw}"[:255]
    return ReviewItem(
        source_key=source_key,
        kind="critic",
        score=_as_int(raw.get("score")),
        quote=_clean(raw.get("quote")),
        author=_clean(raw.get("author")),
        publication=_clean(raw.get("publicationName")),
        review_date=parse_date(raw.get("date")),
        external_url=_clean(raw.get("url")),
        platform_slug=_review_platform_slug(raw),
        spoiler=False,
    )


def parse_user_review(raw: dict[str, Any]) -> ReviewItem:
    """Отзыв пользователя. У него есть стабильный uuid — он и есть source_key."""
    source_key = _clean(raw.get("id"))
    if not source_key:  # подстраховка: без ключа строка не пройдёт UPSERT
        source_key = f"{_clean(raw.get('author')) or 'anon'}:{raw.get('date') or ''}"
    return ReviewItem(
        source_key=source_key[:255],
        kind="user",
        score=_as_int(raw.get("score")),
        quote=_clean(raw.get("quote")),
        author=_clean(raw.get("author")),
        publication=None,
        review_date=parse_date(raw.get("date")),
        external_url=None,
        platform_slug=_review_platform_slug(raw),
        spoiler=bool(raw.get("spoiler")),
    )
