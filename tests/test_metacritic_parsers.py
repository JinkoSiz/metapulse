"""Парсеры Metacritic на реальных ответах API (скачаны 2026-08-20, лежат в fixtures/).

Тесты специально проверяют конкретные значения, а не «что-то распарсилось»: сломанное
поле в неофициальном API должно ронять тест, а не тихо давать None в проде.
"""

from __future__ import annotations

import datetime as dt

from app.metacritic.client import build_userscore_path
from app.metacritic.parsers import (
    build_image_url,
    parse_critic_review,
    parse_finder_item,
    parse_game_detail,
    parse_score_stats,
    parse_user_review,
)
from tests.conftest import load_fixture


def test_carousel_items() -> None:
    payload = load_fixture("finder_carousel.json")
    items = [parse_finder_item(raw) for raw in payload["data"]["items"]]

    assert len(items) == 20
    first = items[0]
    assert first.slug == "stalker-2-heart-of-chornobyl-cost-of-hope"
    assert first.mc_id == 1300728906
    assert first.release_date == dt.date(2026, 8, 20)
    assert first.metascore == 85
    assert first.cover_url == (
        "https://www.metacritic.com/a/img/catalog/provider/7/2/7-1787151432.jpg"
    )
    assert "FPS" in first.genres


def test_browse_page_and_total() -> None:
    payload = load_fixture("finder_browse_p0.json")
    items = [parse_finder_item(raw) for raw in payload["data"]["items"]]

    assert len(items) == 50
    assert payload["data"]["totalResults"] == 176188
    # В browse-выдаче попадаются игры без оценок — парсер обязан пережить null
    assert any(item.metascore is None for item in items)
    assert all(item.slug and item.title for item in items)


def test_game_detail_core_fields() -> None:
    detail = parse_game_detail(load_fixture("game_clair-obscur-expedition-33.json"))

    assert detail.title == "Clair Obscur: Expedition 33"
    assert detail.developer == "Sandfall Interactive"
    assert detail.publisher == "Kepler Interactive"
    assert detail.release_date == dt.date(2025, 4, 24)
    assert detail.esrb_rating == "M"
    assert detail.genres == ["JRPG"]
    assert detail.lead_metascore == 92
    assert detail.cover_url is not None
    assert detail.cover_url.startswith("https://www.metacritic.com/a/img/catalog/")
    assert detail.trailer_embed_url is not None
    assert "cdn.jwplayer.com" in detail.trailer_embed_url
    assert detail.description and len(detail.description) > 100


def test_game_detail_per_platform_metascores() -> None:
    """У каждой платформы своя оценка — это отдельное требование задания."""
    detail = parse_game_detail(load_fixture("game_clair-obscur-expedition-33.json"))
    scores = {p.slug: (p.metascore.score if p.metascore else None) for p in detail.platforms}

    assert scores == {"playstation-5": 92.0, "pc": 91.0, "xbox-series-x": 91.0}
    lead = detail.lead_platform
    assert lead is not None and lead.slug == "playstation-5"
    assert sum(1 for p in detail.platforms if p.is_lead) == 1


def test_second_game_detail_parses() -> None:
    detail = parse_game_detail(load_fixture("game_the-sinking-city-2.json"))
    assert detail.slug == "the-sinking-city-2"
    assert detail.title


def test_critic_reviews() -> None:
    payload = load_fixture("reviews_critic.json")
    reviews = [parse_critic_review(raw) for raw in payload["data"]["items"]]

    assert payload["data"]["totalResults"] == 86
    # Сервер игнорирует limit и отдаёт ровно 10 отзывов на страницу — отсюда пагинация в клиенте
    assert len(reviews) == 10
    assert all(r.kind == "critic" for r in reviews)
    assert all(r.source_key for r in reviews)
    assert len({r.source_key for r in reviews}) == len(reviews)
    first = reviews[0]
    assert first.publication == "4P.de"
    assert first.score == 100
    assert first.external_url and first.external_url.startswith("http")
    assert first.review_date == dt.date(2025, 4, 23)


def test_user_reviews() -> None:
    payload = load_fixture("reviews_user.json")
    reviews = [parse_user_review(raw) for raw in payload["data"]["items"]]

    assert payload["data"]["totalResults"] == 6697
    assert len(reviews) == 30
    assert all(r.kind == "user" for r in reviews)
    assert all(0 <= (r.score or 0) <= 10 for r in reviews)
    # source_key пользовательского отзыва — стабильный uuid из выдачи
    assert len(reviews[0].source_key) == 36
    assert any(r.quote for r in reviews)


def test_platform_userscore_stats() -> None:
    stats = parse_score_stats(load_fixture("userstats_pc.json"))
    assert stats is not None
    assert stats.score == 9.6
    assert stats.review_count == 6109
    assert stats.sentiment == "Universal acclaim"


def test_userscore_path_puts_platform_in_path() -> None:
    """Регрессия: платформа обязана быть path-сегментом.

    Query-параметр сервер молча игнорирует и отдаёт цифры lead-платформы — ошибка
    не проявляется ничем, кроме неверных чисел в карточке.
    """
    path = build_userscore_path("clair-obscur-expedition-33", "pc")
    assert path.endswith("/platform/pc/stats/web")
    assert "?" not in path and "filterByPlatform" not in path


def test_image_url_requires_bucket_pair() -> None:
    assert build_image_url({"bucketType": "catalog", "bucketPath": "/a/b.jpg"}) == (
        "https://www.metacritic.com/a/img/catalog/a/b.jpg"
    )
    assert build_image_url({"bucketType": "catalog"}) is None
    assert build_image_url(None) is None
