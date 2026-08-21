"""Выбор летсплея: самый просматриваемый ролик, но не трейлер и не короткий клип."""

from __future__ import annotations

from app.youtube.search import (
    VideoCandidate,
    build_candidates,
    is_excluded_title,
    parse_iso8601_duration,
    pick_best,
)


def candidate(video_id: str, title: str, views: int, duration: int) -> VideoCandidate:
    return VideoCandidate(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        title=title,
        channel="Channel",
        view_count=views,
        duration_sec=duration,
    )


def test_picks_most_viewed_long_letsplay() -> None:
    best = pick_best(
        [
            candidate("a", "Let's Play Elden Ring #1", 500_000, 45 * 60),
            candidate("b", "Elden Ring полное прохождение", 2_000_000, 90 * 60),
            candidate("c", "Elden Ring gameplay", 900_000, 30 * 60),
        ]
    )
    assert best is not None and best.video_id == "b"


def test_trailer_is_rejected_even_with_most_views() -> None:
    """Поиск по «let's play» стабильно подмешивает трейлеры — они не годятся для пересказа."""
    best = pick_best(
        [
            candidate("t", "Elden Ring - Official Launch Trailer", 50_000_000, 3 * 60),
            candidate("p", "Elden Ring Let's Play part 1", 120_000, 40 * 60),
        ]
    )
    assert best is not None and best.video_id == "p"


def test_short_clip_is_rejected() -> None:
    assert pick_best([candidate("s", "Elden Ring let's play", 10_000_000, 90)]) is None


def test_empty_input() -> None:
    assert pick_best([]) is None


def test_walkthrough_beats_more_popular_non_letsplay() -> None:
    """Разбор с миллионом просмотров не годится: пересказывать надо прохождение."""
    best = pick_best(
        [
            candidate("d", "Elden Ring — 10 фактов, которые вы не знали", 5_000_000, 20 * 60),
            candidate("w", "Elden Ring Walkthrough Part 1", 200_000, 120 * 60),
        ]
    )
    assert best is not None and best.video_id == "w"


def test_falls_back_to_any_suitable_video() -> None:
    """Если прохождений в выдаче нет, берём самое популярное из оставшегося."""
    best = pick_best(
        [
            candidate("a", "Elden Ring — интервью с разработчиками", 100_000, 30 * 60),
            candidate("b", "Elden Ring — история мира", 300_000, 25 * 60),
        ]
    )
    assert best is not None and best.video_id == "b"


def test_before_you_buy_is_excluded() -> None:
    assert is_excluded_title("Elden Ring — Before You Buy")
    assert is_excluded_title("Elden Ring: стоит ли покупать")
    assert not is_excluded_title("Elden Ring Gameplay Walkthrough Part 3")


def test_review_and_soundtrack_titles_excluded() -> None:
    assert is_excluded_title("Elden Ring Review")
    assert is_excluded_title("Обзор Elden Ring")
    assert is_excluded_title("Elden Ring OST - Main Theme")
    assert not is_excluded_title("Elden Ring Let's Play — часть 3")


def test_duration_parsing() -> None:
    assert parse_iso8601_duration("PT1H2M3S") == 3723
    assert parse_iso8601_duration("PT45M") == 2700
    assert parse_iso8601_duration("P0D") == 0  # идущий прямой эфир
    assert parse_iso8601_duration(None) is None
    assert parse_iso8601_duration("мусор") is None


def test_build_candidates_skips_hidden_statistics() -> None:
    """Ролики со скрытыми просмотрами сравнивать не с чем — их отбрасываем."""
    snippets = {
        "ok": {"title": "Let's Play", "channel": "Blogger"},
        "hidden": {"title": "Let's Play 2", "channel": "Blogger"},
    }
    payload = {
        "items": [
            {
                "id": "ok",
                "contentDetails": {"duration": "PT30M"},
                "statistics": {"viewCount": "1234"},
            },
            {"id": "hidden", "contentDetails": {"duration": "PT30M"}, "statistics": {}},
            {"id": "no-duration", "statistics": {"viewCount": "10"}},
        ]
    }

    candidates = build_candidates(snippets, payload)
    assert [c.video_id for c in candidates] == ["ok"]
    assert candidates[0].view_count == 1234
    assert candidates[0].channel == "Blogger"
