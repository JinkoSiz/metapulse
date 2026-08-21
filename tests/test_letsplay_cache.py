"""Кэш разбора летсплея: удачу помним долго, неудачу — коротко."""

from __future__ import annotations

import datetime as dt

from app.config import settings
from app.db.models import LetsPlay
from app.youtube.service import RETRY_FAILED_AFTER_HOURS, _is_fresh

NOW = dt.datetime(2026, 8, 21, 12, 0, tzinfo=dt.UTC)


def row(*, hours_ago: float, conclusion: str | None) -> LetsPlay:
    return LetsPlay(
        game_id=1,
        conclusion=conclusion,
        fetched_at=NOW - dt.timedelta(hours=hours_ago),
    )


def test_successful_parse_is_cached_for_days() -> None:
    fresh = row(hours_ago=24, conclusion="Блогер доволен боевой системой.")
    assert _is_fresh(fresh, now=NOW)

    stale = row(hours_ago=24 * settings.letsplay_ttl_days + 1, conclusion="Заключение")
    assert not _is_fresh(stale, now=NOW)


def test_failure_is_retried_the_same_day() -> None:
    """Причина неудачи обычно внешняя и устранимая — не настроен прокси, нет ключа.

    Кэшируй её на неделю, и после починки конфигурации летсплеи не появились бы
    до следующего понедельника.
    """
    recent_failure = row(hours_ago=1, conclusion=None)
    assert _is_fresh(recent_failure, now=NOW)

    older_failure = row(hours_ago=RETRY_FAILED_AFTER_HOURS + 1, conclusion=None)
    assert not _is_fresh(older_failure, now=NOW)


def test_never_fetched_is_not_fresh() -> None:
    assert not _is_fresh(LetsPlay(game_id=1, fetched_at=None), now=NOW)
