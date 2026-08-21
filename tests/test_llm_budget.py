"""Потолок времени на резюме внутри обхода."""

from __future__ import annotations

import time

from app.pipeline.tasks import _LlmBudget


def test_budget_is_not_exhausted_right_away() -> None:
    budget = _LlmBudget(60)
    assert not budget.exhausted
    assert budget.skipped == 0


def test_zero_means_no_limit() -> None:
    """Облачной модели ограничение ни к чему: 0 отключает его совсем."""
    budget = _LlmBudget(0)
    assert not budget.exhausted
    time.sleep(0.01)
    assert not budget.exhausted


def test_exhausted_budget_counts_skips(monkeypatch) -> None:
    """Исчерпав бюджет, обход дособирает игры без резюме и считает отложенные."""
    budget = _LlmBudget(60)
    assert not budget.exhausted

    # перематываем часы вперёд, вместо того чтобы ждать минуту
    monkeypatch.setattr(time, "monotonic", lambda: time.perf_counter() + 10_000)
    assert budget.exhausted

    budget.note_skip()
    budget.note_skip()
    assert budget.skipped == 2
