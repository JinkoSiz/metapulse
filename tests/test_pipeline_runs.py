"""Жизненный цикл прогонов обхода."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import select

from app.db.models import PipelineRun
from app.pipeline.tasks import _close_orphan_runs
from tests.conftest import postgres_required

pytestmark = postgres_required


async def test_orphan_run_is_closed_on_startup(session) -> None:
    """Прогон, оборванный рестартом контейнера, иначе висел бы в мониторинге вечно."""
    session.add_all(
        [
            PipelineRun(trigger="cron", status="running"),
            PipelineRun(
                trigger="manual",
                status="ok",
                finished_at=dt.datetime.now(dt.UTC),
                stats={"selected": 20},
            ),
        ]
    )
    await session.commit()

    await _close_orphan_runs()

    rows = (await session.scalars(select(PipelineRun).order_by(PipelineRun.id))).all()
    await session.refresh(rows[0])
    assert rows[0].status == "interrupted"
    assert rows[0].finished_at is not None
    assert "перезапустился" in rows[0].error
    # Успешно завершённый прогон трогать нельзя
    assert rows[1].status == "ok"
    assert rows[1].error is None
