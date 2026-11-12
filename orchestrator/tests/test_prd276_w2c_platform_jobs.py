"""PRD-276 Wave 2c (third batch): five more platform jobs are scheduled job kinds.

The nightly edge builder, the daily Composio action sync, the daily field-memory promotion, the
proactive-runs prune and the workspace export sweep each had an APScheduler job on the scheduler
process. Each is now a platform schedule that only queues, and a kind whose handler runs the same
work on a worker. Each catches up ``latest``. A worker that isn't the scheduler leader can run
them: the edge builder reads its window from the settings when it runs, not from ``start``.
"""
from __future__ import annotations

import asyncio
import importlib
from datetime import timedelta
from types import SimpleNamespace

import pytest

from core.jobs import schedules
from core.jobs.registry import CATCH_UP_LATEST, kind_of

JOBS = [
    ("services.edge_builder_scheduler", "EDGE_BUILDER_KIND", "run_edge_builder"),
    ("services.composio_sync_scheduler", "COMPOSIO_SYNC_KIND", "run_composio_sync"),
    ("jobs.promote_field_memory", "FIELD_PROMOTION_KIND", "run_field_promotion_job"),
    ("services.proactive.retention", "PRUNE_KIND", "run_prune_job"),
    ("services.workspace_move.jobs", "SWEEP_KIND", "run_sweep_job"),
]
WORK = {  # what each handler runs, to stub
    "services.edge_builder_scheduler": ("get_edge_builder_scheduler", "_run_edge_builder"),
    "services.composio_sync_scheduler": ("get_composio_sync_scheduler", "_run_sync"),
    "jobs.promote_field_memory": ("get_field_promotion_scheduler", "_run"),
    "services.proactive.retention": (None, "run_prune"),
    "services.workspace_move.jobs": (None, "sweep_exports"),
}


@pytest.mark.parametrize("module_name, kind_name, handler_name", JOBS)
def test_each_is_a_latest_kind_whose_handler_runs_its_work(monkeypatch, module_name, kind_name, handler_name):
    module = importlib.import_module(module_name)
    getter, work = WORK[module_name]
    target = getattr(module, getter)() if getter else module
    ran = []

    if asyncio.iscoroutinefunction(getattr(target, work)):
        async def fake(*args, **kwargs):
            ran.append(work)
    else:
        def fake(*args, **kwargs):
            ran.append(work)

    monkeypatch.setattr(target, work, fake)
    spec = kind_of(getattr(module, kind_name))
    assert (spec.catch_up, spec.concurrency) == (CATCH_UP_LATEST, 1)
    asyncio.run(getattr(module, handler_name)(object()))
    assert ran == [work]


def test_the_edge_builder_reads_its_window_when_it_runs(monkeypatch):
    from config import config
    from services import edge_builder_scheduler as edges

    monkeypatch.setattr(config, "EDGE_BUILDER_WINDOW_DAYS", 9, raising=False)
    windows = []

    async def build(window):
        windows.append(window)
        return SimpleNamespace(failed=True)

    monkeypatch.setattr("core.services.edge_builder_worker.build_edges_in_worker", build)
    fresh = edges.EdgeBuilderScheduler()  # never started here: another process leads the scheduler
    asyncio.run(fresh._run_edge_builder())
    assert windows == [timedelta(days=9)]


def test_field_promotion_is_scheduled_only_when_on_and_stops_cleanly(monkeypatch):
    from config import config
    from jobs import promote_field_memory as field

    added, removed = {}, []
    scheduler = SimpleNamespace(add_job=lambda fn, trigger, **kw: added.setdefault(kw["id"], trigger),
                                get_job=lambda job_id: added.get(job_id), remove_job=removed.append)
    monkeypatch.setattr(config, "FIELD_PROMOTION_ENABLED", False)
    asyncio.run(field.FieldPromotionJobScheduler().start(scheduler))
    assert added == {}
    monkeypatch.setattr(config, "FIELD_PROMOTION_ENABLED", True)
    jobs = field.FieldPromotionJobScheduler()
    asyncio.run(jobs.start(scheduler))
    timer = schedules.job_id(field.FIELD_PROMOTION_KIND, schedules.PLATFORM)
    assert {f.name: str(f) for f in added[timer].fields}["minute"] == "30"
    asyncio.run(jobs.stop())
    assert timer in removed and schedules.declared(field.FIELD_PROMOTION_KIND, schedules.PLATFORM) is None


def test_every_worker_loads_them():
    from core.jobs import worker

    assert {module for module, _, _ in JOBS} <= set(worker.HANDLER_MODULES)
