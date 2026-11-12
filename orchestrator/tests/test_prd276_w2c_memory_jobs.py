"""PRD-276 Wave 2c (second batch): the eight memory jobs are scheduled job kinds.

``MemoryJobScheduler`` used to put eight APScheduler jobs on the scheduler process. Each is now a
platform schedule that only queues, and a job kind whose handler runs the same method on a worker.
Each catches up ``latest`` (every job sweeps all workspaces). A job switched off by its setting
isn't scheduled. The health status keeps its keys, and stopping removes the schedules.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace

import pytest

from core.jobs import schedules
from core.jobs.registry import CATCH_UP_LATEST, kind_of
from services import memory_jobs


class _Scheduler:
    def __init__(self):
        self.jobs = {}

    def add_job(self, fn, trigger, **kw):
        self.jobs[kw["id"]] = SimpleNamespace(fn=fn, trigger=trigger, next_run_time=None, **kw)

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def remove_job(self, job_id):
        self.jobs.pop(job_id)


@pytest.fixture
def started(monkeypatch):
    from config import config

    for switch in ("MEMORY_ARCHIVAL_ENABLED", "MEMORY_SNAPSHOT_ENABLED", "THREAD_CHECKPOINT_ENABLED"):
        monkeypatch.setattr(config, switch, True, raising=False)
    monkeypatch.setattr(config, "MEMORY_CONSOLIDATION_INTERVAL_SECONDS", 1800, raising=False)
    monkeypatch.setattr(config, "MEMORY_PROMOTION_HOUR_UTC", 5, raising=False)
    scheduler, jobs = _Scheduler(), memory_jobs.MemoryJobScheduler()
    asyncio.run(jobs.start(scheduler))
    yield scheduler, jobs
    asyncio.run(jobs.stop())


def test_every_memory_job_is_a_latest_kind_run_one_at_a_time():
    for _, kind, _, _ in memory_jobs._JOBS:
        spec = kind_of(kind)
        assert (spec.catch_up, spec.concurrency) == (CATCH_UP_LATEST, 1), kind


def test_each_is_scheduled_on_its_own_settings(started):
    scheduler, _ = started
    timers = {kind: scheduler.get_job(schedules.job_id(kind, schedules.PLATFORM)) for _, kind, _, _ in memory_jobs._JOBS}
    assert all(timer is not None and timer.fn is schedules.fire for timer in timers.values())
    assert timers["memory.consolidation"].trigger.interval == timedelta(seconds=1800)
    hour = {f.name: str(f) for f in timers["memory.promotion"].trigger.fields}["hour"]
    assert hour == "5"


def test_a_job_switched_off_is_not_scheduled(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "MEMORY_SNAPSHOT_ENABLED", False, raising=False)
    scheduler, jobs = _Scheduler(), memory_jobs.MemoryJobScheduler()
    asyncio.run(jobs.start(scheduler))
    try:
        assert scheduler.get_job(schedules.job_id("memory.snapshot", schedules.PLATFORM)) is None
        assert scheduler.get_job(schedules.job_id("memory.decay", schedules.PLATFORM)) is not None
    finally:
        asyncio.run(jobs.stop())


def test_the_status_keeps_its_keys_and_stopping_removes_the_schedules(started):
    scheduler, jobs = started
    status = jobs.get_status()
    assert status["active"] is True
    assert set(status["jobs"]) == {key for key, _, _, _ in memory_jobs._JOBS}
    assert all(job["active"] for job in status["jobs"].values())
    asyncio.run(jobs.stop())
    assert not any(job_id.startswith("schedule:memory.") for job_id in scheduler.jobs)
    assert schedules.declared("memory.decay", schedules.PLATFORM) is None


@pytest.mark.parametrize("kind, method", [(kind, method) for _, kind, method, _ in memory_jobs._JOBS])
def test_each_handler_runs_its_method(monkeypatch, kind, method):
    ran = []

    async def fake():
        ran.append(method)

    monkeypatch.setattr(memory_jobs.get_memory_job_scheduler(), method, fake)
    asyncio.run(kind_of(kind).handler(object()))
    assert ran == [method]


def test_every_worker_loads_them():
    from core.jobs import worker

    assert "services.memory_jobs" in worker.HANDLER_MODULES
