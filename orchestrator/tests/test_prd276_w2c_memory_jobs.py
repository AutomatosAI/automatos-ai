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


def test_every_memory_job_is_a_latest_kind_run_one_at_a_time_on_a_batch_lease():
    for _, kind, _, _ in memory_jobs._JOBS:
        spec = kind_of(kind)
        assert (spec.catch_up, spec.concurrency) == (CATCH_UP_LATEST, 1), kind
        assert spec.lease_seconds == memory_jobs.MEMORY_JOB_LEASE_SECONDS


def _shape(trigger):
    """A trigger as plain values: an interval's seconds, or a cron's day, hour and minute; and its zone."""
    zone = str(trigger.timezone)
    if hasattr(trigger, "interval"):
        return ("interval", int(trigger.interval.total_seconds()), trigger.start_date == schedules.ANCHOR, zone)
    fields = {f.name: str(f) for f in trigger.fields}
    return ("cron", fields["day"], fields["hour"], fields["minute"], zone)


SETTINGS = {
    "MEMORY_CONSOLIDATION_INTERVAL_SECONDS": 1800, "MEMORY_DECAY_INTERVAL_SECONDS": 7200,
    "MEMORY_PROMOTION_HOUR_UTC": 5, "MEMORY_ARCHIVAL_CRON_DAY": 15, "MEMORY_ARCHIVAL_CRON_HOUR": 2,
    "AUDIT_RETENTION_SWEEP_INTERVAL_SECONDS": 43200, "MEMORY_SNAPSHOT_CRON_HOUR_UTC": 6,
    "SUBSTRATE_METRICS_PRUNE_INTERVAL_SECONDS": 3600, "THREAD_CHECKPOINT_SWEEP_INTERVAL_SECONDS": 600,
}
DEFAULTS = {
    "memory.consolidation": ("interval", 3600, True, "UTC"), "memory.decay": ("interval", 3600, True, "UTC"),
    "memory.promotion": ("cron", "*", "3", "0", "UTC"), "memory.archival": ("cron", "1", "3", "0", "UTC"),
    "audit.retention_sweep": ("interval", 86400, True, "UTC"), "memory.snapshot": ("cron", "*", "4", "0", "UTC"),
    "substrate.metrics_prune": ("interval", 86400, True, "UTC"),
    "memory.thread_checkpoints": ("interval", 900, True, "UTC"),
}
SET = {
    "memory.consolidation": ("interval", 1800, True, "UTC"), "memory.decay": ("interval", 7200, True, "UTC"),
    "memory.promotion": ("cron", "*", "5", "0", "UTC"), "memory.archival": ("cron", "15", "2", "0", "UTC"),
    "audit.retention_sweep": ("interval", 43200, True, "UTC"), "memory.snapshot": ("cron", "*", "6", "0", "UTC"),
    "substrate.metrics_prune": ("interval", 3600, True, "UTC"),
    "memory.thread_checkpoints": ("interval", 600, True, "UTC"),
}


@pytest.mark.parametrize("kind", sorted(DEFAULTS))
def test_each_trigger_matches_its_settings_and_its_defaults(kind):
    assert _shape(memory_jobs._trigger(kind, SimpleNamespace())) == DEFAULTS[kind]  # no setting: the default
    assert _shape(memory_jobs._trigger(kind, SimpleNamespace(**SETTINGS))) == SET[kind]


def test_each_is_scheduled_on_its_own_settings(started):
    scheduler, _ = started
    timers = {kind: scheduler.get_job(schedules.job_id(kind, schedules.PLATFORM)) for _, kind, _, _ in memory_jobs._JOBS}
    assert all(timer is not None and timer.fn is schedules.fire for timer in timers.values())
    assert timers["memory.consolidation"].trigger.interval == timedelta(seconds=1800)
    hour = {f.name: str(f) for f in timers["memory.promotion"].trigger.fields}["hour"]
    assert hour == "5"


@pytest.mark.parametrize("switch, kind", [("MEMORY_SNAPSHOT_ENABLED", "memory.snapshot"),
                                          ("MEMORY_ARCHIVAL_ENABLED", "memory.archival"),
                                          ("THREAD_CHECKPOINT_ENABLED", "memory.thread_checkpoints")])
def test_a_job_switched_off_is_not_scheduled(monkeypatch, switch, kind):
    from config import config

    monkeypatch.setattr(config, switch, False, raising=False)
    scheduler, jobs = _Scheduler(), memory_jobs.MemoryJobScheduler()
    asyncio.run(jobs.start(scheduler))
    try:
        assert scheduler.get_job(schedules.job_id(kind, schedules.PLATFORM)) is None
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


@pytest.mark.parametrize("method, target, work", [
    ("_run_audit_retention", "services.audit_retention.sweep_expired_audit_logs", lambda db: {"total_deleted": 0}),
    ("_run_substrate_prune", "services.substrate_health.prune_substrate_metrics", lambda db, retention_days: 0),
])
def test_the_sync_sweeps_run_off_the_workers_loop(monkeypatch, method, target, work):
    import threading

    threads = []

    def recorded(*args, **kwargs):
        threads.append(threading.get_ident())
        return work(*args, **kwargs)

    monkeypatch.setattr(target, recorded)
    monkeypatch.setattr(memory_jobs, "_on_own_session", lambda fn: fn(object()))

    async def run():
        await getattr(memory_jobs.MemoryJobScheduler(), method)()
        return threading.get_ident()

    loop_thread = asyncio.run(run())
    assert threads and threads[0] != loop_thread


def test_the_checkpoint_candidates_are_read_off_the_workers_loop(monkeypatch):
    import threading

    threads = []
    monkeypatch.setattr(memory_jobs, "_checkpoint_candidates",
                        lambda db: threads.append(threading.get_ident()) or [])
    monkeypatch.setattr(memory_jobs, "_on_own_session", lambda fn: fn(object()))
    monkeypatch.setattr("core.database.database.SessionLocal", lambda: SimpleNamespace(close=lambda: None))

    async def run():
        await memory_jobs.MemoryJobScheduler()._run_thread_checkpoints()
        return threading.get_ident()

    loop_thread = asyncio.run(run())
    assert threads and threads[0] != loop_thread


@pytest.mark.parametrize("method, target", [
    ("_run_audit_retention", "services.audit_retention.sweep_expired_audit_logs"),
    ("_run_archival", "services.memory_archival_job.MemoryArchivalJob"),
])
def test_a_failed_run_is_raised_so_the_queue_retries_it(monkeypatch, method, target):
    def broken(*args, **kwargs):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(target, broken)
    monkeypatch.setattr(memory_jobs, "_on_own_session", lambda fn: fn(object()))
    with pytest.raises(RuntimeError, match="store unavailable"):
        asyncio.run(getattr(memory_jobs.MemoryJobScheduler(), method)())
