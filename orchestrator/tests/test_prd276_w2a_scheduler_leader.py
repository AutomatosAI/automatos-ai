"""PRD-276 Wave 2a: one scheduler across processes and containers.

A Postgres advisory lock, held on a connection of its own, picks the process that runs the shared
scheduler. A second process can't take it while it is held, and takes it once the holder releases
it or its connection dies. The leader starts the scheduled services once, pauses the scheduler
when it loses the lock and resumes it when it holds the lock again. The scheduler keeps its jobs
in memory, never in Redis. Boot no longer uses the ``/tmp`` file lock, and a scheduled service
that fails to start doesn't stop the rest. The lock tests run on the real database, on a key of
their own.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from core.jobs import leadership
from services import scheduler_boot

TEST_KEY = 2_760_999_042  # never the scheduler's own key, so a running local app is untouched


@pytest.fixture
def locks():
    made = []

    def make():
        lock = leadership.SchedulerLock(leadership.database_url(), key=TEST_KEY)
        made.append(lock)
        return lock

    yield make
    for lock in made:
        lock.release()


def test_one_process_holds_the_lock_and_another_takes_it_once_released(locks):
    first, second = locks(), locks()
    assert first.try_acquire() is True and first.held() is True
    assert second.try_acquire() is False and second.held() is False
    first.release()
    assert first.held() is False and second.try_acquire() is True


def test_a_holder_whose_connection_dies_loses_the_lock_and_another_takes_it(locks):
    first, second = locks(), locks()
    assert first.try_acquire()
    pid = first._conn.execute(text("SELECT pg_backend_pid()")).scalar()
    killer = create_engine(leadership.database_url())
    with killer.connect() as conn:
        assert conn.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid}).scalar()
    killer.dispose()
    assert first.held() is False
    for _ in range(20):  # the killed backend frees its lock as it exits, a moment after the kill
        if second.try_acquire():
            break
        time.sleep(0.1)
    assert second.held() is True


class _FakeLock:
    """Answers ``held`` and ``try_acquire`` from a script of whether this process holds the lock."""

    def __init__(self, script):
        self.script, self.released = list(script), False

    def held(self):
        return self.script[0]

    def try_acquire(self):
        return self.script[0]

    def release(self):
        self.released = True


def test_the_leader_starts_once_pauses_when_it_loses_the_lock_and_resumes_when_it_holds_it():
    lock, started, paused = _FakeLock([False]), [], []

    async def start():
        started.append(True)

    lead = leadership.Leadership(lock, start, paused.append)

    async def checks():
        seen = []
        for holds in (False, True, True, False, False, True):
            lock.script[0] = holds
            seen.append(await lead.check())
        return seen

    assert asyncio.run(checks()) == [False, True, True, False, False, True]
    assert started == [True] and paused == [True, False]  # paused once on the loss, resumed once


def test_the_loop_gives_the_lock_back_when_stopped():
    lock = _FakeLock([True])

    async def start():
        return None

    lead = leadership.Leadership(lock, start, lambda paused: None)

    async def run():
        stop = asyncio.Event()
        task = asyncio.create_task(lead.run(stop, 0.01))
        await asyncio.sleep(0.05)
        stop.set()
        await task

    asyncio.run(run())
    assert lead.started and lock.released


def test_two_schedulers_one_leads_and_the_other_takes_over(monkeypatch):
    from services.scheduler import UnifiedScheduler

    def on_test_key(start_services, set_paused, lock=None):
        return leadership.Leadership(leadership.SchedulerLock(leadership.database_url(), key=TEST_KEY),
                                     start_services, set_paused)

    monkeypatch.setattr(leadership, "leadership_for", on_test_key)
    registered = []

    async def services(scheduler):
        registered.append(scheduler)

    async def run():
        first, second = UnifiedScheduler(), UnifiedScheduler()
        try:
            assert await first.lead(services, 60) is True and await second.lead(services, 60) is False
            assert first.apscheduler is not None and second.apscheduler is None
            first.stop()  # the holder goes: its lock is given back
            assert await second._leadership.check() is True and second.apscheduler is not None
        finally:
            first.stop()
            second.stop()

    asyncio.run(run())
    assert len(registered) == 2


def test_the_scheduler_keeps_its_jobs_in_memory_even_with_redis_configured(monkeypatch):
    from apscheduler.jobstores.memory import MemoryJobStore

    from config import config
    from services.scheduler import UnifiedScheduler

    monkeypatch.setattr(type(config), "REDIS_URL", property(lambda self: "redis://localhost:6379/0"))
    monkeypatch.setattr("services.credits.jobs.register_credit_jobs", lambda scheduler: None)

    async def run():
        unified = UnifiedScheduler()
        unified.start()
        try:
            assert isinstance(unified.apscheduler._jobstores["default"], MemoryJobStore)
            unified.set_paused(True)
            assert unified.apscheduler.state == 2  # STATE_PAUSED
            unified.set_paused(False)
            assert unified.apscheduler.state == 1  # STATE_RUNNING
        finally:
            unified.stop()

    asyncio.run(run())


def test_only_a_process_with_the_scheduler_role_wants_a_scheduler(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "HEARTBEAT_ENABLED", True)
    monkeypatch.setattr(config, "PROCESS_ROLES", "api,worker")
    assert scheduler_boot.scheduler_wanted() is False
    monkeypatch.setattr(config, "PROCESS_ROLES", "api,worker,scheduler")
    assert scheduler_boot.scheduler_wanted() is True


def test_every_enabled_service_starts_in_order_and_one_failing_stops_none(monkeypatch):
    from config import config

    started = []

    def service(name, fails=False):
        async def start(scheduler):
            if fails:
                raise RuntimeError(f"{name} could not start")
            started.append(name)

        return lambda: SimpleNamespace(start=start)

    modules = {module: SimpleNamespace(**{getter: service(name, fails=name == "HeartbeatService")})
               for _, name, module, getter in scheduler_boot._BEFORE_SCHEDULED_TASKS + scheduler_boot._AFTER_SCHEDULED_TASKS}
    monkeypatch.setattr(scheduler_boot.importlib, "import_module", lambda module: modules[module])

    async def noted(*args):
        started.append("scheduled tasks and reconcile")

    monkeypatch.setattr(scheduler_boot, "_load_scheduled_tasks", noted)
    monkeypatch.setattr(scheduler_boot, "_start_reconcile", lambda scheduler: asyncio.sleep(0))
    for switch, *_ in scheduler_boot._BEFORE_SCHEDULED_TASKS + scheduler_boot._AFTER_SCHEDULED_TASKS:
        if switch:
            monkeypatch.setattr(config, switch, switch != "HARNESS_ENABLED")
    asyncio.run(scheduler_boot.start_scheduled_services(object()))
    assert started == ["PlaybookSchedulerService", "TaskReconciler", "WatchTicker", "MemoryJobScheduler",
                       "FieldPromotionJobScheduler", "EdgeBuilderScheduler", "ComposioSyncScheduler",
                       "scheduled tasks and reconcile", "CoordinatorService"]


def test_boot_takes_the_scheduler_lock_and_no_longer_uses_a_file_lock():
    import main

    phase_2 = inspect.getsource(main._boot_phase_2_extensions)
    assert "start_scheduler_leadership" in phase_2 and "fcntl" not in inspect.getsource(main)
