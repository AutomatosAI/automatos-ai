"""PRD-276 Wave 2b: the scheduler queues; each schedule has a cursor; missed slots are caught up.

The PRD's Wave 2 tests, on the real database:
* two schedulers firing the same slot queue one job;
* two schedules with the same slot each queue their own job;
* a scheduler stopped across three slots queues them by each kind's policy on start-up: ``all``
  (capped, the newest kept, the rest dropped and recorded), ``latest`` or ``none``;
and: a new schedule never backfills; a slot is queued and the cursor moved in one transaction;
a purge deletes a workspace's cursors with its jobs; credit metering is three platform schedules;
every worker process loads every kind's handlers.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text

from core.jobs import schedules
from core.jobs.registry import CATCH_UP_ALL, CATCH_UP_LATEST, CATCH_UP_NONE, forget, job_handler

T0 = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)  # on the ten-minute grid from ANCHOR
KINDS = {CATCH_UP_ALL: "test_prd276_w2b.all", CATCH_UP_LATEST: "test_prd276_w2b.latest",
         CATCH_UP_NONE: "test_prd276_w2b.none"}


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url
    from core.models.jobs import Job, JobSchedule

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    for model in (Job, JobSchedule):  # a test database built before the tables
        model.__table__.create(eng, checkfirst=True)
    yield eng
    eng.dispose()


@pytest.fixture
def kinds(new_session):
    async def nothing(job):
        return None

    for policy, kind in KINDS.items():
        job_handler(kind, catch_up=policy)(nothing)
    yield KINDS
    for kind in KINDS.values():
        forget(kind)
    s = new_session.sweep()
    s.execute(text("DELETE FROM jobs.jobs WHERE kind LIKE 'test_prd276_w2b.%'"))
    s.execute(text("DELETE FROM jobs.job_schedules WHERE kind LIKE 'test_prd276_w2b.%'"))
    s.commit()
    s.close()


def _do(new_session, work, *args, **kwargs):
    s = new_session()
    try:
        out = work(s, *args, **kwargs)
        s.commit()
        return out
    finally:
        s.close()


def _every_ten(kind, schedule_id="report:acme"):
    return schedules.Schedule(kind, schedule_id, schedules.interval(minutes=10))


def _queued(new_session, kind):
    return _do(new_session, lambda s: [row[0] for row in s.execute(text(
        "SELECT payload->>'slot' FROM jobs.jobs WHERE kind = :k ORDER BY payload->>'slot'"), {"k": kind}).all()])


def _cursor(new_session, spec):
    return _do(new_session, lambda s: s.execute(text(
        "SELECT last_slot FROM jobs.job_schedules WHERE kind = :k AND schedule_id = :sid"),
        {"k": spec.kind, "sid": spec.schedule_id}).scalar())


def test_interval_slots_are_the_same_in_every_process():
    one, two = schedules.interval(minutes=10), schedules.interval(minutes=10)
    after, until = T0 + timedelta(minutes=3), T0 + timedelta(minutes=34)
    expected = [T0 + timedelta(minutes=m) for m in (10, 20, 30)]
    assert schedules.due_slots(one, after, until, 24) == schedules.due_slots(two, after, until, 24) == (expected, 3)


def test_each_policy_chooses_its_slots():
    slots = [T0 + timedelta(minutes=m) for m in (10, 20, 30)]
    now = T0 + timedelta(minutes=30, seconds=5)
    assert schedules.chosen(CATCH_UP_ALL, slots, 3, now, 24) == (slots, 0)
    assert schedules.chosen(CATCH_UP_ALL, slots[1:], 3, now, 2) == (slots[1:], 1)  # the newest kept
    assert schedules.chosen(CATCH_UP_ALL, slots, 3, now, 0) == (slots[-1:], 2)  # a cap below 1 is 1
    assert schedules.chosen(CATCH_UP_LATEST, slots[-1:], 3, now, 24) == (slots[-1:], 0)
    assert schedules.chosen(CATCH_UP_NONE, slots[-1:], 3, now, 24) == (slots[-1:], 0)  # due right now
    assert schedules.chosen(CATCH_UP_NONE, slots[-1:], 3, now + timedelta(minutes=5), 24) == ([], 0)  # missed


def test_a_long_backlog_is_counted_whole_and_its_latest_slot_is_the_real_latest():
    every_second = schedules.interval(seconds=1)
    after, until = T0, T0 + timedelta(hours=30)  # 108,000 slots: more than any one scan used to hold
    kept, total = schedules.due_slots(every_second, after, until, 1)
    assert total == 108_000 and kept == [until]


def test_a_schedule_id_too_long_for_its_job_keys_is_refused():
    with pytest.raises(ValueError, match="at most"):
        schedules.Schedule("test_prd276_w2b.latest", "x" * (schedules.MAX_SCHEDULE_ID + 1), schedules.interval(minutes=1))
    longest = schedules.Schedule("k", "x" * schedules.MAX_SCHEDULE_ID, schedules.interval(minutes=1))
    assert len(f"{longest.schedule_id}:{datetime.max.replace(tzinfo=timezone.utc).isoformat()}") <= 255


def test_a_new_schedule_starts_now_and_never_backfills(new_session, kinds):
    spec = _every_ten(kinds[CATCH_UP_ALL])
    assert _do(new_session, schedules.tick, spec, T0 + timedelta(minutes=3)) == 0
    assert _cursor(new_session, spec) == T0 + timedelta(minutes=3) and _queued(new_session, spec.kind) == []


@pytest.mark.parametrize("policy, expected", [(CATCH_UP_ALL, ["12:10", "12:20", "12:30"]),
                                              (CATCH_UP_LATEST, ["12:30"]), (CATCH_UP_NONE, [])])
def test_a_scheduler_stopped_across_three_slots_queues_them_by_policy(new_session, kinds, policy, expected):
    spec = _every_ten(kinds[policy])
    _do(new_session, schedules.tick, spec, T0 + timedelta(minutes=3))  # declared at 12:03
    later = T0 + timedelta(minutes=36)  # back at 12:36: 12:10, 12:20 and 12:30 passed with no scheduler
    _do(new_session, schedules.tick, spec, later)
    assert [slot[11:16] for slot in _queued(new_session, spec.kind)] == expected
    assert _cursor(new_session, spec) == T0 + timedelta(minutes=30)  # past every missed slot, whatever was queued


def test_two_schedulers_firing_the_same_slot_queue_one_job(new_session, kinds):
    spec = _every_ten(kinds[CATCH_UP_LATEST])
    _do(new_session, schedules.tick, spec, T0 + timedelta(minutes=3))
    fired = T0 + timedelta(minutes=10, seconds=1)
    assert _do(new_session, schedules.tick, spec, fired) == 1
    assert _do(new_session, schedules.tick, spec, fired) == 0  # the cursor already moved
    _do(new_session, lambda s: s.execute(text("UPDATE jobs.job_schedules SET last_slot = :t WHERE kind = :k"),
                                         {"t": T0, "k": spec.kind}))  # as if the second never saw it move
    assert _do(new_session, schedules.tick, spec, fired) == 0 and len(_queued(new_session, spec.kind)) == 1


def test_two_schedules_with_the_same_slot_each_queue_their_own_job(new_session, kinds):
    kind = kinds[CATCH_UP_LATEST]
    for schedule_id in ("report:acme", "report:globex"):
        spec = _every_ten(kind, schedule_id)
        _do(new_session, schedules.tick, spec, T0 + timedelta(minutes=3))
        assert _do(new_session, schedules.tick, spec, T0 + timedelta(minutes=10, seconds=1)) == 1
    assert len(_queued(new_session, kind)) == 2


def test_slots_past_the_cap_are_skipped_logged_and_recorded(new_session, kinds, monkeypatch):
    from config import config

    errors = MagicMock()
    monkeypatch.setattr("core.utils.exception_telemetry.record_error", errors)
    monkeypatch.setattr(config, "JOBS_CATCH_UP_MAX", 2)
    spec = _every_ten(kinds[CATCH_UP_ALL])
    _do(new_session, schedules.tick, spec, T0 + timedelta(minutes=3))
    assert _do(new_session, schedules.tick, spec, T0 + timedelta(minutes=36)) == 2
    assert [slot[11:16] for slot in _queued(new_session, spec.kind)] == ["12:20", "12:30"]
    assert errors.call_args.kwargs["operation"] == "catch_up" and errors.call_args.kwargs["extra"]["dropped"] == 1


def test_a_failed_tick_queues_nothing_and_leaves_the_cursor(new_session, kinds, monkeypatch):
    spec = _every_ten(kinds[CATCH_UP_ALL])
    _do(new_session, schedules.tick, spec, T0 + timedelta(minutes=3))

    def refuses(*args, **kwargs):
        raise RuntimeError("database went away")

    monkeypatch.setattr("core.jobs.queue.enqueue", refuses)
    with pytest.raises(RuntimeError):
        _do(new_session, schedules.tick, spec, T0 + timedelta(minutes=36))
    assert _cursor(new_session, spec) == T0 + timedelta(minutes=3) and _queued(new_session, spec.kind) == []


def test_declaring_a_schedule_wakes_it_on_its_trigger_and_once_at_once(kinds):
    added = []
    scheduler = SimpleNamespace(add_job=lambda fn, trigger, **kw: added.append((fn, trigger, kw)))
    trigger = schedules.interval(minutes=10)
    schedules.schedule(scheduler, kinds[CATCH_UP_LATEST], schedules.PLATFORM, trigger)
    try:
        (timer, catch_up) = added
        assert timer[0] is schedules.fire and timer[1] is trigger and timer[2]["coalesce"]
        assert catch_up[1] == "date" and catch_up[2]["args"] == [kinds[CATCH_UP_LATEST], schedules.PLATFORM]
        with pytest.raises(KeyError):
            schedules.schedule(scheduler, "test_prd276_w2b.nobody", schedules.PLATFORM, trigger)
    finally:
        schedules.forget(kinds[CATCH_UP_LATEST], schedules.PLATFORM)
    assert asyncio.run(schedules.fire(kinds[CATCH_UP_LATEST], schedules.PLATFORM)) == 0  # no longer declared


def test_a_closed_workspaces_schedule_queues_nothing_and_its_purge_deletes_the_rest(new_session, kinds):
    from core.jobs.erasure import jobs_stop_first

    ws = str(uuid.uuid4())
    _do(new_session, lambda s: s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w2b')"),
                                         {"id": ws}))
    spec = schedules.Schedule(kinds[CATCH_UP_LATEST], f"report:{ws}", schedules.interval(minutes=10), ws)
    _do(new_session, schedules.tick, spec, T0 + timedelta(minutes=3))
    assert _do(new_session, schedules.tick, spec, T0 + timedelta(minutes=10, seconds=1)) == 1
    _do(new_session, lambda s: s.execute(text("UPDATE workspaces SET deleted_at = now() WHERE id = CAST(:id AS uuid)"),
                                         {"id": ws}))
    assert _do(new_session, schedules.tick, spec, T0 + timedelta(minutes=20, seconds=1)) == 0  # closed: nothing
    left = []

    @jobs_stop_first
    def first_data_step(workspace_id, result):
        left.append(_do(new_session, lambda s: s.execute(text(
            "SELECT (SELECT count(*) FROM jobs.job_schedules WHERE workspace_id = CAST(:w AS uuid)) + "
            "(SELECT count(*) FROM jobs.jobs WHERE workspace_id = CAST(:w AS uuid))"), {"w": ws}).scalar()))
        return True

    try:
        assert first_data_step(uuid.UUID(ws), SimpleNamespace(skipped_reason=None)) is True and left == [0]
        assert _do(new_session, schedules.tick, spec, T0 + timedelta(minutes=30, seconds=1)) == 0
        assert _cursor(new_session, spec) is None  # the still-declared schedule made no cursor again
    finally:
        _do(new_session, lambda s: s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws}))


def test_credit_metering_is_three_platform_schedules_with_their_policies():
    from core.jobs.registry import kind_of
    from services.credits import jobs as credit_jobs

    added = []
    credit_jobs.register_credit_jobs(SimpleNamespace(add_job=lambda fn, trigger, **kw: added.append(kw["id"])))
    try:
        assert {schedules.job_id(kind, schedules.PLATFORM) for kind in
                (credit_jobs.METER_KIND, credit_jobs.OVERHEAD_KIND, credit_jobs.EXPIRY_KIND)} <= set(added)
        policies = {kind: kind_of(kind).catch_up for kind in
                    (credit_jobs.METER_KIND, credit_jobs.OVERHEAD_KIND, credit_jobs.EXPIRY_KIND)}
        assert policies == {credit_jobs.METER_KIND: CATCH_UP_LATEST, credit_jobs.OVERHEAD_KIND: CATCH_UP_ALL,
                            credit_jobs.EXPIRY_KIND: CATCH_UP_LATEST}
    finally:
        for kind in (credit_jobs.METER_KIND, credit_jobs.OVERHEAD_KIND, credit_jobs.EXPIRY_KIND):
            schedules.forget(kind, schedules.PLATFORM)


def test_a_nightly_overhead_slot_records_the_day_before_it(monkeypatch):
    from services.credits import jobs as credit_jobs

    days = []
    monkeypatch.setattr(credit_jobs, "overhead_nightly", days.append)
    job = SimpleNamespace(payload={"slot": "2026-11-10T03:15:00+00:00"})
    asyncio.run(credit_jobs.run_overhead(job))
    assert [d.isoformat() for d in days] == ["2026-11-09"]


def test_every_worker_loads_every_kinds_handlers(monkeypatch):
    from core.jobs import worker
    from core.jobs.registry import registered

    loaded = []
    monkeypatch.setattr("importlib.import_module", loaded.append)
    worker.load_handlers()
    assert tuple(loaded) == worker.HANDLER_MODULES and "services.credits.jobs" in loaded
    for module in worker.HANDLER_MODULES:
        __import__(module)
    assert {"run_callback.deliver", "widget_destination.deliver", "credits.meter"} <= set(registered())


def test_a_handler_module_that_fails_to_load_is_retried_until_it_loads(monkeypatch):
    from core.jobs import worker

    attempts = []

    def flaky(module):
        attempts.append(module)
        if len(attempts) == 1:
            raise ImportError("a dependency is not ready yet")

    monkeypatch.setattr("importlib.import_module", flaky)
    assert worker.load_handlers(("services.credits.jobs",)) == ("services.credits.jobs",)
    monkeypatch.setattr(worker, "RETRY_HANDLERS_EVERY_TICKS", 1)

    async def idle(self):
        return 0

    monkeypatch.setattr(worker.JobWorker, "tick", idle)

    async def run():
        jobs = worker.JobWorker("w-test", missing_handlers=("services.credits.jobs",))
        stop = asyncio.Event()
        task = asyncio.create_task(jobs.run(stop, 0.5, 7))
        await asyncio.sleep(0.1)
        stop.set()
        await task
        return jobs.missing_handlers

    assert asyncio.run(run()) == () and attempts == ["services.credits.jobs"] * 2


def test_a_scheduler_says_when_nothing_would_run_its_queue(monkeypatch):
    from config import config
    from services import scheduler_boot

    errors = MagicMock()
    monkeypatch.setattr("core.utils.exception_telemetry.record_error", errors)
    monkeypatch.setattr(config, "PROCESS_ROLES", "api,worker,scheduler")
    monkeypatch.setattr(config, "JOBS_ENABLED", True)
    assert scheduler_boot.check_the_queue_runs() is True and not errors.called
    monkeypatch.setattr(config, "PROCESS_ROLES", "scheduler")
    assert scheduler_boot.check_the_queue_runs() is False and not errors.called  # another process may run it
    monkeypatch.setattr(config, "JOBS_ENABLED", False)
    assert scheduler_boot.check_the_queue_runs() is False
    assert errors.call_args.kwargs["operation"] == "start_scheduler"
