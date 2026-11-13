"""PRD-276 Wave 1b: a run callback is a durable job.

The job's id is the delivery id, so a retry after a worker died mid-send carries the same id, and
a receiver that drops ids it has seen records one effect. A send happens only under the job's
fence: a worker whose claim the sweep cleared, or whose workspace closed, sends nothing. Tickets
end in many places, so a sweep queues a job for each callback still pending well after its ticket
ended; that includes, once, what was pending at cutover. A ticket still running, or one that just
ended, is left alone, and a callback is never queued twice. Real database; the receiver is a fake.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import text

from core.jobs import queue
from core.jobs.registry import kind_of
from core.security.web_access import OutboundTarget
from services import run_callbacks as rc

PUBLIC = OutboundTarget(ok=True, reason="", host="hooks.example.com", ip="93.184.216.34", port=443, scheme="https")


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url
    from core.models.jobs import Job
    from core.models.run_callbacks import RunCallback, WorkspaceSigningSecret

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    for model in (WorkspaceSigningSecret, RunCallback, Job):
        model.__table__.create(eng, checkfirst=True)
    yield eng
    eng.dispose()


@pytest.fixture
def tickets(new_session):
    """Five tickets with callbacks: how long ago each ended (None: still running)."""
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w1b-callbacks')"), {"id": ws})
    made = {"ws": ws}
    for name, status, ended_ago, callback in (("cut_off", "done", 900, "pending"), ("failed_run", "failed", 900, "pending"),
                                              ("just_ended", "done", 30, "pending"), ("running", "in_progress", None, "pending"),
                                              ("delivered", "done", 900, "delivered")):
        task_id = s.execute(text(
            "INSERT INTO board_tasks (workspace_id, title, status, result, completed_at) VALUES (CAST(:w AS uuid), :t, "
            ":st, 'Renewal reminder drafted.', CASE WHEN CAST(:ago AS integer) IS NULL THEN NULL "
            "ELSE now() - make_interval(secs => :ago) END) RETURNING id"),
            {"w": ws, "t": name, "st": status, "ago": ended_ago}).scalar()
        s.execute(text("INSERT INTO run_callbacks (id, workspace_id, task_id, url, status) VALUES "
                       "(gen_random_uuid(), CAST(:w AS uuid), :t, 'https://hooks.example.com/runs', :st)"),
                  {"w": ws, "t": task_id, "st": callback})
        made[name] = task_id
    s.commit()
    s.close()
    yield made
    s = new_session.sweep()
    for table in ("jobs.jobs", "run_callbacks", "workspace_signing_secrets", "board_tasks"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


@pytest.fixture
def receiver(monkeypatch):
    """A receiver that answers each POST with the next code given, and keeps one effect per
    delivery id, as the callback contract asks receivers to; answers are set with ``answer``."""
    got, effects, answers = [], {}, [200]

    async def resolve_async(url, *, enforce_switch=True):
        return PUBLIC

    def handle(request):
        got.append(request)
        code = answers.pop(0) if len(answers) > 1 else answers[0]
        if 200 <= code < 300:
            effects.setdefault(request.headers[rc.DELIVERY_HEADER], json.loads(request.content))
        return httpx.Response(code)

    monkeypatch.setattr("core.security.web_access.resolve_outbound_async", resolve_async)
    monkeypatch.setattr(rc, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handle),
                                                                  follow_redirects=False))
    return SimpleNamespace(got=got, effects=effects, answer=lambda *codes: answers.__setitem__(slice(None), codes))


def _do(new_session, work, *args):
    s = new_session()
    try:
        out = work(s, *args)
        s.commit()
        return out
    finally:
        s.close()


def _claim(new_session):
    return _do(new_session, queue.claim, kind_of(rc.DELIVER_KIND), "w-test", 10)


def _jobs(new_session, ws):
    return _do(new_session, lambda s: s.execute(text(
        "SELECT idempotency_key, status FROM jobs.jobs WHERE kind = :k AND workspace_id = CAST(:w AS uuid) "
        "ORDER BY idempotency_key"), {"k": rc.DELIVER_KIND, "w": ws}).all())


def _expire_and_sweep(new_session, job_id):
    _do(new_session, lambda s: s.execute(text("UPDATE jobs.jobs SET lease_until = clock_timestamp() - interval '1 second' "
                                              "WHERE id = :id"), {"id": job_id}))
    _do(new_session, queue.sweep_expired)
    _do(new_session, lambda s: s.execute(text("UPDATE jobs.jobs SET run_at = clock_timestamp() WHERE id = :id"),
                                         {"id": job_id}))


def test_the_sweep_queues_only_callbacks_left_pending_after_their_ticket_ended(new_session, tickets):
    assert asyncio.run(rc.queue_stale_callbacks()) >= 2
    keys = {key for key, _ in _jobs(new_session, tickets["ws"])}
    assert keys == {rc.job_key(tickets["cut_off"]), rc.job_key(tickets["failed_run"])}
    asyncio.run(rc.queue_stale_callbacks())  # a callback with a job is never queued again
    assert len(_jobs(new_session, tickets["ws"])) == 2


def test_a_callback_pending_at_cutover_is_queued_once_and_delivered(new_session, tickets, receiver):
    asyncio.run(rc.queue_stale_callbacks())
    for job in _claim(new_session):
        assert asyncio.run(rc.deliver_run_callback(job)) == "delivered"
        _do(new_session, queue.complete, job)
    statuses = _do(new_session, lambda s: dict(s.execute(text(
        "SELECT task_id, status FROM run_callbacks WHERE workspace_id = CAST(:w AS uuid)"), {"w": tickets["ws"]}).all()))
    assert statuses[tickets["cut_off"]] == statuses[tickets["failed_run"]] == "delivered"
    assert statuses[tickets["running"]] == statuses[tickets["just_ended"]] == "pending"


def test_a_retry_after_a_worker_died_mid_send_keeps_the_delivery_id(new_session, tickets, receiver, monkeypatch):
    record, died = rc._record, []

    def dies_once(*args):  # the receiver accepted; the process is killed before the outcome is written
        if not died:
            died.append(True)
            raise SystemExit("killed")
        return record(*args)

    monkeypatch.setattr(rc, "_record", dies_once)
    _do(new_session, rc.queue_if_pending, tickets["cut_off"])
    (first,) = _claim(new_session)
    with pytest.raises(SystemExit):
        asyncio.run(rc.deliver_run_callback(first))
    _expire_and_sweep(new_session, first.id)
    (second,) = _claim(new_session)
    assert second.id == first.id and asyncio.run(rc.deliver_run_callback(second)) == "delivered"
    ids = [r.headers[rc.DELIVERY_HEADER] for r in receiver.got]
    assert ids == [str(first.id)] * 2 and len(receiver.effects) == 1  # two deliveries, one effect


def test_a_retryable_answer_requeues_the_job_and_records_the_attempt(new_session, tickets, receiver):
    receiver.answer(503)
    _do(new_session, rc.queue_if_pending, tickets["cut_off"])
    (job,) = _claim(new_session)
    with pytest.raises(rc.NotYetDelivered):
        asyncio.run(rc.deliver_run_callback(job))
    _do(new_session, queue.fail, job, "503")
    row = _do(new_session, lambda s: s.execute(text("SELECT status, attempts, last_status_code FROM run_callbacks "
                                                    "WHERE task_id = :t"), {"t": tickets["cut_off"]}).first())
    assert tuple(row) == ("pending", 1, 503) and _jobs(new_session, tickets["ws"])[0][1] == "queued"


def test_a_worker_that_lost_its_claim_or_its_workspace_sends_nothing(new_session, tickets, receiver):
    _do(new_session, rc.queue_if_pending, tickets["cut_off"])
    (stale,) = _claim(new_session)
    _expire_and_sweep(new_session, stale.id)  # the sweep cleared its claim
    assert asyncio.run(rc.deliver_run_callback(stale)) is None
    (live,) = _claim(new_session)
    _do(new_session, lambda s: s.execute(text("UPDATE workspaces SET deleted_at = now() WHERE id = CAST(:w AS uuid)"),
                                         {"w": tickets["ws"]}))
    assert asyncio.run(rc.deliver_run_callback(live)) is None and receiver.got == []


def test_the_shared_scheduler_runs_the_sweep_once_it_starts():
    added = []
    scheduler = SimpleNamespace(add_job=lambda fn, trigger, **kw: added.append((fn, trigger, kw)))

    @rc.schedules_the_callback_sweep
    def start(self):
        self._scheduler = scheduler

    from datetime import timedelta

    from core.jobs import schedules

    start(SimpleNamespace())  # PRD-276 Wave 2c: a platform schedule queues the sweep
    (timer, trigger, kw), (_, catch_up, _) = added
    assert timer is schedules.fire and catch_up == "date"
    assert trigger.interval == timedelta(seconds=rc.SWEEP_EVERY_SECONDS)
    assert kw["id"] == schedules.job_id(rc.SWEEP_KIND, schedules.PLATFORM)


def test_the_unified_scheduler_carries_it():
    from services.scheduler import UnifiedScheduler

    assert UnifiedScheduler.start.__code__.co_name == "started"
