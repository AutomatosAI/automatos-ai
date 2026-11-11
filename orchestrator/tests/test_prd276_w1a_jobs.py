"""PRD-276 Wave 1a: the durable job queue (``core/jobs``), proven on the real database.

A job is queued once per (kind, workspace, key); a worker claims it with a fresh claim id and a lease;
every write about it is fenced by that claim, so a stale worker's completion matches nothing once the
sweep has cleared its claim; an expired lease is a failed attempt, and a job whose lease keeps
expiring becomes dead at its limit; a closed workspace's jobs are never claimed, and its purge
deletes them before its data; a workspace's managers retry or clear its dead jobs.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from core.jobs import queue
from core.jobs.registry import forget, job_handler, kind_of

KIND = "test_prd276.echo"
FLAKY = "test_prd276.flaky"
SEEN = []


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url
    from core.models.jobs import Job

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    Job.__table__.create(eng, checkfirst=True)  # a test database built before the table
    yield eng
    eng.dispose()


@pytest.fixture
def kinds():
    async def echo(job):
        SEEN.append(job.payload)

    async def flaky(job):
        raise RuntimeError("receiver down")

    job_handler(KIND, backoff=(0,), max_attempts=2, lease_seconds=30, workspace_concurrency=1)(echo)
    job_handler(FLAKY, backoff=(60,), max_attempts=2, lease_seconds=30)(flaky)
    SEEN.clear()
    yield
    forget(KIND)
    forget(FLAKY)


@pytest.fixture
def workspaces(new_session, kinds):
    made = []

    def make(deleted=False):
        ws = str(uuid.uuid4())
        s = new_session()
        s.execute(text("INSERT INTO workspaces (id, name, deleted_at) VALUES (CAST(:id AS uuid), 'prd276', "
                       "CASE WHEN :d THEN now() END)"), {"id": ws, "d": deleted})
        s.commit()
        s.close()
        made.append(ws)
        return ws

    yield make
    s = new_session.sweep()
    s.execute(text("DELETE FROM jobs WHERE kind LIKE 'test_prd276.%'"))
    for ws in made:
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def _do(new_session, fn, *args, **kwargs):
    s = new_session()
    try:
        out = fn(s, *args, **kwargs)
        s.commit()
        return out
    finally:
        s.close()


def _claim(new_session, kind=KIND, limit=10):
    return _do(new_session, queue.claim, kind_of(kind), "w-test", limit)


def _expire(new_session, job_id):
    _do(new_session, lambda s: s.execute(text("UPDATE jobs SET lease_until = clock_timestamp() - interval '1 second' "
                                              "WHERE id = :id"), {"id": job_id}))


def _row(new_session, job_id):
    return _do(new_session, lambda s: s.execute(text("SELECT status, attempts, claim_id, last_error FROM jobs "
                                                     "WHERE id = :id"), {"id": job_id}).first())


def test_a_key_queues_once_per_kind_and_workspace_and_a_platform_job_counts_as_one(new_session, workspaces):
    a, b = workspaces(), workspaces()
    first = _do(new_session, queue.enqueue, KIND, {"n": 1}, workspace_id=a, key="ticket:42")
    assert first and _do(new_session, queue.enqueue, KIND, {"n": 2}, workspace_id=a, key="ticket:42") is None
    assert _do(new_session, queue.enqueue, KIND, {"n": 3}, workspace_id=b, key="ticket:42")
    assert _do(new_session, queue.enqueue, KIND, {}, key="platform:tick")
    assert _do(new_session, queue.enqueue, KIND, {}, key="platform:tick") is None
    with pytest.raises(KeyError):
        _do(new_session, queue.enqueue, "test_prd276.nobody", {})


def test_a_claim_takes_due_work_once_and_skips_a_closed_workspace(new_session, workspaces):
    live, closed = workspaces(), workspaces(deleted=True)
    due = _do(new_session, queue.enqueue, KIND, {}, workspace_id=live)
    _do(new_session, queue.enqueue, KIND, {}, workspace_id=closed)
    from datetime import datetime, timedelta, timezone
    _do(new_session, queue.enqueue, KIND, {}, workspace_id=live, key="later",
        run_at=datetime.now(timezone.utc) + timedelta(hours=1))
    (job,) = _claim(new_session)
    assert (job.id, job.attempts) == (due, 1) and job.claim_id is not None
    assert _row(new_session, due)[0] == "running" and _claim(new_session) == []


def test_a_workspace_runs_no_more_than_its_kinds_limit_at_once(new_session, workspaces):
    ws = workspaces()
    other = workspaces()
    for n in range(3):
        _do(new_session, queue.enqueue, KIND, {"n": n}, workspace_id=ws)
    _do(new_session, queue.enqueue, KIND, {"n": 9}, workspace_id=other)
    first = _claim(new_session)  # workspace_concurrency=1: one each, and the busy one doesn't block the other
    assert sorted(str(j.workspace_id) for j in first) == sorted([ws, other]) and _claim(new_session) == []


def test_a_stale_workers_writes_match_nothing_once_the_sweep_cleared_its_claim(new_session, workspaces):
    ws = workspaces()
    job_id = _do(new_session, queue.enqueue, KIND, {}, workspace_id=ws)
    (stale,) = _claim(new_session)
    _expire(new_session, job_id)
    assert _do(new_session, queue.fence, stale, 30) is True  # expired but unswept: still the only holder
    assert _do(new_session, queue.sweep_expired) == 0  # and the fence extended its lease past the sweep
    _expire(new_session, job_id)
    assert _do(new_session, queue.sweep_expired) >= 1
    status, attempts, claim_id, error = _row(new_session, job_id)
    assert (status, claim_id, error) == ("queued", None, queue.LEASE_EXPIRED)
    assert _do(new_session, queue.complete, stale) is False and _do(new_session, queue.fence, stale, 30) is False
    assert _do(new_session, queue.renew, stale, 30) is False and _row(new_session, job_id)[0] == "queued"


def test_a_job_whose_lease_keeps_expiring_becomes_dead_at_its_limit(new_session, workspaces):
    ws = workspaces()
    job_id = _do(new_session, queue.enqueue, KIND, {}, workspace_id=ws)
    for _ in range(2):  # max_attempts=2, backoff 0
        (job,) = _claim(new_session)
        _expire(new_session, job.id)
        _do(new_session, queue.sweep_expired)
    assert _row(new_session, job_id)[:2] == ("dead", 2) and _claim(new_session) == []


def test_a_failure_waits_its_backoff_then_dies_at_its_limit(new_session, workspaces):
    ws = workspaces()
    job_id = _do(new_session, queue.enqueue, FLAKY, {}, workspace_id=ws)
    (job,) = _claim(new_session, FLAKY)
    assert _do(new_session, queue.fail, job, "RuntimeError: receiver down") is True
    assert _row(new_session, job_id)[0] == "queued" and _claim(new_session, FLAKY) == []  # 60 s backoff
    _do(new_session, lambda s: s.execute(text("UPDATE jobs SET run_at = clock_timestamp() WHERE id = :id"),
                                         {"id": job_id}))
    (again,) = _claim(new_session, FLAKY)
    _do(new_session, queue.fail, again, "RuntimeError: receiver down")
    assert _row(new_session, job_id)[:2] == ("dead", 2)


def test_the_fence_extends_the_lease_and_refuses_a_closed_workspace(new_session, workspaces):
    ws = workspaces()
    job_id = _do(new_session, queue.enqueue, KIND, {}, workspace_id=ws)
    (job,) = _claim(new_session)
    _expire(new_session, job_id)
    assert _do(new_session, queue.fence, job, 120) is True
    left = _do(new_session, lambda s: s.execute(text("SELECT extract(epoch FROM lease_until - clock_timestamp()) "
                                                     "FROM jobs WHERE id = :id"), {"id": job_id}).scalar())
    assert 100 < left <= 120
    _do(new_session, lambda s: s.execute(text("UPDATE workspaces SET deleted_at = now() WHERE id = CAST(:ws AS uuid)"),
                                         {"ws": ws}))
    assert _do(new_session, queue.fence, job, 120) is False


def test_the_worker_runs_a_handler_and_records_a_failure(new_session, workspaces, monkeypatch):
    from core.jobs import worker as jobs_worker

    ws = workspaces()
    ok = _do(new_session, queue.enqueue, KIND, {"hello": "acme"}, workspace_id=ws)
    bad = _do(new_session, queue.enqueue, FLAKY, {}, workspace_id=ws)
    monkeypatch.setattr(jobs_worker, "_record", lambda job, exc: None)
    monkeypatch.setattr(jobs_worker, "registered", lambda: (KIND, FLAKY))

    async def one_pass():
        w = jobs_worker.JobWorker("w-test")
        assert await w.tick() == 2
        while w.inflight():
            await asyncio.sleep(0.05)

    asyncio.run(one_pass())
    assert SEEN == [{"hello": "acme"}] and _row(new_session, ok)[0] == "done"
    assert _row(new_session, bad)[0] == "queued" and "receiver down" in _row(new_session, bad)[3]


def test_a_purge_deletes_the_workspaces_jobs_before_its_data(new_session, workspaces):
    from core.jobs.erasure import jobs_stop_first

    ws = workspaces(deleted=True)
    _do(new_session, queue.enqueue, KIND, {}, workspace_id=ws)
    seen = []

    @jobs_stop_first
    def first_data_step(workspace_id, result):
        seen.append(_do(new_session, lambda s: s.execute(text("SELECT count(*) FROM jobs WHERE workspace_id = "
                                                               "CAST(:ws AS uuid)"), {"ws": ws}).scalar()))
        return True

    assert first_data_step(uuid.UUID(ws), SimpleNamespace(skipped_reason=None)) is True and seen == [0]


def test_a_workspaces_managers_retry_or_clear_its_dead_jobs_and_no_one_elses(new_session, workspaces):
    from fastapi import HTTPException

    from api import jobs as jobs_api

    mine, theirs = workspaces(), workspaces()
    ids = []
    for ws in (mine, mine, theirs):
        job_id = _do(new_session, queue.enqueue, KIND, {}, workspace_id=ws)
        _do(new_session, lambda s, i=job_id: s.execute(text("UPDATE jobs SET status = 'dead', attempts = 2, "
                                                            "finished_at = now() WHERE id = :id"), {"id": i}))
        ids.append(job_id)
    ctx = SimpleNamespace(workspace_id=uuid.UUID(mine))
    listed = _do(new_session, lambda s: jobs_api.list_dead_jobs(limit=50, ctx=ctx, db=s))
    assert {j["id"] for j in listed["jobs"]} == {str(ids[0]), str(ids[1])}
    _do(new_session, lambda s: jobs_api.retry_dead_job(ids[0], ctx=ctx, db=s))
    _do(new_session, lambda s: jobs_api.clear_dead_job(ids[1], ctx=ctx, db=s))
    assert _row(new_session, ids[0])[:2] == ("queued", 0) and _row(new_session, ids[1]) is None
    for call in (jobs_api.retry_dead_job, jobs_api.clear_dead_job):
        with pytest.raises(HTTPException) as other:
            _do(new_session, lambda s, c=call: c(ids[2], ctx=ctx, db=s))
        assert other.value.status_code == 404


def test_prune_deletes_old_finished_jobs(new_session, workspaces):
    ws = workspaces()
    job_id = _do(new_session, queue.enqueue, KIND, {}, workspace_id=ws)
    _do(new_session, lambda s: s.execute(text("UPDATE jobs SET status = 'done', finished_at = now() - "
                                              "interval '8 days' WHERE id = :id"), {"id": job_id}))
    assert _do(new_session, queue.prune, 7) >= 1 and _row(new_session, job_id) is None


def test_only_a_worker_role_process_starts_the_worker(monkeypatch):
    from config import config
    from core.jobs import worker as jobs_worker

    calls = []
    started = jobs_worker.also_starts_the_job_worker(lambda app, trust: calls.append(trust))

    async def boot(roles):
        monkeypatch.setattr(config, "PROCESS_ROLES", roles)
        app = SimpleNamespace(state=SimpleNamespace())
        started(app, True)
        task = getattr(app.state, "job_worker_task", None)
        if task:
            app.state.job_worker_stop.set()
            await task
        return task is not None

    assert asyncio.run(boot("api")) is False and asyncio.run(boot("api,worker,scheduler")) is True
    assert calls == [True, True]


def test_a_kinds_scrub_keys_last_only_until_the_job_ends(new_session, workspaces):
    sealed = "test_prd276.sealed"
    job_handler(sealed, backoff=(0,), max_attempts=2, scrub=("contact",))(lambda job: None)
    try:
        ws = workspaces()
        payload = lambda job_id: _do(new_session, lambda s: s.execute(  # noqa: E731
            text("SELECT payload FROM jobs WHERE id = :id"), {"id": job_id}).scalar())
        done = _do(new_session, queue.enqueue, sealed, {"contact": "sealed", "ref": 1}, workspace_id=ws, key="a")
        dies = _do(new_session, queue.enqueue, sealed, {"contact": "sealed", "ref": 2}, workspace_id=ws, key="b")
        first = {j.id: j for j in _claim(new_session, sealed)}
        _do(new_session, queue.complete, first[done])
        _do(new_session, queue.fail, first[dies], "boom")
        assert payload(done) == {"ref": 1} and payload(dies) == {"contact": "sealed", "ref": 2}  # requeued: kept
        (last,) = _claim(new_session, sealed)
        _do(new_session, queue.fail, last, "boom")
        assert _row(new_session, dies)[0] == "dead" and payload(dies) == {"ref": 2}
    finally:
        forget(sealed)
