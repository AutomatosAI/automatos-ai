"""PRD-276 §10-7 and §12.3: the queue in its own schema, its metrics, and claims that take turns.

* The table is ``jobs.jobs``. The migration moves ``public.jobs`` into the schema with its rows and
  indexes; on a database ``create_all`` built first, it copies any rows from the stray
  ``public.jobs`` across and drops it. Both paths run inside a transaction that is rolled back.
* The metrics read, per kind, the queued, running and dead counts and the oldest due job's wait,
  and register as two gauges only when OpenTelemetry is on.
* A claim takes each workspace's next job before any workspace's second.
Real database.
"""
from __future__ import annotations

import importlib.util
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from core.jobs import metrics as jobs_metrics
from core.jobs import queue
from core.jobs.registry import forget, job_handler, kind_of

KIND = "test_prd276_107.report"
VERSIONS = Path(__file__).resolve().parents[1] / "alembic" / "versions"


def _migration(name: str):
    spec = importlib.util.spec_from_file_location(name, VERSIONS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url
    from core.models.jobs import Job

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    Job.__table__.create(eng, checkfirst=True)  # its schema first (the before_create hook)
    yield eng
    eng.dispose()


@pytest.fixture
def workspaces(new_session):
    job_handler(KIND, backoff=(0,), max_attempts=2, workspace_concurrency=2)(lambda job: None)
    made = []

    def make():
        ws = str(uuid.uuid4())
        s = new_session()
        s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd276-107')"), {"id": ws})
        s.commit()
        s.close()
        made.append(ws)
        return ws

    yield make
    forget(KIND)
    s = new_session.sweep()
    s.execute(text("DELETE FROM jobs.jobs WHERE kind = :k"), {"k": KIND})
    for ws in made:
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
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


def test_the_model_lives_in_the_jobs_schema(engine):
    from core.models.jobs import JOBS_SCHEMA, Job

    assert Job.__table__.schema == JOBS_SCHEMA == "jobs" and Job.__table__.fullname == "jobs.jobs"
    with engine.connect() as conn:
        assert conn.execute(text("SELECT to_regclass('jobs.jobs') IS NOT NULL")).scalar()


def _rehearse(engine, setup):
    """Runs ``setup`` then the migration's upgrade in one transaction; answers what it left, then
    rolls everything back."""
    first, moved = _migration("jobs_20261111"), _migration("jobs_schema_20261112")
    with engine.connect() as conn:
        tx = conn.begin()
        try:
            conn.execute(text("SET LOCAL lock_timeout = '5s'"))
            setup(conn, first)
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS jobs"))
            conn.execute(text(moved._MOVE))
            return conn.execute(text(
                "SELECT to_regclass('public.jobs') IS NULL, to_regclass('jobs.jobs') IS NOT NULL, "
                "(SELECT count(*) FROM jobs.jobs WHERE kind = 'probe.moved'), "
                "(SELECT count(*) FROM pg_indexes WHERE schemaname = 'jobs' AND tablename = 'jobs' "
                " AND indexname IN ('uq_jobs_kind_workspace_key', 'ix_jobs_claim', 'ix_jobs_workspace'))")).first()
        finally:
            tx.rollback()


def _public_table_with_a_row(conn, first):
    conn.execute(text("DROP TABLE IF EXISTS public.jobs"))
    conn.execute(text("SET LOCAL search_path = public"))
    conn.execute(text(first._TABLE))
    for statement in first._INDEXES:
        conn.execute(text(statement))
    conn.execute(text("INSERT INTO public.jobs (id, kind, max_attempts) VALUES (gen_random_uuid(), 'probe.moved', 1)"))


def test_the_migration_moves_the_table_with_its_rows_and_indexes(engine):
    def only_public(conn, first):
        conn.execute(text("DROP TABLE jobs.jobs"))
        _public_table_with_a_row(conn, first)

    assert tuple(_rehearse(engine, only_public)) == (True, True, 1, 3)


def test_after_create_all_the_stray_public_table_is_folded_in_and_dropped(engine):
    assert tuple(_rehearse(engine, _public_table_with_a_row)) == (True, True, 1, 3)


def test_a_claim_takes_each_workspaces_next_job_before_any_workspaces_second(new_session, workspaces):
    busy, quiet = workspaces(), workspaces()
    older = datetime.now(timezone.utc) - timedelta(minutes=5)
    for n in range(3):
        _do(new_session, queue.enqueue, KIND, {"n": n}, workspace_id=busy, run_at=older + timedelta(seconds=n))
    _do(new_session, queue.enqueue, KIND, {"n": "quiet"}, workspace_id=quiet)
    claimed = _do(new_session, queue.claim, kind_of(KIND), "w-test", 2)  # both workspaces may run two
    assert sorted(str(j.workspace_id) for j in claimed) == sorted([busy, quiet])


def test_the_metrics_read_counts_and_the_oldest_due_wait(new_session, workspaces):
    ws = workspaces()
    _do(new_session, queue.enqueue, KIND, {}, workspace_id=ws, run_at=datetime.now(timezone.utc) - timedelta(seconds=90))
    _do(new_session, queue.enqueue, KIND, {}, workspace_id=ws, key="later",
        run_at=datetime.now(timezone.utc) + timedelta(hours=1))
    dead = _do(new_session, queue.enqueue, KIND, {}, workspace_id=ws, key="dead")
    _do(new_session, lambda s: s.execute(text("UPDATE jobs.jobs SET status = 'dead' WHERE id = :id"), {"id": dead}))
    counts = _do(new_session, jobs_metrics.read_counts)
    assert counts[(KIND, "queued")] == 2 and counts[(KIND, "dead")] == 1 and (KIND, "running") not in counts
    assert 85 <= _do(new_session, jobs_metrics.read_waits)[KIND] < 600  # the future job isn't due


def test_the_gauges_register_only_when_opentelemetry_is_on(monkeypatch):
    made = []
    meter = SimpleNamespace(create_observable_gauge=lambda name, **kw: made.append((name, kw)))
    monkeypatch.setattr("opentelemetry.metrics.get_meter", lambda name: meter)
    monkeypatch.setattr("core.observability.otel.tracing_enabled", lambda: False)
    assert jobs_metrics.observe_queue() is False and made == []

    monkeypatch.setattr("core.observability.otel.tracing_enabled", lambda: True)
    monkeypatch.setattr(jobs_metrics, "_observed", lambda read, to_points: to_points(
        {(KIND, "queued"): 4} if read is jobs_metrics.read_counts else {KIND: 12.5}))
    assert jobs_metrics.observe_queue() is True
    gauges = {name: kw["callbacks"][0](None) for name, kw in made}
    (count,) = gauges[jobs_metrics.COUNT]
    (wait,) = gauges[jobs_metrics.OLDEST_DUE_WAIT]
    assert (count.value, dict(count.attributes)) == (4, {"job.kind": KIND, "job.status": "queued"})
    assert (wait.value, dict(wait.attributes)) == (12.5, {"job.kind": KIND})


def test_a_failed_read_reports_nothing(monkeypatch):
    def broken(work, *args):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr("core.jobs.worker.on_own_session", broken)
    assert jobs_metrics._observed(jobs_metrics.read_counts, jobs_metrics._count_points) == []
