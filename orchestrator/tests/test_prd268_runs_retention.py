"""PRD-268 §6.2 — ``proactive_runs``, the check-in audit, is kept 90 days.

``services/proactive/retention.py`` against Postgres, inside the suite's rolled-back
transaction (joined by SAVEPOINT, so the prune's commits stay inside it), at a clock fixed in
2001: only this file's rows are that old, so the counts are exact. Proven here:

* a run row written longer ago than the retention is deleted, one younger stays, and a row
  written exactly at the cutoff stays; the job counts what went by run kind;
* the delete goes in bounded batches, each at most the size asked for and committed before the
  next is chosen, until one comes back short; a failure part way keeps the batches before it,
  and the job returns and logs their count;
* nothing else is touched: an old suggestion (naming a pruned run), an old follow-up
  (``agent_scheduled_tasks``), an old brief claim (``mobile_digest_posts``), an old calibration
  and a person's settings all stay;
* the minimum: ``PROACTIVE_RUNS_RETENTION_DAYS`` defaults to 90 and is never under 1, as
  config parses it and where the prune reads it, so a zero never empties the audit;
* a database that fails is logged with its traceback, rolled back and closed, and never raises
  out of the job, nor does a session that cannot be opened or will not close;
* ``register`` adds one daily job at 04:45 UTC with no args, is False without a running
  scheduler, and ``start_schedule_reconcile`` registers it on the leader.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain when the suite's
# own environment does not set it (CI does, and these defaults never override it).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ORCH = Path(__file__).resolve().parents[1]
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

from services import schedule_reconcile  # noqa: E402
from services.proactive import retention  # noqa: E402
from services.proactive.runs import KIND_AGENT, KIND_FOLLOW_UP, KIND_WORKSPACE  # noqa: E402

NOW = datetime(2001, 1, 1, 12, 0, tzinfo=timezone.utc)
DAYS = 90
CUTOFF = NOW - timedelta(days=DAYS)
OLD = NOW - timedelta(days=DAYS + 1)
YOUNG = NOW - timedelta(days=DAYS - 1)
SETTING = "PROACTIVE_RUNS_RETENTION_DAYS"


# --------------------------------------------------------------------------- fixtures


class _BorrowedSession:
    """The test's session, lent to code that opens its own: ``close()`` is a no-op."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        return None


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here): the
    prune commits once per batch, and each commit reaches the savepoint only."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def ws(seed_workspace, db_session) -> UUID:
    workspace_id = UUID(seed_workspace(name="prd268 runs retention"))
    db_session.commit()
    return workspace_id


@pytest.fixture
def ninety_days(monkeypatch):
    monkeypatch.setattr(retention.config, SETTING, DAYS)


def _run(db, ws: UUID, at: datetime, kind: str = KIND_AGENT) -> UUID:
    run_id = uuid4()
    db.execute(text(
        "INSERT INTO proactive_runs (id, workspace_id, kind, outcome, created_at) "
        "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), :kind, 'silent', :at)"),
        {"id": str(run_id), "ws": str(ws), "kind": kind, "at": at})
    db.commit()
    return run_id


def _runs_left(db, ws: UUID) -> set:
    rows = db.execute(text("SELECT id FROM proactive_runs WHERE workspace_id = CAST(:ws AS uuid)"),
                      {"ws": str(ws)}).scalars().all()
    return {UUID(str(row)) for row in rows}


def _prune(db, now: datetime = NOW) -> dict:
    return retention.run_prune(session_factory=lambda: _BorrowedSession(db), now=now)


# --------------------------------------------------------------------------- what goes, what stays


def test_rows_older_than_the_retention_go_and_newer_rows_stay(db_session, ws, ninety_days):
    gone = {
        _run(db_session, ws, OLD, KIND_AGENT),
        _run(db_session, ws, OLD - timedelta(days=200), KIND_WORKSPACE),
        _run(db_session, ws, CUTOFF - timedelta(seconds=1), KIND_FOLLOW_UP),
    }
    kept = {
        _run(db_session, ws, CUTOFF, KIND_AGENT),
        _run(db_session, ws, YOUNG, KIND_AGENT),
        _run(db_session, ws, NOW, KIND_WORKSPACE),
    }

    deleted = _prune(db_session)

    assert _runs_left(db_session, ws) == kept and not gone & _runs_left(db_session, ws)
    assert deleted == {KIND_AGENT: 1, KIND_FOLLOW_UP: 1, KIND_WORKSPACE: 1}


def test_the_delete_goes_in_bounded_batches(db_session, ws):
    for _ in range(5):
        _run(db_session, ws, OLD)
    young = _run(db_session, ws, YOUNG)
    left = []

    batches = []
    for kinds in retention.prune_batches(db_session, days=DAYS, now=NOW, batch_size=2):
        batches.append(kinds)
        left.append(len(_runs_left(db_session, ws)))

    assert [len(kinds) for kinds in batches] == [2, 2, 1]
    assert {kind for batch in batches for kind in batch} == {KIND_AGENT}
    assert left == [4, 2, 1], "each batch is deleted before the next is chosen"
    assert _runs_left(db_session, ws) == {young}


def test_a_failure_part_way_keeps_and_counts_the_batches_before_it(db_session, ws, ninety_days, monkeypatch,
                                                                   caplog):
    for _ in range(5):
        _run(db_session, ws, OLD)
    young = _run(db_session, ws, YOUNG)
    real_batch, calls = retention._delete_batch, []

    def second_fails(db, before, batch_size):
        calls.append(batch_size)
        if len(calls) == 2:
            raise OperationalError("DELETE FROM proactive_runs", {}, ConnectionError("server closed the connection"))
        return real_batch(db, before, batch_size)

    monkeypatch.setattr(retention, "PRUNE_BATCH", 2)
    monkeypatch.setattr(retention, "_delete_batch", second_fails)

    with caplog.at_level(logging.ERROR, logger=retention.__name__):
        deleted = _prune(db_session)

    assert deleted == {KIND_AGENT: 2} and calls == [2, 2]
    assert len(_runs_left(db_session, ws)) == 4 and young in _runs_left(db_session, ws)
    assert "failed after 2 row(s) deleted" in caplog.text


def test_a_prune_with_nothing_old_deletes_nothing(db_session, ws, ninety_days):
    kept = {_run(db_session, ws, YOUNG), _run(db_session, ws, NOW)}

    assert _prune(db_session) == {}
    assert _runs_left(db_session, ws) == kept


# --------------------------------------------------------------------------- only proactive_runs


def _user(db) -> int:
    name = f"prd268rr_{uuid4().hex[:10]}"
    return int(db.execute(text(
        "INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
        {"u": name, "e": f"{name}@t.test", "c": name}).scalar())


def _agent(db, ws: UUID) -> int:
    return int(db.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
        "VALUES ('Auto', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(ws)}).scalar())


def _old_suggestion(db, ws: UUID, user: int, run_id: UUID) -> None:
    db.execute(text(
        "INSERT INTO proactive_suggestions (id, workspace_id, user_id, source_type, run_id, title, "
        "proposed_action, subject_key, status, created_at, last_seen_at) VALUES (CAST(:id AS uuid), "
        "CAST(:ws AS uuid), :user, 'check_in', CAST(:run AS uuid), 'Ticket #0846 failed', "
        "CAST(:action AS jsonb), 'title:ticket #0846 failed', 'hidden', :at, :at)"),
        {"id": str(uuid4()), "ws": str(ws), "user": user, "run": str(run_id),
         "action": json.dumps({"kind": "chat", "payload": {}}), "at": OLD})


def _old_follow_up(db, ws: UUID, user: int) -> None:
    auto = _agent(db, ws)
    db.execute(text(
        "INSERT INTO agent_scheduled_tasks (workspace_id, created_by_agent_id, target_agent_id, task_type, "
        "description, schedule, next_run_at, deliver_as, for_user_id, status, created_at) VALUES "
        "(CAST(:ws AS uuid), :auto, :auto, 'one_shot', 'Check the October report', :schedule, :at, "
        "'follow_up', :user, 'completed', :at)"),
        {"ws": str(ws), "auto": auto, "schedule": OLD.isoformat(), "at": OLD, "user": user})


def _old_brief_calibration_and_settings(db, ws: UUID, user: int) -> None:
    db.execute(text(
        "INSERT INTO mobile_digest_posts (workspace_id, user_id, local_date, posted_at) "
        "VALUES (CAST(:ws AS uuid), :user, :day, :at)"), {"ws": str(ws), "user": user, "day": OLD.date(), "at": OLD})
    db.execute(text(
        "INSERT INTO proactive_calibrations (id, workspace_id, user_id, status, created_at) "
        "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), :user, 'answered', :at)"),
        {"id": str(uuid4()), "ws": str(ws), "user": user, "at": OLD})
    db.execute(text(
        "INSERT INTO proactive_person_settings (workspace_id, user_id, paused, updated_at) "
        "VALUES (CAST(:ws AS uuid), :user, true, :at)"), {"ws": str(ws), "user": user, "at": OLD})


def _others(db, ws: UUID) -> dict:
    tables = ("proactive_suggestions", "agent_scheduled_tasks", "mobile_digest_posts",
              "proactive_calibrations", "proactive_person_settings")
    return {table: db.execute(text(f"SELECT COUNT(*) FROM {table} WHERE workspace_id = CAST(:ws AS uuid)"),
                              {"ws": str(ws)}).scalar() for table in tables}


def test_only_proactive_runs_is_touched(db_session, ws, ninety_days):
    user = _user(db_session)
    pruned = _run(db_session, ws, OLD)
    _old_suggestion(db_session, ws, user, pruned)
    _old_follow_up(db_session, ws, user)
    _old_brief_calibration_and_settings(db_session, ws, user)
    db_session.commit()
    before = _others(db_session, ws)

    deleted = _prune(db_session)

    assert deleted == {KIND_AGENT: 1} and _runs_left(db_session, ws) == set()
    assert before == dict.fromkeys(before, 1)
    assert _others(db_session, ws) == before


# --------------------------------------------------------------------------- the minimum


def _config_value(raw) -> str:
    env = {key: value for key, value in os.environ.items() if key != SETTING}
    if raw is not None:
        env[SETTING] = raw
    done = subprocess.run([sys.executable, "-c", f"from config import config; print(config.{SETTING})"],
                          cwd=ORCH, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("raw, parsed", [(None, "90"), ("45", "45"), ("0", "1")])
def test_config_parses_the_retention_at_least_one_day(raw, parsed):
    assert _config_value(raw) == parsed


@pytest.mark.parametrize("configured, kept", [(0, 1), (-3, 1), (1, 1), (30, 30), (DAYS, DAYS)])
def test_the_prune_reads_the_retention_never_under_one_day(monkeypatch, configured, kept):
    monkeypatch.setattr(retention.config, SETTING, configured)

    assert retention.retention_days() == kept
    assert retention.cutoff(NOW, configured) == NOW - timedelta(days=kept)


def test_a_zero_retention_never_empties_the_audit(db_session, ws, monkeypatch):
    monkeypatch.setattr(retention.config, SETTING, 0)
    two_days = _run(db_session, ws, NOW - timedelta(days=2))
    half_a_day = _run(db_session, ws, NOW - timedelta(hours=12))

    _prune(db_session)

    assert two_days not in _runs_left(db_session, ws) and _runs_left(db_session, ws) == {half_a_day}


def test_the_setting_is_declared():
    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"]
    assert SETTING in surface


# --------------------------------------------------------------------------- a failure


class _FailingSession:
    """A session whose database is gone: every statement raises."""

    def __init__(self, *, rollback_raises: bool = False):
        self.rollback_raises = rollback_raises
        self.rolled_back = self.closed = False

    def execute(self, *args, **kwargs):
        raise OperationalError("DELETE FROM proactive_runs", {}, ConnectionError("server closed the connection"))

    def commit(self):
        raise AssertionError("nothing to commit after a failed statement")

    def rollback(self):
        self.rolled_back = True
        if self.rollback_raises:
            raise OperationalError("ROLLBACK", {}, ConnectionError("server closed the connection"))

    def close(self):
        self.closed = True


def _errors(caplog) -> list:
    return [record for record in caplog.records
            if record.name == retention.__name__ and record.levelno == logging.ERROR]


def test_a_database_failure_is_logged_and_never_raises_out_of_the_job(caplog, ninety_days):
    session = _FailingSession()

    with caplog.at_level(logging.ERROR, logger=retention.__name__):
        deleted = retention.run_prune(session_factory=lambda: session, now=NOW)

    assert deleted == {}
    assert session.rolled_back and session.closed
    (record,) = _errors(caplog)
    assert record.exc_info is not None and record.exc_info[0] is OperationalError
    assert "runs prune failed" in record.getMessage()


def test_a_session_that_cannot_be_opened_is_logged_and_never_raises(caplog, ninety_days):
    def no_database():
        raise OperationalError("connect", {}, ConnectionError("connection refused"))

    with caplog.at_level(logging.ERROR, logger=retention.__name__):
        deleted = retention.run_prune(session_factory=no_database, now=NOW)

    assert deleted == {}
    (record,) = _errors(caplog)
    assert record.exc_info[0] is OperationalError


def test_a_session_that_will_not_roll_back_is_logged_and_never_raises(caplog, ninety_days):
    session = _FailingSession(rollback_raises=True)

    with caplog.at_level(logging.ERROR, logger=retention.__name__):
        deleted = retention.run_prune(session_factory=lambda: session, now=NOW)

    assert deleted == {}
    assert session.rolled_back and session.closed, "a failed rollback still closes the session"
    assert [record.exc_info[0] for record in _errors(caplog)] == [OperationalError, OperationalError]
    assert "did not roll back" in _errors(caplog)[-1].getMessage()


# --------------------------------------------------------------------------- the scheduler


class _FakeScheduler:
    """The slice of APScheduler ``register`` touches."""

    def __init__(self, running=True):
        self.running = running
        self.jobs = []

    def add_job(self, func, trigger=None, **kwargs):
        self.jobs.append({"func": func, "trigger": trigger, **kwargs})


def test_register_adds_one_daily_job_with_no_args():
    scheduler = _FakeScheduler()

    assert retention.register(scheduler) is True

    (job,) = scheduler.jobs
    assert job["func"] is retention.run_prune and job["id"] == retention.PRUNE_JOB_ID
    assert "args" not in job and "kwargs" not in job
    assert (job["replace_existing"], job["max_instances"], job["coalesce"]) == (True, 1, True)
    after = datetime(2026, 10, 31, 5, 0, tzinfo=timezone.utc)
    before = datetime(2026, 10, 31, 4, 0, tzinfo=timezone.utc)
    assert job["trigger"].get_next_fire_time(None, after) == datetime(2026, 11, 1, 4, 45, tzinfo=timezone.utc)
    assert job["trigger"].get_next_fire_time(None, before) == datetime(2026, 10, 31, 4, 45, tzinfo=timezone.utc)


def test_register_needs_a_running_scheduler():
    stopped = _FakeScheduler(running=False)

    assert retention.register(None) is False and retention.register(stopped) is False
    assert stopped.jobs == []


def test_start_schedule_reconcile_registers_the_prune_on_the_leader(monkeypatch):
    scheduler = _FakeScheduler()
    monkeypatch.setattr(schedule_reconcile, "run_reconcile_once", lambda s, db=None: {})

    assert asyncio.run(schedule_reconcile.start_schedule_reconcile(scheduler)) is True

    jobs = [job for job in scheduler.jobs if job["id"] == retention.PRUNE_JOB_ID]
    assert len(jobs) == 1 and jobs[0]["func"] is retention.run_prune
