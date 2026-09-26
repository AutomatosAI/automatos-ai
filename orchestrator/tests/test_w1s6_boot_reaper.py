"""PRD-142 Wave 1 · WS-C · W1-S6 — boot reaper for orphaned runs.

When the orchestrator restarts, any in-flight row whose background ``asyncio``
executor died with the old process is stranded forever: a board task stuck in
``in_progress``, a wizard profile stuck in ``scraping``/``scanning``, a workflow
execution stuck in ``running``/``pending``. Nothing remains to move it to a
terminal state.

``reap_orphaned_runs`` runs once per deploy (under the boot leader lock) and
sweeps those three surfaces. A row is reaped only if it has been in-flight
longer than ``BOOT_REAPER_STALE_MINUTES`` — long enough that no legitimately
running job (the wizard scrape, ~10–20 min, is the slowest) could still own it.
Each surface is marked terminal using its OWN failure convention:

  - board task  → ``failed`` with the reason, through the one completion
    writer (``finalize_board_task_run``, F175's rule; faked here);
  - wizard profile → ``status="failed"`` + ``quality_findings``;
  - workflow execution → ``status="failed"`` + ``error_message`` + ``completed_at``.

Each reap fires ``record_error(subsystem=<surface>, operation="boot_reap")`` so
the sweep surfaces on the ERRORS-by-subsystem dashboard tile (the WS-A sink).

These tests prove the contract with no real DB: a fake Session returns seeded
rows and staleness is filtered in Python (the row volume is tiny pre-launch).
"""
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# Importing the reaper pulls in core.utils.exception_telemetry → SessionLocal →
# the SQLAlchemy engine, which refuses to build without POSTGRES_* creds. These
# tests never touch a real DB; setdefault means a real .env still wins.
for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

import core.boot.reaper as reaper  # noqa: E402
from core.models.business_profiles import BusinessProfile  # noqa: E402
from core.models.core import BoardTask, WorkflowExecution  # noqa: E402

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
OLD = NOW - timedelta(minutes=90)          # well past the 30-min default
FRESH = NOW - timedelta(minutes=1)         # comfortably inside the window
OLD_NAIVE = (NOW - timedelta(minutes=90)).replace(tzinfo=None)  # workflow tz-naive


# ---------------------------------------------------------------------------
# Fake DB — query(Model) returns seeded rows; staleness is filtered in Python.
# ---------------------------------------------------------------------------

class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):  # SQLAlchemy expressions are ignored
        return self

    def all(self):
        return list(self._rows)

    def scalar(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    def __init__(self, rows_by_model=None, raise_for=None):
        self._rows_by_model = rows_by_model or {}
        self._raise_for = raise_for or set()
        self.query_calls = 0
        self.commits = 0
        self.rollbacks = 0

    def query(self, model):
        self.query_calls += 1
        if model is BoardTask.status:  # the reaper's re-read of a ticket it could not finish closing
            return _FakeQuery([r.status for r in self._rows_by_model.get(BoardTask, [])])
        if model in self._raise_for:
            raise RuntimeError(f"query({model.__name__}) blew up")
        return _FakeQuery(self._rows_by_model.get(model, []))

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _board(id_, status, started_at, updated_at=None):
    return SimpleNamespace(
        id=id_, status=status, started_at=started_at,
        updated_at=updated_at, completed_at=None, error_message=None,
        workspace_id=None, assigned_agent_id=None,
    )


def _completion_writer(monkeypatch, *rows):
    """finalize_board_task_run, faked: closes the seeded row as an error result does."""
    import api.board_tasks as board_tasks

    by_id, calls = {r.id: r for r in rows}, []

    async def _finalize(db, *, task_id, workspace_id, agent_id, exec_result, **_):
        calls.append((task_id, exec_result))
        row = by_id[task_id]
        row.status, row.error_message, row.completed_at = "failed", exec_result["error"], NOW
        return "failed"

    monkeypatch.setattr(board_tasks, "finalize_board_task_run", _finalize)
    return calls


def _reap(db):
    return asyncio.run(reaper.reap_orphaned_runs(db, now=NOW))


def _profile(id_, status, updated_at):
    return SimpleNamespace(
        id=id_, status=status, updated_at=updated_at,
        quality_findings=None, workspace_id=None,
    )


def _wfe(id_, status, started_at):
    return SimpleNamespace(
        id=id_, status=status, started_at=started_at,
        completed_at=None, error_message=None, workspace_id=None,
    )


# ---------------------------------------------------------------------------
# _is_stale — naive/aware coercion + the None guard
# ---------------------------------------------------------------------------

def test_is_stale_aware_old_is_stale():
    cutoff = NOW - timedelta(minutes=30)
    assert reaper._is_stale(OLD, cutoff) is True


def test_is_stale_naive_old_is_stale():
    """A tz-naive timestamp (WorkflowExecution.started_at) is assumed UTC."""
    cutoff = NOW - timedelta(minutes=30)
    assert reaper._is_stale(OLD_NAIVE, cutoff) is True


def test_is_stale_recent_is_not_stale():
    cutoff = NOW - timedelta(minutes=30)
    assert reaper._is_stale(FRESH, cutoff) is False


def test_is_stale_none_is_not_stale():
    """No timestamp → cannot prove staleness → never reaped."""
    cutoff = NOW - timedelta(minutes=30)
    assert reaper._is_stale(None, cutoff) is False


# ---------------------------------------------------------------------------
# Per-surface terminal conventions
# ---------------------------------------------------------------------------

def test_reaps_stale_board_task_as_failed_through_the_completion_writer(monkeypatch):
    rec = MagicMock()
    monkeypatch.setattr(reaper, "record_error", rec)
    row = _board(7, "in_progress", OLD)
    closed = _completion_writer(monkeypatch, row)
    db = _FakeSession({BoardTask: [row]})

    n = _reap(db)

    assert n == 1
    assert [(task_id, result["status"]) for task_id, result in closed] == [(7, "error")]
    assert row.status == "failed" and row.completed_at is not None
    assert "orphan" in (row.error_message or "").lower()
    rec.assert_called_once()
    kw = rec.call_args.kwargs
    assert kw["subsystem"] == "board"
    assert kw["operation"] == "boot_reap"


def test_reaps_stale_wizard_profile_as_failed(monkeypatch):
    rec = MagicMock()
    monkeypatch.setattr(reaper, "record_error", rec)
    row = _profile("p1", "scraping", OLD)
    db = _FakeSession({BusinessProfile: [row]})

    n = _reap(db)

    assert n == 1
    assert row.status == "failed"
    assert row.quality_findings  # an error note was recorded
    kw = rec.call_args.kwargs
    assert kw["subsystem"] == "wizard"
    assert kw["operation"] == "boot_reap"


def test_reaps_stale_workflow_execution_as_failed(monkeypatch):
    rec = MagicMock()
    monkeypatch.setattr(reaper, "record_error", rec)
    row = _wfe(42, "running", OLD_NAIVE)
    db = _FakeSession({WorkflowExecution: [row]})

    n = _reap(db)

    assert n == 1
    assert row.status == "failed"
    assert row.completed_at is not None
    assert "orphan" in (row.error_message or "").lower()
    kw = rec.call_args.kwargs
    assert kw["subsystem"] == "workflow"
    assert kw["operation"] == "boot_reap"


# ---------------------------------------------------------------------------
# Staleness gate — fresh rows survive
# ---------------------------------------------------------------------------

def test_fresh_rows_are_not_reaped(monkeypatch):
    rec = MagicMock()
    monkeypatch.setattr(reaper, "record_error", rec)
    fresh = _board(1, "in_progress", FRESH)
    closed = _completion_writer(monkeypatch, fresh)
    db = _FakeSession({
        BoardTask: [fresh],
        BusinessProfile: [_profile("p", "scraping", FRESH)],
        WorkflowExecution: [_wfe(2, "running", FRESH.replace(tzinfo=None))],
    })

    n = _reap(db)

    assert n == 0 and closed == []
    rec.assert_not_called()


# ---------------------------------------------------------------------------
# Aggregate behaviour
# ---------------------------------------------------------------------------

def test_reap_returns_total_count_across_surfaces(monkeypatch):
    monkeypatch.setattr(reaper, "record_error", MagicMock())
    stale = _board(1, "in_progress", OLD)
    _completion_writer(monkeypatch, stale)
    db = _FakeSession({
        BoardTask: [stale],
        BusinessProfile: [_profile("p", "scraping", OLD)],
        WorkflowExecution: [_wfe(2, "running", OLD_NAIVE)],
    })

    assert _reap(db) == 3
    assert db.commits >= 1


def test_disabled_flag_short_circuits(monkeypatch):
    monkeypatch.setattr(reaper.config, "BOOT_REAPER_ENABLED", False)
    rec = MagicMock()
    monkeypatch.setattr(reaper, "record_error", rec)
    db = _FakeSession({BoardTask: [_board(1, "in_progress", OLD)]})

    assert _reap(db) == 0
    assert db.query_calls == 0          # short-circuit before any query
    rec.assert_not_called()


def test_one_surface_failure_does_not_abort_others(monkeypatch):
    """A surface that blows up is recorded but the others still get reaped."""
    monkeypatch.setattr(reaper, "record_error", MagicMock())
    db = _FakeSession(
        rows_by_model={
            BusinessProfile: [_profile("p", "scraping", OLD)],
            WorkflowExecution: [_wfe(2, "running", OLD_NAIVE)],
        },
        raise_for={BoardTask},
    )

    # board raises, wizard + workflow still reaped → 2
    assert _reap(db) == 2


# ---------------------------------------------------------------------------
# Review MEDIUMs on adc84365e: a failed close is contained, a slow notice is cut
# ---------------------------------------------------------------------------

def test_a_failed_surface_is_rolled_back_before_the_next(monkeypatch):
    monkeypatch.setattr(reaper, "record_error", MagicMock())
    db = _FakeSession(rows_by_model={WorkflowExecution: [_wfe(2, "running", OLD_NAIVE)]}, raise_for={BoardTask})

    assert _reap(db) == 1
    assert db.rollbacks >= 1          # before: the poisoned transaction was carried into the next surface


def test_one_orphan_that_cannot_be_closed_does_not_stop_the_others(monkeypatch):
    import api.board_tasks as board_tasks

    monkeypatch.setattr(reaper, "record_error", MagicMock())
    stuck, orphan = _board(1, "in_progress", OLD), _board(2, "in_progress", OLD)

    async def _finalize(db, *, task_id, exec_result, **_):
        if task_id == 1:
            raise RuntimeError("commit failed")
        orphan.status = "failed"
        return "failed"

    monkeypatch.setattr(board_tasks, "finalize_board_task_run", _finalize)
    db = _FakeSession({BoardTask: [stuck, orphan]})

    assert _reap(db) == 1 and orphan.status == "failed"    # before: 0, and #2 left in progress
    assert db.rollbacks >= 1


def test_a_slow_notice_is_cut_short_not_waited_for(monkeypatch):
    import api.board_tasks as board_tasks

    monkeypatch.setattr(reaper, "record_error", MagicMock())
    monkeypatch.setattr(reaper, "ORPHAN_CLOSE_SECONDS", 0.05, raising=False)
    row = _board(7, "in_progress", OLD)

    async def _finalize(db, *, task_id, exec_result, **_):
        row.status = "failed"            # the status is committed first
        await asyncio.sleep(30)          # then a notice to a channel that does not answer
        return "failed"

    monkeypatch.setattr(board_tasks, "finalize_board_task_run", _finalize)
    started = time.monotonic()

    assert _reap(_FakeSession({BoardTask: [row]})) == 1 and row.status == "failed"
    assert time.monotonic() - started < 5                  # before: boot waited the full 30 s
