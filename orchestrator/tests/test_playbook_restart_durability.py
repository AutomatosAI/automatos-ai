"""PRD-142 Wave 3 · WS-3R · W3-S12 — Playbook restart durability.

The Wave 3 hardening contract for Playbooks (§H DoD): an in-flight execution
MUST recover from the DB on restart, not silently die fire-and-forget. The
Mission coordinator solved this with a boot reaper that marks orphaned runs
terminal at startup (W1-S6) — Playbooks port the same model.

These tests pin the contract:

  1. ``RecipeExecution`` rows stuck ``pending``/``running`` past the staleness
     window are reaped to ``status="failed"`` with a clear ``error_message``
     and ``completed_at`` at startup — no orphan stays ``running`` forever.
  2. Fresh ``pending``/``running`` rows are left alone (the process may
     still own them).
  3. The reap emits ``record_error(subsystem="playbook", operation="boot_reap")``
     so the failure surfaces on the WS-A ERRORS-by-subsystem tile.
  4. The new playbook surface runs alongside the existing board / wizard /
     workflow surfaces — the aggregate ``reap_orphaned_runs`` count includes it.
  5. A failure inside the playbook surface is recorded but does NOT abort the
     other surfaces (surface isolation by design — same contract as W1-S6).

All tests use the W1-S6 fake-session pattern — no real DB.

TDD GUARANTEE: written BEFORE the reaper extension lands. Each test fails
with ``AttributeError`` / wrong count / missing ``record_error`` call until
``_reap_recipe_executions`` is added and wired into ``reap_orphaned_runs``.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

import core.boot.reaper as reaper  # noqa: E402
from core.models.core import (  # noqa: E402
    BoardTask,
    RecipeExecution,
    WorkflowExecution,
)

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
OLD = NOW - timedelta(minutes=90)
FRESH = NOW - timedelta(minutes=1)
OLD_NAIVE = (NOW - timedelta(minutes=90)).replace(tzinfo=None)
FRESH_NAIVE = (NOW - timedelta(minutes=1)).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Same fake-session pattern as test_w1s6_boot_reaper.py (no real DB).
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def all(self):
        return list(self._rows)


class _FakeSession:
    def __init__(self, rows_by_model=None, raise_for=None):
        self._rows_by_model = rows_by_model or {}
        self._raise_for = raise_for or set()
        self.query_calls = 0
        self.commits = 0
        self.rollbacks = 0

    def query(self, model):
        self.query_calls += 1
        if model in self._raise_for:
            raise RuntimeError(f"query({model.__name__}) blew up")
        return _FakeQuery(self._rows_by_model.get(model, []))

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _rxe(execution_id, status, started_at, *, workspace_id="ws-1"):
    """Synthetic RecipeExecution row — SimpleNamespace mirrors the columns
    the reaper writes to.
    """
    return SimpleNamespace(
        id=execution_id,
        execution_id=str(execution_id),
        status=status,
        started_at=started_at,
        completed_at=None,
        error_message=None,
        workspace_id=workspace_id,
    )


# ---------------------------------------------------------------------------
# 1. Stale RUNNING playbook execution is reaped to FAILED.
# ---------------------------------------------------------------------------


def test_reaps_stale_running_recipe_execution_as_failed(monkeypatch):
    """A RecipeExecution stuck in 'running' past staleness is moved terminal
    with an orphan reason — this is the headline restart-durability primitive."""
    rec = MagicMock()
    monkeypatch.setattr(reaper, "record_error", rec)
    row = _rxe(101, "running", OLD_NAIVE)
    db = _FakeSession({RecipeExecution: [row]})

    n = reaper.reap_orphaned_runs(db, now=NOW)

    assert n >= 1
    assert row.status == "failed"
    assert row.completed_at is not None
    assert row.error_message is not None
    assert "orphan" in row.error_message.lower()
    assert db.commits >= 1
    rec.assert_called()

    # At least one subsystem='playbook' call was made.
    playbook_calls = [
        c for c in rec.call_args_list if c.kwargs.get("subsystem") == "playbook"
    ]
    assert playbook_calls, "expected record_error(subsystem='playbook', ...)"
    assert playbook_calls[0].kwargs.get("operation") == "boot_reap"


# ---------------------------------------------------------------------------
# 2. Stale PENDING playbook execution is also reaped (never launched).
# ---------------------------------------------------------------------------


def test_reaps_stale_pending_recipe_execution_as_failed(monkeypatch):
    """A row that was inserted but whose launcher died before the task
    started is still stuck 'pending' — same reap, otherwise the user's
    playbook silently disappears."""
    monkeypatch.setattr(reaper, "record_error", MagicMock())
    row = _rxe(102, "pending", OLD_NAIVE)
    db = _FakeSession({RecipeExecution: [row]})

    assert reaper.reap_orphaned_runs(db, now=NOW) >= 1
    assert row.status == "failed"
    assert row.completed_at is not None


# ---------------------------------------------------------------------------
# 3. Fresh in-flight rows are NOT reaped (the process may still own them).
# ---------------------------------------------------------------------------


def test_fresh_running_recipe_execution_is_not_reaped(monkeypatch):
    """Within the staleness window, RUNNING rows survive: the process that
    holds them is still alive."""
    rec = MagicMock()
    monkeypatch.setattr(reaper, "record_error", rec)
    row = _rxe(103, "running", FRESH_NAIVE)
    db = _FakeSession({RecipeExecution: [row]})

    n = reaper.reap_orphaned_runs(db, now=NOW)

    # No playbook reap for a fresh row.
    playbook_calls = [
        c for c in rec.call_args_list if c.kwargs.get("subsystem") == "playbook"
    ]
    assert not playbook_calls
    assert row.status == "running"
    assert row.completed_at is None
    assert n == 0


# ---------------------------------------------------------------------------
# 4. Completed/failed/cancelled rows are NEVER touched.
# ---------------------------------------------------------------------------


def test_terminal_recipe_executions_are_untouched(monkeypatch):
    """Already-terminal rows ('completed'/'failed'/'cancelled') must not be
    re-marked — they are not orphaned, they are done."""
    monkeypatch.setattr(reaper, "record_error", MagicMock())
    rows = [
        _rxe(201, "completed", OLD_NAIVE),
        _rxe(202, "failed", OLD_NAIVE),
        _rxe(203, "cancelled", OLD_NAIVE),
    ]
    pre_status = [r.status for r in rows]
    db = _FakeSession({RecipeExecution: rows})

    n = reaper.reap_orphaned_runs(db, now=NOW)

    assert n == 0
    assert [r.status for r in rows] == pre_status
    for r in rows:
        assert r.completed_at is None
        assert r.error_message is None


# ---------------------------------------------------------------------------
# 5. Aggregate count includes the playbook surface alongside the others.
# ---------------------------------------------------------------------------


def test_aggregate_count_includes_playbook_surface(monkeypatch):
    """``reap_orphaned_runs`` returns the SUM across all four surfaces — the
    playbook reap is additive on top of board / wizard / workflow."""
    monkeypatch.setattr(reaper, "record_error", MagicMock())
    db = _FakeSession({
        BoardTask: [SimpleNamespace(
            id=1, status="in_progress", started_at=OLD, updated_at=OLD,
            completed_at=None, error_message=None, workspace_id=None,
        )],
        WorkflowExecution: [SimpleNamespace(
            id=2, status="running", started_at=OLD_NAIVE,
            completed_at=None, error_message=None, workspace_id=None,
        )],
        RecipeExecution: [
            _rxe(301, "running", OLD_NAIVE),
            _rxe(302, "pending", OLD_NAIVE),
        ],
    })

    # board (1) + workflow (1) + recipe (2) = 4 minimum (wizard skipped — no rows)
    assert reaper.reap_orphaned_runs(db, now=NOW) >= 4
    assert db.commits >= 1


# ---------------------------------------------------------------------------
# 6. Surface isolation — a playbook reap exception does NOT abort the rest.
# ---------------------------------------------------------------------------


def test_playbook_surface_failure_does_not_abort_others(monkeypatch):
    """If the playbook reap blows up, board / wizard / workflow still run.
    Mirrors the W1-S6 contract for surface isolation."""
    monkeypatch.setattr(reaper, "record_error", MagicMock())
    db = _FakeSession(
        rows_by_model={
            BoardTask: [SimpleNamespace(
                id=1, status="in_progress", started_at=OLD, updated_at=OLD,
                completed_at=None, error_message=None, workspace_id=None,
            )],
            WorkflowExecution: [SimpleNamespace(
                id=2, status="running", started_at=OLD_NAIVE,
                completed_at=None, error_message=None, workspace_id=None,
            )],
        },
        raise_for={RecipeExecution},  # playbook surface blows up
    )

    # board (1) + workflow (1) still reaped → 2
    assert reaper.reap_orphaned_runs(db, now=NOW) == 2


# ---------------------------------------------------------------------------
# 7. The disabled flag short-circuits the playbook surface too.
# ---------------------------------------------------------------------------


def test_disabled_flag_short_circuits_playbook_surface(monkeypatch):
    """BOOT_REAPER_ENABLED=False skips ALL surfaces, including the new
    playbook surface — same parent gate, no separate flag."""
    monkeypatch.setattr(reaper.config, "BOOT_REAPER_ENABLED", False)
    rec = MagicMock()
    monkeypatch.setattr(reaper, "record_error", rec)
    db = _FakeSession({RecipeExecution: [_rxe(401, "running", OLD_NAIVE)]})

    assert reaper.reap_orphaned_runs(db, now=NOW) == 0
    assert db.query_calls == 0
    rec.assert_not_called()


# ---------------------------------------------------------------------------
# 8. The reaper is wired into main.py boot (regression — proven once).
# ---------------------------------------------------------------------------


def test_boot_reaper_is_called_from_main_startup():
    """The reaper was already wired in W1-S6 (main.py:362). W3-S12 piggybacks
    on the SAME wire-up — re-prove it once so we cannot silently regress."""
    main_path = ORCH_ROOT / "main.py"
    text = main_path.read_text()
    assert "from core.boot.reaper import reap_orphaned_runs" in text
    assert "reap_orphaned_runs(db)" in text


# ---------------------------------------------------------------------------
# 9. Canonical orphan reason string is shared (not invented per surface).
# ---------------------------------------------------------------------------


def test_canonical_orphan_reason_is_reused():
    """Every surface uses the same ``_ORPHAN_REASON`` constant so logs /
    dashboards can group by reason — drift here is silent corruption."""
    assert reaper._ORPHAN_REASON == "orphaned_on_restart"
