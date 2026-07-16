"""PRD-204 S4 — closing the silent-failure holes.

Mock-DB suite (test_prd128_notification_dispatcher.py pattern): each new
event lands exactly one ``notifications`` row with the right ``event_type``
and ``link_id``; the board mapping stops lying about failed missions; the
scheduler benches loudly (once per breaker-open period); the dead
``notify_budget_exceeded`` stays deleted.
"""
from __future__ import annotations

import asyncio
import os

# The imported modules pull core.database.database -> config.py, which needs
# Postgres env vars at import time. Harmless defaults; no real DB is touched.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")

from unittest.mock import MagicMock  # noqa: E402
from uuid import uuid4  # noqa: E402

import pytest  # noqa: E402

from core.services.notification_dispatcher import VALID_EVENT_TYPES  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _make_db() -> MagicMock:
    """Mock session: no workspace/user rows, no notification preferences."""
    db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.first.return_value = None
    db.query.return_value = q
    result = MagicMock()
    result.fetchall.return_value = []
    db.execute.return_value = result
    return db


def _insert_params(db: MagicMock) -> list[dict]:
    """All params dicts from INSERT INTO notifications calls."""
    found = []
    for call in db.execute.call_args_list:
        if not call.args:
            continue
        sql_obj = call.args[0]
        if "INSERT INTO notifications" in str(sql_obj):
            found.append(call.args[1])
    return found


def _mock_run(**overrides):
    run = MagicMock()
    run.id = overrides.get("id", uuid4())
    run.workspace_id = overrides.get("workspace_id", uuid4())
    run.created_by = overrides.get("created_by", "user_abc")
    run.goal = overrides.get("goal", "Compile the weekly market report")
    run.stop_reason = overrides.get("stop_reason", "coordinator_error")
    run.stop_detail = overrides.get(
        "stop_detail", "Task 'research' failed: max_retries_exhausted"
    )
    run.budget_spent = overrides.get("budget_spent", {"cost": 1.25, "tokens": 5000})
    run.budget_config = overrides.get("budget_config", {"max_cost": 1.0})
    run.tokens_used = overrides.get("tokens_used", 5000)
    run.token_budget_estimate = overrides.get("token_budget_estimate", 4000)
    return run


# ---------------------------------------------------------------------------
# 1. Event vocabulary
# ---------------------------------------------------------------------------


def test_valid_event_types_include_prd204_events():
    for event in (
        "mission_failed",
        "mission_budget_paused",
        "playbook_benched",
        "watch_verdict",
        "watch_action",
        "watch_escalation",
    ):
        assert event in VALID_EVENT_TYPES, f"{event} missing from VALID_EVENT_TYPES"


# ---------------------------------------------------------------------------
# 2. mission_failed lands one notifications row
# ---------------------------------------------------------------------------


def test_notify_mission_failed_inserts_row():
    from services.coordinator_service import notify_mission_failed

    db = _make_db()
    run = _mock_run()
    _run(notify_mission_failed(db, run))

    inserts = _insert_params(db)
    assert len(inserts) == 1, "exactly one notifications row expected"
    row = inserts[0]
    assert row["event_type"] == "mission_failed"
    assert row["link_type"] == "mission"
    assert row["link_id"] == str(run.id)
    assert row["status"] == "error"
    assert "max_retries_exhausted" in row["message"]
    assert row["title"].startswith("Mission failed:")


def test_notify_mission_failed_never_raises(monkeypatch):
    """The seam is fail-soft: a broken dispatcher must not break the caller."""
    from services import coordinator_service

    async def _boom(*args, **kwargs):
        raise RuntimeError("dispatcher down")

    monkeypatch.setattr(
        "core.services.notification_dispatcher.NotificationDispatcher.dispatch",
        _boom,
    )
    # Must not raise.
    _run(coordinator_service.notify_mission_failed(_make_db(), _mock_run()))


# ---------------------------------------------------------------------------
# 3. mission_budget_paused lands one notifications row
# ---------------------------------------------------------------------------


def test_notify_mission_budget_paused_inserts_row():
    from services.coordinator_service import notify_mission_budget_paused

    db = _make_db()
    run = _mock_run()
    _run(notify_mission_budget_paused(db, run))

    inserts = _insert_params(db)
    assert len(inserts) == 1
    row = inserts[0]
    assert row["event_type"] == "mission_budget_paused"
    assert row["link_type"] == "mission"
    assert row["link_id"] == str(run.id)
    assert row["status"] == "warning"
    assert "$1.25" in row["message"]  # spent surfaced
    assert "resume" in row["message"].lower()


# ---------------------------------------------------------------------------
# 4. Board mapping stops lying (OS-review F023)
# ---------------------------------------------------------------------------


def test_failed_mission_board_card_shows_failed():
    from core.models.orchestration_enums import RunState
    from services.orchestration_board_bridge import _RUN_STATE_TO_BOARD_STATUS

    assert _RUN_STATE_TO_BOARD_STATUS[RunState.FAILED.value] == "failed"
    # cancelled stays done — the user deliberately closed it.
    assert _RUN_STATE_TO_BOARD_STATUS[RunState.CANCELLED.value] == "done"
    assert _RUN_STATE_TO_BOARD_STATUS[RunState.COMPLETED.value] == "done"


def test_board_failed_status_is_valid_board_status():
    """Drift guard: the bridge writes only statuses the board accepts."""
    import api.board_tasks as bt
    from services.orchestration_board_bridge import _RUN_STATE_TO_BOARD_STATUS

    for status in _RUN_STATE_TO_BOARD_STATUS.values():
        assert status in bt.VALID_STATUSES, f"bridge writes invalid status {status}"


# ---------------------------------------------------------------------------
# 5. playbook_benched — once per breaker-open period
# ---------------------------------------------------------------------------


@pytest.fixture
def scheduler_service():
    from services.playbook_scheduler import PlaybookSchedulerService

    return PlaybookSchedulerService()


def test_benched_notifies_once_per_open_period(
    scheduler_service, mock_playbook, monkeypatch
):
    """Two skips on one open period -> one notification; breaker closes and
    reopens -> a second notification."""
    db = _make_db()
    mock_playbook.steps = [{"step_id": "s1", "order": 1}]

    # Scheduler-local session factory -> our mock.
    monkeypatch.setattr("core.database.database.SessionLocal", lambda: db)

    # Playbook lookup returns the fixture playbook.
    q = MagicMock()
    q.filter.return_value = q
    q.first.return_value = mock_playbook
    db.query.return_value = q

    breaker_states = iter([True, True, False, True])
    monkeypatch.setattr(
        "services.playbook_breaker.breaker_is_open",
        lambda _db, _rid: next(breaker_states),
    )

    notified = []

    async def _record(_db, playbook, threshold):
        notified.append(playbook.id)

    monkeypatch.setattr(scheduler_service, "_notify_playbook_benched", _record)

    # Breaker-closed pass-through needs the launch seams stubbed.
    class _Allowed:
        allowed = True
        reason = None

    async def _check_concurrency(_ws, _db):
        return _Allowed()

    monkeypatch.setattr(
        "services.concurrency_guard.check_concurrency", _check_concurrency
    )
    engine = MagicMock()
    monkeypatch.setattr(
        "services.playbook_engine.get_playbook_engine", lambda: engine
    )

    ws = str(mock_playbook.workspace_id)

    # Open period 1: two skipped fires, ONE notification.
    _run(scheduler_service._fire_playbook(mock_playbook.id, ws))
    _run(scheduler_service._fire_playbook(mock_playbook.id, ws))
    assert notified == [mock_playbook.id]

    # Breaker closes: fire passes through and the latch clears.
    _run(scheduler_service._fire_playbook(mock_playbook.id, ws))
    assert engine.launch.called
    assert mock_playbook.id not in scheduler_service._benched_notified

    # Open period 2: notify again (exactly once more).
    _run(scheduler_service._fire_playbook(mock_playbook.id, ws))
    assert notified == [mock_playbook.id, mock_playbook.id]


def test_notify_playbook_benched_inserts_row(scheduler_service, mock_playbook):
    db = _make_db()
    _run(scheduler_service._notify_playbook_benched(db, mock_playbook, threshold=3))

    inserts = _insert_params(db)
    assert len(inserts) == 1
    row = inserts[0]
    assert row["event_type"] == "playbook_benched"
    assert row["link_type"] == "playbook"
    assert row["link_id"] == str(mock_playbook.id)
    assert row["status"] == "warning"
    assert "last 3 runs" in row["message"]


# ---------------------------------------------------------------------------
# 6. Reconciler notify seam is fail-soft
# ---------------------------------------------------------------------------


def test_reconciler_notify_run_failed_never_raises(monkeypatch):
    from modules.coordination.reconciler import MissionReconciler

    async def _boom(*args, **kwargs):
        raise RuntimeError("notify path down")

    monkeypatch.setattr(
        "services.coordinator_service.notify_mission_failed", _boom
    )
    # Must not raise into reconciliation.
    _run(MissionReconciler._notify_run_failed(_make_db(), _mock_run()))


# ---------------------------------------------------------------------------
# 7. Dead code stays dead
# ---------------------------------------------------------------------------


def test_notify_budget_exceeded_is_deleted():
    """PRD-204 S4 chose deletion over wiring: exactly one owner
    (notify_mission_budget_paused) for the budget-pause boundary."""
    from services import escalation_service

    assert not hasattr(escalation_service, "notify_budget_exceeded")
    # The live escalation helpers survive.
    assert hasattr(escalation_service, "escalate_stalled_task")
    assert hasattr(escalation_service, "check_blocked_escalations")
