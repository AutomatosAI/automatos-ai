"""F190 (a) — a ticket moved to in progress by the general PATCH starts clean.

PATCH /api/v1/tasks/{id}/status puts the last run on record (keep_previous_run)
and clears its error, result and finish time when a ticket goes in progress. The
general PATCH only set started_at, so a re-run's card kept the old run's error
("Stalled: …") and result beside the new work. It now does what PATCH /status
does. (F190's other half, a send-back kept from review, came with F198.)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import UUID

WS = UUID("00000000-0000-0000-0000-0000000000c1")
CTX = NS(workspace_id=WS, user=NS(id=1, clerk_user_id="u1", email="owner@cafe.test"))


class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def _patch(monkeypatch, task, body):
    from api import board_tasks as bt
    from tests.test_board_task_handlers import _FakeSession

    launched = []
    monkeypatch.setattr(bt, "_launch_task_execution", lambda **kw: launched.append(kw["task_id"]))
    monkeypatch.setattr(bt, "notify_board_event", lambda *a, **k: None)
    asyncio.run(bt.update_task(task.id, _Req(body), ctx=CTX, db=_FakeSession(agent=NS(id=5), task=task)))
    return launched


def _failed_ticket():
    from core.models.core import BoardTask

    return BoardTask(id=41, workspace_id=WS, title="Weekly numbers", status="failed", assigned_agent_id=5,
                     source_type="user", review_mode="auto", result="Revenue: £4,210 (partial)",
                     error_message="Stalled: in_progress for >300s with no active execution",
                     completed_at=datetime(2026, 9, 26, 5, 0, tzinfo=timezone.utc), planning_data={})


def test_a_rerun_moved_to_in_progress_starts_clean_and_keeps_the_last_run(monkeypatch):
    task = _failed_ticket()

    launched = _patch(monkeypatch, task, {"status": "in_progress"})

    assert (task.status, task.error_message, task.result, task.completed_at) == ("in_progress", None, None, None)
    kept = task.planning_data["previous_runs"][-1]                              # night: the error stayed on the card
    assert (kept["status"], kept["result"], kept["why"]) == ("failed", "Revenue: £4,210 (partial)",
                                                            "moved to in progress")
    assert launched == [41]


def test_a_patch_that_sets_its_own_result_keeps_it(monkeypatch):
    task = _failed_ticket()

    _patch(monkeypatch, task, {"status": "in_progress", "result": "Started over from Monday's figures."})

    assert task.result == "Started over from Monday's figures." and task.error_message is None


# ── Review MEDIUM: a repeat of in_progress on a running ticket changes nothing ──

def _drag(monkeypatch, task, status):
    """PATCH /tasks/{id}/status, the board's drag-and-drop."""
    from api import board_tasks as bt
    from tests.test_board_task_handlers import _FakeSession

    launched = []
    monkeypatch.setattr(bt, "_launch_task_execution", lambda **kw: launched.append(kw["task_id"]))
    monkeypatch.setattr(bt, "record_operator_consent", lambda *a, **k: None)
    monkeypatch.setattr(bt, "notify_board_event", lambda *a, **k: None)
    asyncio.run(bt.update_task_status(task.id, _Req({"status": status}), ctx=CTX,
                                      db=_FakeSession(agent=NS(id=5), task=task)))
    return launched


def _running_ticket():
    from core.models.core import BoardTask

    return BoardTask(id=42, workspace_id=WS, title="Weekly numbers", status="in_progress", assigned_agent_id=5,
                     source_type="user", review_mode="auto", result="Revenue so far: £1,200",
                     started_at=datetime(2026, 9, 26, 6, 0, tzinfo=timezone.utc), planning_data={})


def test_a_second_drag_to_in_progress_leaves_the_running_ticket_alone(monkeypatch):
    task = _running_ticket()

    launched = _drag(monkeypatch, task, "in_progress")

    assert launched == []                                   # before: the agent ran a second time
    assert task.result == "Revenue so far: £1,200"          # before: the live run's result wiped
    assert task.started_at == datetime(2026, 9, 26, 6, 0, tzinfo=timezone.utc)
    assert "previous_runs" not in (task.planning_data or {})


def test_a_repeat_patch_to_in_progress_launches_nothing(monkeypatch):
    task = _running_ticket()

    assert _patch(monkeypatch, task, {"status": "in_progress"}) == []   # before: a second run


def test_a_drag_into_in_progress_still_starts_a_clean_run(monkeypatch):
    task = _failed_ticket()
    task.status = "assigned"

    assert _drag(monkeypatch, task, "in_progress") == [41]
    assert (task.result, task.error_message) == (None, None)
