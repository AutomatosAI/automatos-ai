"""PRD-252 R6 — a drag on the board does what the matching button does.

Review → Done by drag skipped the ticket's approval action (a blog post that
was never published), and Review → Assigned sent the work back with no note. A
drag to In progress launched the bare brief, without the corrections and the
owner's answers the dispatch loop folds in, and kept the last run's result on
the card (F190). Now a drag that needs a decision is refused with the button's
name, and a drag to In progress is Run Now.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException

WS = UUID("00000000-0000-0000-0000-0000000000c1")
CTX = NS(workspace_id=WS, user=NS(id=1, clerk_user_id="u1", email="owner@cafe.test"))


class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def _ticket(status, **over):
    from core.models.core import BoardTask

    fields = dict(id=61, workspace_id=WS, title="Spring newsletter", status=status, assigned_agent_id=5,
                  source_type="user", review_mode="human", result="Draft 1: Spring is here.",
                  completed_at=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc), planning_data={}, runtime_ref={})
    fields.update(over)
    return BoardTask(**fields)


@pytest.fixture
def board(monkeypatch):
    """The drag route over a seeded ticket, recording what it set off."""
    from api import board_tasks as bt
    from tests.test_board_task_handlers import _FakeSession

    seen = NS(launched=[], consent=[], available=[], completed=[])
    monkeypatch.setattr(bt, "_launch_task_execution", lambda **kw: seen.launched.append(kw["task_id"]))
    monkeypatch.setattr(bt, "record_operator_consent", lambda *a, **kw: seen.consent.append(kw.get("why")))
    monkeypatch.setattr(bt, "notify_board_event", lambda *a, **k: None)
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: seen.available.append(kw["task_id"]))

    async def _completed(db, workspace_id, task):
        seen.completed.append(task.id)

    monkeypatch.setattr(bt, "_dispatch_task_complete", _completed)

    def drag(task, status):
        return asyncio.run(bt.update_task_status(task.id, _Req({"status": status}), ctx=CTX,
                                                 db=_FakeSession(agent=NS(id=5), task=task)))

    seen.drag = drag
    return seen


@pytest.mark.parametrize("to, button", [("done", "Use Approve"), ("assigned", "Use Reject")])
def test_a_drag_that_needs_a_verdict_is_refused_with_the_buttons_name(board, to, button):
    task = _ticket("review", planning_data={"approval_action": {"type": "publish_blog", "post_id": "p1"}})

    with pytest.raises(HTTPException) as refused:
        board.drag(task, to)

    assert refused.value.status_code == 409 and refused.value.detail.startswith(button)
    assert task.status == "review" and task.result == "Draft 1: Spring is here."
    assert board.completed == []                    # night: Done by drag, the post never published


def test_a_drag_to_in_progress_is_run_now(board):
    from services.board_consent import WHY_MOVED_TO_IN_PROGRESS

    task = _ticket("done", review_mode="auto")

    board.drag(task, "in_progress")

    assert board.launched == []                     # before: the bare brief, launched directly
    assert board.available == [61]                  # the dispatch loop claims it, answers and corrections in
    assert board.consent == [WHY_MOVED_TO_IN_PROGRESS]
    assert task.status == "assigned"
    assert task.result is None                      # F190: off the card...
    assert task.planning_data["previous_runs"][-1]["result"] == "Draft 1: Spring is here."   # ...and on record


def test_run_now_starts_a_clean_run_too(monkeypatch):
    """F190: Run Now on a finished ticket left its old result on the new run's card,
    and finalize kept the longer of the two."""
    from api import board_tasks as bt
    from tests.test_board_task_handlers import _FakeSession

    monkeypatch.setattr(bt, "record_operator_consent", lambda *a, **k: None)
    monkeypatch.setattr(bt, "notify_task_available", lambda *a, **k: None)
    task = _ticket("failed", result="Half the numbers", error_message="Provider timed out")

    out = asyncio.run(bt.run_task_now(task.id, ctx=CTX, db=_FakeSession(agent=NS(id=5), task=task)))

    assert out["rerun_of"] == "failed" and task.status == "assigned"
    assert (task.result, task.error_message) == (None, None)
    assert task.planning_data["previous_runs"][-1]["result"] == "Half the numbers"


def test_other_drags_still_move_the_ticket(board):
    task = _ticket("review")

    board.drag(task, "blocked")

    assert task.status == "blocked" and task.blocked_at is not None
