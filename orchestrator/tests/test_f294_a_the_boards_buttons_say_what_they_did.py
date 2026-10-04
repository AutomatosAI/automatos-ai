"""F294 (night 8) — small board gaps: what the board's own buttons do and say.

- Cancel on a Done card answered 200 ``{"status": "done", "applied": false}`` with no
  words, and a drag to Cancelled moved the same card with no note (#0422).
- Approve with a note on a card that finished by itself answered the raw "Task must
  be in review status (currently: done)" (#0345).
- The board's Assign left no note on the card (#0313, #0332).
- A cancelled card with no answer was let into Review (#0250.1).
- Run now after a send-back whose redo failed dropped the owner's reason (#0273).
On the real schema where a row is written.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from services.board_drag_rules import drag_refusal
from tests import test_prd252_drags_match_buttons as r6

KILOS = "0 kg would leave the cafés short on Thursday 15 October. Put the kilos on the first line."


class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture
def board(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    agent = db_session.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES ('Content Creator', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(ws)}).scalar()
    ctx = NS(workspace_id=ws, user=NS(id="owner@cafe.test", clerk_user_id=None, email="owner@cafe.test"))
    return NS(db=db_session, ws=ws, agent=agent, ctx=ctx)


def _card(board, status, **over):
    from core.models.core import BoardTask

    fields = dict(workspace_id=board.ws, title="Reply to Owen", priority="medium", source_type="user",
                  status=status, review_mode="auto", result="To: owen@example.test …",
                  completed_at=datetime(2026, 10, 4, 6, 40, tzinfo=timezone.utc), runtime_ref={}, planning_data={})
    fields.update(over)
    task = BoardTask(**fields)
    board.db.add(task)
    board.db.flush()
    return task


def _notes(board, task):
    board.db.expire_all()
    from core.models.core import BoardTask

    row = board.db.get(BoardTask, task.id)
    return row, [n["note"] for n in (row.runtime_ref or {}).get("session_notes") or []]


def test_cancel_on_a_done_card_calls_it_off_and_says_so(board):
    from api.board_tasks import cancel_task

    card = _card(board, "done")

    answer = asyncio.run(cancel_task(card.id, ctx=board.ctx, db=board.db))

    row, notes = _notes(board, card)
    assert answer["applied"] is True and "is cancelled now" in answer["message"]   # night 8: applied false, no words
    assert row.status == "cancelled" and row.result == "To: owen@example.test …"   # its answer is kept
    assert row.runtime_ref["cancelled"]["by"] == "user:owner@cafe.test" and notes == ["Cancelled this."]


def test_cancel_on_a_card_already_cancelled_says_there_is_nothing_to_cancel(board):
    from api.board_tasks import cancel_task

    card = _card(board, "cancelled")

    answer = asyncio.run(cancel_task(card.id, ctx=board.ctx, db=board.db))

    assert answer["applied"] is False and "already cancelled: there is nothing to cancel" in answer["message"]


def test_a_drag_of_a_done_card_to_cancelled_leaves_the_same_note(board):
    from api.board_tasks import update_task_status

    card = _card(board, "done")

    asyncio.run(update_task_status(card.id, _Req({"status": "cancelled"}), ctx=board.ctx, db=board.db))

    row, notes = _notes(board, card)
    assert row.status == "cancelled" and notes == ["Cancelled this."]             # #0422: no note


def test_approve_with_a_note_on_a_card_that_finished_by_itself_keeps_the_note(board):
    from api.board_tasks import approve_task
    from services.ticket_redo import agent_lessons

    card = _card(board, "done", assigned_agent_id=board.agent)
    note = "Right: To: first. Next time no exclamation mark at the end."

    answer = asyncio.run(approve_task(card.id, _Req({"note": note}), ctx=board.ctx, db=board.db))

    row, notes = _notes(board, card)
    assert answer["applied"] is True and row.status == "done"      # night 8: "Task must be in review status"
    assert notes == [f"Approved: {note}"]
    assert any("exclamation mark" in lesson for lesson in agent_lessons(board.db, board.ws, board.agent))


def test_approve_on_a_card_nobody_worked_on_says_why_in_words(board):
    from api.board_tasks import approve_task

    card = _card(board, "inbox", result=None, completed_at=None)

    with pytest.raises(HTTPException) as refused:
        asyncio.run(approve_task(card.id, _Req({}), ctx=board.ctx, db=board.db))

    assert refused.value.status_code == 422
    assert "is in the Inbox: no one has worked on it yet" in refused.value.detail
    assert "Task must be in review status" not in refused.value.detail


def test_the_boards_assign_says_on_the_card_who_it_went_to(board, monkeypatch):
    import api.board_tasks as bt

    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: None)
    monkeypatch.setattr(bt, "consent_for_created_ticket", lambda *a, **kw: None)
    card = _card(board, "inbox", result=None, completed_at=None)

    asyncio.run(bt.update_task(card.id, _Req({"assigned_agent_id": board.agent}), ctx=board.ctx, db=board.db))

    _row, notes = _notes(board, card)
    assert notes == ["Gave this to Content Creator."]                          # #0313, #0332: nothing


def test_a_cancelled_card_with_no_answer_is_not_let_into_review():
    step = r6._ticket("cancelled", source_type="orchestration_task", result=None, workspace_seq=250)

    refusal = drag_refusal(step, "review", running=False, mission_ticket=True)

    assert refusal is not None and "was cancelled with no answer on it, so there is nothing to review" in refusal


def test_run_now_after_a_failed_redo_carries_the_owners_reason_again():
    from services.ticket_redo import SENT_BACK, redo_again, redo_block

    task = NS(id=273, status="failed", result=None, review_feedback=None, assigned_agent_id=None, workspace_id=None,
              planning_data={
                  "owner_corrections": [{"note": KILOS, "at": "2026-10-04T01:59:00+00:00"}],
                  "previous_runs": [{"status": "review", "result": "0 kg", "why": SENT_BACK,
                                     "at": "2026-10-04T01:59:00+00:00"}]})

    assert redo_again(task) == KILOS                                     # night 8: the reason was gone
    assert KILOS in redo_block(task) and "0 kg" in redo_block(task)       # the draft it corrects comes too


def test_run_now_after_a_second_failure_still_carries_it():
    from services.ticket_redo import RUN_NOW, SENT_BACK, redo_again

    task = NS(status="failed", result=None, review_feedback=None, planning_data={
        "owner_corrections": [{"note": KILOS, "at": "2026-10-04T01:59:00+00:00"}],
        "previous_runs": [{"status": "review", "result": "0 kg", "why": SENT_BACK},
                          {"status": "failed", "result": "", "why": RUN_NOW}]})

    assert redo_again(task) == KILOS


def test_run_now_on_a_plain_failure_changes_nothing():
    from services.ticket_redo import redo_again

    task = NS(status="failed", result=None, review_feedback=None, planning_data={"previous_runs": []})

    assert redo_again(task) is None and task.review_feedback is None
