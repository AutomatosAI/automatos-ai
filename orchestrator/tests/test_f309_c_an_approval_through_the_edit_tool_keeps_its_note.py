"""F309 (night 9): an approval with a note through platform_update_task keeps the note.

#1866, iteration 1: "Please approve card 1866 with this note: Good, this is the figure
I'll use. Ignore the June review from now on." Auto's first call was platform_update_task
{task_id: 1866, notes: "…", status: "done"}; the edit tool took no status, and the call
was refused before it ran. Auto split it into a plain note and a bare move to Done, and
the approval's note ("Approved: …") was never written. #1879 in iteration 2 kept it: one
call, platform_update_task_status {status: "done", note}.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from tests import test_1094_a_ticket_with_no_agent_is_never_in_progress as f1094

board = f1094.board        # a workspace, its Content Creator, Auto's real tools
OWNER = "owner@cafe.test"
NOTE = "Good, this is the figure I'll use. Ignore the June review from now on."
MARGIN = "Kirinyaga AA 250g retail bag margin: £8.68 (66.8%), from margin-sheet-sep-2026.csv."


@pytest.fixture
def shop(board, monkeypatch):
    """Ticket notes written in the test's session; filing a done card's report is not this test's."""
    import api.board_tasks as bt
    import core.database.database as database
    import services.report_knowledge as report_knowledge

    @contextmanager
    def this_session():
        yield board.db

    async def filed(db, workspace_id, task):
        board.filed.append(task.id)

    board.filed = []
    monkeypatch.setattr(database, "get_db_session", this_session)
    monkeypatch.setattr(bt, "notify_task_available", lambda *a, **k: None)
    monkeypatch.setattr(report_knowledge, "file_done_ticket", filed)
    return board


def _in_review(shop):
    return f1094._ticket(shop, status="review", assigned_agent_id=shop.agent, result=MARGIN,
                         completed_at=datetime.now(timezone.utc))


def _edit(shop, **params):
    """platform_update_task as Auto's call reaches it: the executor's aliases and its
    check of the params the action takes (unified_executor), then the tool."""
    from modules.tools.discovery import get_action_registry
    from modules.tools.discovery.handlers_board_tasks import update_board_task
    from modules.tools.execution.unified_executor import map_optional_aliases, undeclared_params_refusal

    action = get_action_registry().get("platform_update_task")
    params = map_optional_aliases("platform_update_task", action, params, "f309")
    refused = undeclared_params_refusal("platform_update_task", action, params, "f309")
    assert refused is None, refused                                           # night 9: refused here
    return asyncio.run(update_board_task(shop.db, shop.ws, params))


def _notes(task):
    return [(n.get("by"), n.get("note")) for n in (task.runtime_ref or {}).get("session_notes") or []]


def test_approve_with_notes_and_a_status_on_the_edit_tool_is_the_approval(shop):
    task = _in_review(shop)

    out = _edit(shop, task_id=task.id, notes=NOTE, status="done", _user_id=OWNER)

    shop.db.refresh(task)
    assert out["success"] is True and task.status == "done"                    # night 9: "Nothing to change"
    assert ("you", f"Approved: {NOTE}") in _notes(task)                        # the approval's own note
    assert shop.filed == [task.id]                                            # Done files the card, as ever


def test_an_edit_and_a_status_in_one_call_do_both(shop):
    task = _in_review(shop)

    out = _edit(shop, task_id=task.id, priority="high", status="approved", note=NOTE, _user_id=OWNER)

    shop.db.refresh(task)
    assert out["success"] is True and (task.priority, task.status) == ("high", "done")
    assert ("you", f"Approved: {NOTE}") in _notes(task)


def test_a_new_brief_on_an_answered_card_takes_no_status(shop):
    """A Re-brief sends the card back itself: a status beside it would undo that."""
    from modules.tools.discovery.ticket_edit_moves import REBRIEF_MOVES_IT

    task = _in_review(shop)

    out = _edit(shop, task_id=task.id, description="Use the September sheet only.", status="done", _user_id=OWNER)

    shop.db.refresh(task)
    assert out["success"] is False and out["error"] == REBRIEF_MOVES_IT
    assert (task.status, task.result) == ("review", MARGIN)


def test_the_move_is_recorded_as_the_move_it_made():
    from modules.tools.execution.call_effects import call_effects

    assert call_effects("platform_update_task", {"task_id": 1866, "status": "done"}) == (
        "platform_update_task_status:done",)
    assert call_effects("platform_update_task", {"task_id": 1866, "title": "Margin"}) == ()
