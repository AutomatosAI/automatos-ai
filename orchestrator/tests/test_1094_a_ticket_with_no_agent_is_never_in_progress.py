"""#1094 — a ticket with no agent is never in progress.

Night 6: Auto set ticket #1094 'in_progress' before it had an agent, then
assigned one. The first write ran nothing, and assigning moves only an inbox
ticket to 'assigned', so the ticket sat 'in progress' with nothing running it
and the dispatcher never claimed it. platform_update_task_status and the board's
PATCHes now refuse in_progress for a ticket with no agent ("Assign an agent
first"); a PATCH that assigns an agent in the same body may start it.

PRD-227 P227-RVW-4's redo reset lived on that no-agent write, the only
in_progress that reached it. It moves to the start an agent's redo takes, so a
redo still begins without the last run's completed_at, error_message and result.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import text


@pytest.fixture
def board(db_session, seed_workspace, monkeypatch):
    import api.board_tasks as bt
    from modules.tools.discovery import handlers_board_tasks as handlers

    ws = UUID(seed_workspace())
    agent = db_session.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES ('Content Creator', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(ws)}).scalar()
    woken, launched = [], []
    monkeypatch.setattr(handlers, "_notify_dispatch_safe", lambda db, ws, task_id: woken.append(task_id))
    monkeypatch.setattr(bt, "_launch_task_execution", lambda **kw: launched.append(kw["task_id"]))
    return NS(db=db_session, ws=ws, agent=agent, woken=woken, launched=launched,
              ctx=NS(workspace_id=ws, user=NS(id="owner@cafe.test", email="owner@cafe.test")))


def _ticket(board, **fields):
    from core.models.core import BoardTask

    task = BoardTask(**{"workspace_id": board.ws, "title": "Write the shop descriptions", "priority": "medium",
                        "source_type": "user", "status": "inbox", "attempts": 0, **fields})
    board.db.add(task)
    board.db.flush()
    return task


def _status_by_tool(board, **params):
    from modules.tools.discovery.handlers_board_tasks import update_board_task_status

    return asyncio.run(update_board_task_status(board.db, board.ws, {"status": "in_progress", **params}))


class _Request:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def _refusal():
    from api.board_tasks import NO_AGENT_NO_PROGRESS

    return NO_AGENT_NO_PROGRESS


# ── the tool ────────────────────────────────────────────────────────────────

def test_1094s_path_leaves_a_ticket_the_dispatcher_runs(board):
    from modules.tools.discovery.handlers_board_tasks import _is_dispatch_claimable, assign_board_task

    task = _ticket(board)
    first = _status_by_tool(board, task_id=task.id)
    reply = asyncio.run(assign_board_task(board.db, board.ws, {"task_id": task.id, "agent_name": "Content Creator"}))
    board.db.refresh(task)
    assert reply["status"] == task.status == "assigned" and task.started_at is None   # not 'in progress', unrun
    assert _is_dispatch_claimable(task) and board.woken == [task.id]
    assert first == {"success": False, "error": _refusal()}


def test_a_bulk_move_names_the_ticket_with_no_agent(board):
    ready, bare = _ticket(board, status="assigned", assigned_agent_id=board.agent), _ticket(board)
    reply = _status_by_tool(board, task_ids=[ready.id, bare.id])
    assert reply["updated"] == [ready.id] and board.launched == [ready.id]
    assert reply["failed"] == [{"task_id": bare.id, "error": _refusal()}]


def _redo(board):
    return _ticket(board, status="done", assigned_agent_id=board.agent, completed_at=datetime.now(timezone.utc),
                   error_message="prior run blew up", result="partial output")


def test_an_agents_redo_starts_clean(board):
    task = _redo(board)
    reply = _status_by_tool(board, task_id=task.id)
    board.db.refresh(task)
    assert reply["triggered_execution"] is True and board.launched == [task.id]
    assert (task.status, task.completed_at, task.error_message, task.result) == ("in_progress", None, None, None)


def test_a_redo_that_succeeds_does_not_render_failed(board):
    task = _redo(board)
    _status_by_tool(board, task_id=task.id)
    _status_by_tool(board, task_id=task.id, status="done")
    board.db.refresh(task)
    assert (task.status, task.error_message, task.result) == ("done", None, None) and task.completed_at


# ── the board ───────────────────────────────────────────────────────────────

def test_dragging_a_ticket_with_no_agent_to_in_progress_is_refused(board):
    import api.board_tasks as bt

    task = _ticket(board)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(bt.update_task_status(task.id, _Request({"status": "in_progress"}), ctx=board.ctx, db=board.db))
    assert (refused.value.status_code, refused.value.detail) == (409, _refusal())
    board.db.refresh(task)
    assert task.status == "inbox" and task.started_at is None and board.launched == []


def test_a_patch_setting_in_progress_with_no_agent_changes_nothing(board):
    import api.board_tasks as bt

    task = _ticket(board)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(bt.update_task(task.id, _Request({"status": "in_progress", "title": "Renamed"}),
                                   ctx=board.ctx, db=board.db))
    assert (refused.value.status_code, refused.value.detail) == (409, _refusal())
    board.db.refresh(task)
    assert (task.status, task.title) == ("inbox", "Write the shop descriptions")


def test_a_patch_that_assigns_an_agent_may_start_the_ticket(board):
    import api.board_tasks as bt

    task = _ticket(board)
    reply = asyncio.run(bt.update_task(task.id, _Request({"status": "in_progress", "assigned_agent_id": board.agent}),
                                       ctx=board.ctx, db=board.db))
    assert reply["status"] == "in_progress" and board.launched == [task.id]


# ── review fixups ───────────────────────────────────────────────────────────

def _in_review(board):
    return _ticket(board, status="review", assigned_agent_id=board.agent, result="Draft v1: Dear Sunil…",
                   completed_at=datetime.now(timezone.utc))


def test_an_agents_redo_keeps_the_draft_under_review_on_record(board):
    """Review MEDIUM: the redo reset cleared the result a person was reviewing, with no copy."""
    task = _in_review(board)
    _status_by_tool(board, task_id=task.id)
    board.db.refresh(task)
    assert task.result is None
    assert (task.planning_data or {}).get("previous_runs", [{}])[-1].get("result") == "Draft v1: Dear Sunil…"


def test_dragging_a_ticket_back_from_review_keeps_its_draft_on_record(board):
    import api.board_tasks as bt

    task = _in_review(board)
    asyncio.run(bt.update_task_status(task.id, _Request({"status": "in_progress"}), ctx=board.ctx, db=board.db))
    board.db.refresh(task)
    assert (task.planning_data or {}).get("previous_runs", [{}])[-1].get("result") == "Draft v1: Dear Sunil…"


@pytest.mark.parametrize("agent", ["Content Creator", [1], {"id": 1}, True])
def test_a_patch_whose_agent_is_not_an_id_is_refused_before_anything_changes(board, agent):
    """Review LOW: the in-progress check judged the raw value's truth."""
    import api.board_tasks as bt

    task = _ticket(board)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(bt.update_task(task.id, _Request({"status": "in_progress", "assigned_agent_id": agent}),
                                   ctx=board.ctx, db=board.db))
    assert refused.value.status_code == 422
    board.db.refresh(task)
    assert task.status == "inbox"
