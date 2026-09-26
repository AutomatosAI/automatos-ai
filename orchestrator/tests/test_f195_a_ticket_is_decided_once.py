"""F195 — a ticket in review is decided once.

approve_task checked the status it had read, ran the ticket's approval action,
then set done, so a double click (or two admins) that both read 'review' both
ran it: two blog missions from one create_blog ticket. reject_task did the same
and could reset a redo that had already started. Both now move the ticket with
a compare-and-set on the status the request saw; the request that loses is told
the ticket was already decided, and nothing runs again.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, text

TOPIC = "Why our oat flat white costs 20p more"
WS_GONE = UUID("00000000-0000-0000-0000-0000000000c1")


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the decide-once race tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture
def board(engine, new_session, monkeypatch):
    import api.board_tasks as bt
    import modules.tools.discovery.handlers_blog as blog

    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f195')"), {"id": ws})
    agent = s.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
        "VALUES ('VECTOR', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"), {"w": ws}).scalar()
    ticket = s.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, status, assigned_agent_id, review_mode, planning_data) "
        "VALUES (CAST(:w AS uuid), 'Blog idea', 'review', :a, 'human', CAST(:pd AS jsonb)) RETURNING id"),
        {"w": ws, "a": agent, "pd": json.dumps({"approval_action": {"type": "create_blog", "topic": TOPIC}})}).scalar()
    s.commit()

    missions, outcome = [], {"success": True, "during": None}

    async def _start_blog_mission(db, workspace_id, params):
        if outcome["during"]:                                      # something lands while the planner runs
            await outcome["during"]()
        if outcome["success"]:
            missions.append(params["topic"])
            return {"success": True, "mission_id": f"m-{len(missions)}", "task_count": 5}
        return {"success": False, "error": "planner unavailable"}

    async def _no_notice(*a, **k):
        return None

    monkeypatch.setattr(blog, "create_blog_post_from_topic", _start_blog_mission)
    monkeypatch.setattr(bt, "_dispatch_task_complete", _no_notice)
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: None)
    ctx = NS(workspace_id=UUID(ws), user=NS(id=7, clerk_user_id=None, email="owner@cafe.test"))
    yield NS(id=ticket, ws=ws, ctx=ctx, missions=missions, outcome=outcome, new=new_session)

    s = new_session.sweep()
    for table, col in (("board_tasks", "workspace_id"), ("agents", "workspace_id"), ("workspaces", "id")):
        s.execute(text(f"DELETE FROM {table} WHERE {col} = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def _stale_read(board):
    """A second click's session, holding the ticket as it read it while in review."""
    from core.models.core import BoardTask

    session = board.new()
    seen = session.query(BoardTask).filter(BoardTask.id == board.id).first()
    assert seen.status == "review"
    return session, seen


def _call(endpoint, board, session, body):
    try:
        return asyncio.run(endpoint(board.id, _Req(body), ctx=board.ctx, db=session))
    except HTTPException as refused:
        return refused


def _status(board):
    return board.new().execute(text("SELECT status FROM board_tasks WHERE id = :i"), {"i": board.id}).scalar()


def _already_decided(outcome):
    return (isinstance(outcome, HTTPException) and outcome.status_code == 422
            and "already decided" in outcome.detail)


def test_two_approvals_at_once_start_one_blog_mission(board):
    import api.board_tasks as bt

    second, _held = _stale_read(board)

    first = _call(bt.approve_task, board, board.new(), {})
    late = _call(bt.approve_task, board, second, {})

    assert first["status"] == "done" and first["action_result"]["mission_id"] == "m-1"
    assert board.missions == [TOPIC]                              # night: two missions
    assert _already_decided(late)


def test_a_second_send_back_leaves_the_redo_that_started_alone(board):
    import api.board_tasks as bt

    second, _held = _stale_read(board)
    assert _call(bt.reject_task, board, board.new(), {"feedback": "Shorter."})["status"] == "assigned"
    claim = board.new()                                           # the board picks the redo up
    claim.execute(text("UPDATE board_tasks SET status = 'in_progress', started_at = now(), attempts = 1 "
                       "WHERE id = :i"), {"i": board.id})
    claim.commit()

    late = _call(bt.reject_task, board, second, {"feedback": "Shorter."})

    assert _status(board) == "in_progress"                        # night: reset to 'assigned', run twice
    assert _already_decided(late)


def test_a_send_back_that_read_review_after_the_approval_landed_is_already_decided(board):
    import api.board_tasks as bt

    second, _held = _stale_read(board)
    assert _call(bt.approve_task, board, board.new(), {})["status"] == "done"

    late = _call(bt.reject_task, board, second, {"feedback": "Not this topic."})

    assert _status(board) == "done"                               # night: the approved ticket sent back unseen
    assert _already_decided(late)


def test_an_approval_whose_action_failed_leaves_the_ticket_in_review_to_approve_again(board):
    """The approval is committed before its action runs; a failed action gives the ticket back."""
    import api.board_tasks as bt

    board.outcome["success"] = False
    failed = _call(bt.approve_task, board, board.new(), {})
    assert isinstance(failed, HTTPException) and failed.status_code == 500
    assert _status(board) == "review" and board.missions == []

    board.outcome["success"] = True
    assert _call(bt.approve_task, board, board.new(), {})["status"] == "done"
    assert board.missions == [TOPIC]


def test_a_send_back_while_the_approval_runs_keeps_the_ticket_it_sent_back(board):
    """The approval's 'done' is committed before its action runs; a send-back in that window wins the ticket."""
    import api.board_tasks as bt

    async def _send_back():
        sent = await bt.reject_task(board.id, _Req({"feedback": "Redo it."}), ctx=board.ctx, db=board.new())
        assert sent["status"] == "assigned"

    board.outcome["during"] = _send_back
    approved = _call(bt.approve_task, board, board.new(), {})

    assert board.missions == [TOPIC]                              # the approval's own action ran
    row = board.new().execute(text("SELECT status, result FROM board_tasks WHERE id = :i"), {"i": board.id}).one()
    assert tuple(row) == ("assigned", None)                       # before: its result written onto the sent-back ticket
    assert approved["status"] == "assigned"


def test_a_ticket_deleted_while_it_was_being_decided_is_not_found():
    import api.board_tasks as bt
    from core.models.core import BoardTask
    from sqlalchemy.exc import InvalidRequestError

    class _Gone:
        def filter(self, *a, **k):
            return self

        def first(self):
            return BoardTask(id=9, workspace_id=WS_GONE, title="t", status="review", planning_data={})

        def update(self, *a, **k):
            return 0

    class _Session:
        def query(self, _model):
            return _Gone()

        def rollback(self):
            pass

        def refresh(self, _row):
            raise InvalidRequestError("Could not refresh instance")

    ctx = NS(workspace_id=WS_GONE, user=NS(id=7, clerk_user_id=None, email="owner@cafe.test"))
    with pytest.raises(HTTPException) as gone:                      # before: InvalidRequestError, a 500
        asyncio.run(bt.approve_task(9, _Req({}), ctx=ctx, db=_Session()))
    assert gone.value.status_code == 404
