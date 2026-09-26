"""#1115 — Run Now says why a ticket cannot start, never "started".

Night 6: Run Now on #1115, a Claude Code agent's ticket, answered "Ticket #1115
started." No CLI host serves that workspace (the only one is paired to another),
so nothing could claim it. Run Now still queues it, since a host claims it the
moment one is back, but now says it cannot start yet and why. An agent whose
model cannot run is not handed the ticket at all (F141's rule, as on create and
assign).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import text


@pytest.fixture
def board(db_session, seed_workspace, monkeypatch):
    import api.board_tasks as bt

    ws = UUID(seed_workspace())

    def agent(name, configuration):
        return db_session.execute(
            text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
                 "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', CAST(:c AS json)) RETURNING id"),
            {"n": name, "w": str(ws), "c": configuration}).scalar()

    woken = []
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: woken.append(kw["task_id"]))
    return NS(db=db_session, ws=ws, woken=woken, api_agent=agent("Content Creator", "{}"),
              cli_agent=agent("Numbers (on my Mac)", '{"runtime": "cli"}'),
              ctx=NS(workspace_id=ws, user=NS(id="owner@cafe.test", email="owner@cafe.test")))


def _ticket(board, agent):
    from core.models.core import BoardTask

    task = BoardTask(workspace_id=board.ws, title="Reconcile the September takings", priority="medium",
                     source_type="user", status="blocked", assigned_agent_id=agent, attempts=1)
    board.db.add(task)
    board.db.flush()
    return task


def _run_now(board, task):
    import api.board_tasks as bt

    return asyncio.run(bt.run_task_now(task.id, ctx=board.ctx, db=board.db))


def test_1115_a_ticket_no_host_can_claim_is_queued_and_says_it_cannot_start(board):
    task = _ticket(board, board.cli_agent)
    reply = _run_now(board, task)
    assert "nothing can start it yet: Waiting for a CLI host" in reply["message"]
    assert reply["started"] is False and reply["status"] == "assigned" and board.woken == [task.id]


def test_a_ticket_the_dispatcher_can_run_says_started(board):
    task = _ticket(board, board.api_agent)
    reply = _run_now(board, task)
    assert reply["started"] is True and reply["message"] == f"Ticket #{task.id} started."


def test_an_agent_whose_model_cannot_run_is_not_handed_the_ticket(board, monkeypatch):
    monkeypatch.setattr("core.llm.model_refusals.unavailable_reason",
                        lambda db, agent: "Its model 'gpt-5.6' is quarantined")
    task = _ticket(board, board.api_agent)
    with pytest.raises(HTTPException) as refused:
        _run_now(board, task)
    assert refused.value.status_code == 409
    assert refused.value.detail.startswith("Its model 'gpt-5.6' is quarantined, so it cannot take this task")
    board.db.refresh(task)
    assert task.status == "blocked" and board.woken == []
