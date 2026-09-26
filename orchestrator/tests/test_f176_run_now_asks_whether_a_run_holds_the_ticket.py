"""F176 — Run Now asks whether a run really holds the ticket, not what its status word says.

Night 6, ticket #1094: 'in_progress' with 0 attempts, no lease and no execution.
Auto had set that status before the ticket had an agent, and that write runs
nothing. Run Now answered "already running — nothing to start". It now calls a
ticket running only when a run holds it: a live claim (the dispatch lease every
run renews, and the lease a CLI host renews for its session) or, for a playbook
step, its playbook run still going. The lease heartbeat now renews at once, so a
run launched directly holds its claim from the start, not after half a lease window.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import text


@pytest.fixture
def board(db_session, seed_workspace, monkeypatch):
    import api.board_tasks as bt

    ws = UUID(seed_workspace())
    agent = db_session.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES ('Content Creator', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(ws)}).scalar()
    woken = []
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: woken.append(kw))
    return NS(db=db_session, ws=ws, agent=agent, woken=woken,
              ctx=NS(workspace_id=ws, user=NS(id="owner@cafe.test", email="owner@cafe.test")))


def _ticket(board, **fields):
    from core.models.core import BoardTask

    task = BoardTask(**{"workspace_id": board.ws, "title": "Write the shop descriptions", "priority": "medium",
                        "source_type": "user", "assigned_agent_id": board.agent, "attempts": 0, **fields})
    board.db.add(task)
    board.db.flush()
    return task


def _run_now(board, task):
    import api.board_tasks as bt

    return asyncio.run(bt.run_task_now(task.id, ctx=board.ctx, db=board.db))


def test_a_ticket_only_its_status_word_calls_running_is_started(board):
    task = _ticket(board, status="in_progress", started_at=datetime.now(timezone.utc) - timedelta(minutes=1))  # #1094
    reply = _run_now(board, task)
    assert reply["status"] == "assigned" and "nothing was running it" in reply["message"]
    assert board.woken                                                  # the dispatch loop is woken


def test_a_ticket_a_run_holds_is_left_alone(board):
    task = _ticket(board, status="in_progress", started_at=datetime.now(timezone.utc) - timedelta(minutes=1),
                   lease_until=datetime.now(timezone.utc) + timedelta(minutes=10))
    with pytest.raises(HTTPException) as refused:
        _run_now(board, task)
    assert refused.value.status_code == 409 and "already running" in refused.value.detail
    assert not board.woken


@pytest.mark.parametrize("source", ["orchestration", "orchestration_task"])
def test_a_missions_step_is_never_run_again_by_the_board(board, source):
    """Review HIGH: a mission's steps hold no board lease; the mission engine runs them."""
    task = _ticket(board, status="in_progress", source_type=source, source_id="run-7:task-3",
                   started_at=datetime.now(timezone.utc) - timedelta(minutes=20))
    with pytest.raises(HTTPException) as refused:
        _run_now(board, task)
    assert refused.value.status_code == 409 and not board.woken


def test_a_playbook_step_is_running_while_its_playbook_run_is():
    import api.board_tasks as bt

    db = MagicMock()
    db.execute.return_value.first.return_value = (1,)
    step = NS(status="in_progress", lease_until=None, source_type="recipe", source_id="recipe:exec-42:2")
    assert bt._running_now(db, step) is True
    assert db.execute.call_args.args[1] == {"e": "exec-42"}
    db.execute.return_value.first.return_value = None
    assert bt._running_now(db, step) is False


def test_a_run_holds_its_claim_from_its_first_moment():
    import api.board_tasks as bt

    renewed = []

    async def _stop(_seconds):
        raise asyncio.CancelledError

    with patch("core.database.database.SessionLocal", MagicMock()), \
            patch("services.board_dispatcher.renew_lease",
                  lambda db, task_id, lease_seconds, **_: renewed.append(task_id) or True), \
            patch.object(bt.asyncio, "sleep", _stop):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(bt._lease_heartbeat(1094))
    assert renewed == [1094]                                            # before the first wait
