"""PRD-274 Wave 1 (§8-8): ``POST /api/agents/{id}/execute`` starts a real run.

It files a ticket assigned to the agent, which the board's dispatch loop claims and runs, and
answers at once with the ticket's id. It replaced a route that made up an execution id and ran
nothing; the copy at ``/api/system/agent/{id}/execute`` is gone. Real database.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response
from pydantic import ValidationError

from api.agent_execute import AgentRunRequest, execute_agent

ASKED = "Draft the renewal reminder for Acme's March invoice.\nKeep it under 120 words."


@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


@pytest.fixture
def seeded(new_session):
    """A throwaway workspace with an active agent and a paused one; swept on teardown."""
    from sqlalchemy import text

    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w1-execute')"), {"id": ws})
    ids = [s.execute(text("INSERT INTO agents (name, agent_type, workspace_id, status) "
                          "VALUES (:n, 'custom', CAST(:w AS uuid), :st) RETURNING id"),
                     {"n": name, "w": ws, "st": st}).scalar() for name, st in (("Billing", "active"),
                                                                                ("Paused", "inactive"))]
    s.commit()
    s.close()
    yield ws, ids[0], ids[1]
    s = new_session.sweep()
    for table in ("approval_grants", "board_tasks", "agents"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def _ctx(ws: str) -> SimpleNamespace:
    return SimpleNamespace(workspace_id=uuid.UUID(ws), auth_type="clerk",
                           user=SimpleNamespace(id="u-1", clerk_user_id="clerk_owner", email=None))


def test_execute_files_a_ticket_the_dispatch_loop_claims(seeded, new_session):
    from core.models import BoardTask
    from services import board_dispatcher

    ws, agent_id, _ = seeded
    s = new_session()
    answer = execute_agent(agent_id, AgentRunRequest(input=ASKED), Response(), ctx=_ctx(ws), db=s)
    s.close()
    assert answer["agent_id"] == agent_id and answer["status"] == "assigned"
    assert answer["task_url"] == f"/api/v1/tasks/{answer['task_id']}"

    s = new_session()
    task = s.get(BoardTask, answer["task_id"])
    assert (str(task.workspace_id), task.assigned_agent_id, task.source_type) == (ws, agent_id, "api")
    assert task.raw_prompt == ASKED and task.title == ASKED.splitlines()[0]
    claimed = board_dispatcher.claim_tasks(s, worker_id="w1-execute", limit=50, lease_seconds=60)
    s.commit()
    s.close()
    assert answer["task_id"] in [row["id"] if isinstance(row, dict) else row.id for row in claimed]


def test_a_title_given_is_the_tickets_title(seeded, new_session):
    from core.models import BoardTask

    ws, agent_id, _ = seeded
    s = new_session()
    answer = execute_agent(agent_id, AgentRunRequest(input=ASKED, title="Acme renewal"), Response(), ctx=_ctx(ws), db=s)
    assert s.get(BoardTask, answer["task_id"]).title == "Acme renewal"
    s.close()


def test_an_inactive_agent_or_another_workspaces_agent_is_refused(seeded, new_session):
    from sqlalchemy import text

    ws, agent_id, paused_id = seeded
    s = new_session()
    try:
        with pytest.raises(HTTPException) as paused:
            execute_agent(paused_id, AgentRunRequest(input=ASKED), Response(), ctx=_ctx(ws), db=s)
        with pytest.raises(HTTPException) as stranger:
            execute_agent(agent_id, AgentRunRequest(input=ASKED), Response(), ctx=_ctx(str(uuid.uuid4())), db=s)
        filed = s.execute(text("SELECT count(*) FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)"),
                          {"w": ws}).scalar()
    finally:
        s.close()
    assert (paused.value.status_code, stranger.value.status_code, filed) == (409, 404, 0)


@pytest.mark.parametrize("body", [{}, {"input": ""}, {"input": "x" * 20_001},
                                  {"input": "Go", "title": ""}, {"input": "Go", "parameters": {"a": 1}}])
def test_the_request_is_checked(body):
    with pytest.raises(ValidationError):
        AgentRunRequest(**body)


def test_the_made_up_routes_are_gone_and_the_real_one_is_gated():
    from fastapi.routing import APIRoute, iter_route_contexts

    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR
    from main import app

    routes = {(m, r.path): r for r in iter_route_contexts(app.routes)
              if isinstance(r.original_route, APIRoute) for m in r.methods}
    assert ("POST", "/api/system/agent/{agent_id}/execute") not in routes
    route = routes[("POST", "/api/agents/{agent_id}/execute")]
    assert route.endpoint is execute_agent and route.original_route.status_code == 202
    marks = [getattr(d.call, PERMISSION_MARKER_ATTR, None) for d in route.dependant.dependencies]
    assert "agents:execute" in marks
