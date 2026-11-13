"""PRD-274 Wave 1 (§5.2): an ``Idempotency-Key`` makes a retried ``POST /api/agents/{id}/execute`` safe.

The key files one ticket per workspace. Repeating the request answers 200 with the run it started
(``Idempotent-Replayed: true``); the same key on a different request is a 409; two requests racing
on one key still file one ticket. Real database.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response
from sqlalchemy import text

from api import agent_execute
from api.agent_execute import AgentRunRequest, execute_agent

ASKED = "Draft Acme's renewal reminder."


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    import importlib.util
    from pathlib import Path

    from sqlalchemy import create_engine

    from core.database.database import get_database_url
    from core.models.run_callbacks import RunCallback

    migration = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "execute_idempotency_20261109.py"
    spec = importlib.util.spec_from_file_location("execute_idempotency_20261109", migration)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    eng = create_engine(get_database_url(), pool_pre_ping=True)
    RunCallback.__table__.create(eng, checkfirst=True)
    with eng.begin() as conn:  # a test database built before the index
        conn.execute(text(module._INDEX))
    yield eng
    eng.dispose()


@pytest.fixture
def seeded(new_session):
    workspaces = [str(uuid.uuid4()), str(uuid.uuid4())]
    s = new_session()
    agents = []
    for ws in workspaces:
        s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w1-idempotency')"), {"id": ws})
        agents.append(s.execute(text("INSERT INTO agents (name, agent_type, workspace_id, status) "
                                     "VALUES ('Billing', 'custom', CAST(:w AS uuid), 'active') RETURNING id"),
                                {"w": ws}).scalar())
    s.commit()
    s.close()
    yield list(zip(workspaces, agents))
    s = new_session.sweep()
    for ws in workspaces:
        for table in ("run_callbacks", "approval_grants", "board_tasks", "agents"):
            s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def _ctx(ws):
    return SimpleNamespace(workspace_id=uuid.UUID(ws), auth_type="clerk",
                           user=SimpleNamespace(id="u-1", clerk_user_id="clerk_owner", email=None))


def _execute(new_session, ws, agent_id, key, asked=ASKED):
    response, s = Response(), new_session()
    try:
        answer = execute_agent(agent_id, AgentRunRequest(input=asked), response, idempotency_key=key,
                               ctx=_ctx(ws), db=s)
    finally:
        s.close()
    return answer, response


def _tickets(new_session, ws):
    s = new_session()
    try:
        return s.execute(text("SELECT count(*) FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)"),
                         {"w": ws}).scalar()
    finally:
        s.close()


def test_a_repeat_with_the_same_key_answers_with_the_run_it_started(seeded, new_session):
    (ws, agent_id), _ = seeded
    first, first_response = _execute(new_session, ws, agent_id, "renewal-acme-0042")
    again, again_response = _execute(new_session, ws, agent_id, "renewal-acme-0042")
    assert "replayed" not in first and agent_execute.REPLAYED_HEADER not in first_response.headers
    assert again["task_id"] == first["task_id"] and again["replayed"] is True
    assert again["agent_name"] == "Billing" and again_response.status_code == 200
    assert again_response.headers[agent_execute.REPLAYED_HEADER] == "true"
    assert _tickets(new_session, ws) == 1


def test_the_same_key_on_a_different_request_is_a_conflict(seeded, new_session):
    (ws, agent_id), _ = seeded
    _execute(new_session, ws, agent_id, "renewal-acme-0042")
    with pytest.raises(HTTPException) as other:
        _execute(new_session, ws, agent_id, "renewal-acme-0042", asked="Draft Acme's cancellation notice.")
    assert other.value.status_code == 409 and _tickets(new_session, ws) == 1


def test_no_key_or_another_key_or_another_workspace_files_a_new_run(seeded, new_session):
    (ws, agent_id), (other_ws, other_agent) = seeded
    _execute(new_session, ws, agent_id, None)
    _execute(new_session, ws, agent_id, None)
    _execute(new_session, ws, agent_id, "k-1")
    _execute(new_session, ws, agent_id, "k-2")
    theirs, _ = _execute(new_session, other_ws, other_agent, "k-1")
    assert _tickets(new_session, ws) == 4 and _tickets(new_session, other_ws) == 1 and "replayed" not in theirs


def test_two_requests_racing_on_one_key_file_one_ticket(seeded, new_session, monkeypatch):
    (ws, agent_id), _ = seeded
    winner, _ = _execute(new_session, ws, agent_id, "race-1")
    looked = []
    real = agent_execute._earlier_run

    def first_look_misses(db, workspace_id, key):  # the loser looked before the winner committed
        looked.append(key)
        return None if len(looked) == 1 else real(db, workspace_id, key)

    monkeypatch.setattr(agent_execute, "_earlier_run", first_look_misses)
    loser, response = _execute(new_session, ws, agent_id, "race-1")
    assert loser["task_id"] == winner["task_id"] and loser["replayed"] is True and response.status_code == 200
    assert len(looked) == 2 and _tickets(new_session, ws) == 1


@pytest.mark.parametrize("key", ["", "has space", "tab\\there", "é", "x" * 256])
def test_a_malformed_key_is_refused(key):
    with pytest.raises(HTTPException) as refused:
        agent_execute._checked_key(key.replace("\\t", "\t"))
    assert refused.value.status_code == 422


def test_the_model_declares_the_migrations_index():
    from core.models.core import BoardTask

    index = next(i for i in BoardTask.__table__.indexes if i.name == "uq_board_tasks_api_idempotency")
    assert index.unique and [c.name for c in index.columns] == ["workspace_id", "source_id"]
    assert "source_type = 'api'" in str(index.dialect_options["postgresql"]["where"])


def test_the_route_reads_the_key_from_the_idempotency_key_header():
    from main import app

    operation = app.openapi()["paths"]["/api/agents/{agent_id}/execute"]["post"]
    headers = {p["name"]: p for p in operation["parameters"] if p["in"] == "header"}
    assert "Idempotency-Key" in headers and not headers["Idempotency-Key"].get("required", False)
