"""PRD-274 Wave 1 (§5.2): a product's server key reaches the routes named for it, with a scope.

``require_workspace_permission_or_server_key`` lets a server key (``ak_srv_``) that lists the
route's scope through, bound to the key's workspace. It refuses a publishable key, a key without
the scope (an empty list grants nothing), a revoked key, and a workspace header that isn't the
key's. Anyone else meets ``require_workspace_permission`` as before. Wired first on
``POST /api/agents/{id}/execute`` and ``POST /api/v1/tasks``. Real database, real keys.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response
from sqlalchemy import text
from starlette.requests import Request

from core.auth import server_key_gate as gate
from core.auth.scopes import AGENTS_EXECUTE, TASKS_WRITE

ASKED = "Draft Acme's renewal reminder."


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


@pytest.fixture
def workspace(new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w1-server-keys')"), {"id": ws})
    agent_id = s.execute(text("INSERT INTO agents (name, agent_type, workspace_id, status) "
                              "VALUES ('Billing', 'custom', CAST(:w AS uuid), 'active') RETURNING id"), {"w": ws}).scalar()
    s.commit()
    s.close()
    yield ws, agent_id
    s = new_session.sweep()
    for table in ("run_callbacks", "approval_grants", "board_tasks", "agents", "sdk_api_keys"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def _mint(new_session, ws, *, key_type="server", permissions=(AGENTS_EXECUTE,)):
    from core.services.api_key_service import ApiKeyService

    s = new_session()
    try:
        return ApiKeyService.create_api_key(s, uuid.UUID(ws), "acme backend", key_type=key_type,
                                            permissions=list(permissions))
    finally:
        s.close()


def _request(token=None, **headers):
    pairs = [(k.lower().replace("_", "-").encode(), v.encode()) for k, v in headers.items()]
    if token:
        pairs.append((b"authorization", f"Bearer {token}".encode()))
    return Request({"type": "http", "method": "POST", "path": "/", "headers": pairs, "query_string": b"",
                    "path_params": {}})


def _pass(request, permission="agents:execute", scope=AGENTS_EXECUTE, db=None):
    return asyncio.run(gate.require_workspace_permission_or_server_key(permission, scope)(request, db=db))


def test_a_server_key_with_the_scope_is_bound_to_its_workspace(new_session, workspace):
    ws, _ = workspace
    key = _mint(new_session, ws)
    ctx = _pass(_request(key["key"]))
    assert (str(ctx.workspace_id), ctx.auth_type, ctx.api_key_id) == (ws, "sdk_key", key["id"])
    assert gate.is_server_key(ctx)
    assert str(_pass(_request(key["key"], x_workspace_id=ws)).workspace_id) == ws


@pytest.mark.parametrize("permissions", [(), ("chat",), (TASKS_WRITE,)])
def test_a_server_key_without_the_scope_is_refused(new_session, workspace, permissions):
    ws, _ = workspace
    key = _mint(new_session, ws, permissions=permissions)
    with pytest.raises(HTTPException) as refused:
        _pass(_request(key["key"]))
    assert refused.value.status_code == 403


def test_a_publishable_key_is_refused_even_with_the_scope(new_session, workspace):
    ws, _ = workspace
    key = _mint(new_session, ws, key_type="public")
    assert key["key"].startswith(gate.PUBLISHABLE_KEY_PREFIX)
    with pytest.raises(HTTPException) as refused:
        _pass(_request(key["key"]))
    assert (refused.value.status_code, refused.value.detail) == (403, gate.PUBLISHABLE_KEY_REFUSED)


def test_a_revoked_or_unknown_key_or_another_workspace_header_is_refused(new_session, workspace):
    from core.services.api_key_service import ApiKeyService

    ws, _ = workspace
    key = _mint(new_session, ws)
    with pytest.raises(HTTPException) as mismatch:
        _pass(_request(key["key"], x_workspace_id=str(uuid.uuid4())))
    s = new_session()
    ApiKeyService.revoke_api_key(s, uuid.UUID(key["id"]), uuid.UUID(ws))
    s.close()
    with pytest.raises(HTTPException) as revoked:
        _pass(_request(key["key"]))
    with pytest.raises(HTTPException) as unknown:
        _pass(_request("ak_srv_" + "0" * 40))
    assert (mismatch.value.status_code, revoked.value.status_code, unknown.value.status_code) == (403, 401, 401)


def test_anyone_else_meets_the_member_gate_as_before(monkeypatch):
    member = SimpleNamespace(workspace_id=uuid.uuid4(), auth_type="clerk",
                             user=SimpleNamespace(id="u", clerk_user_id="c", system_role="user"))

    async def signed_in(request):
        return member

    granted = {}
    monkeypatch.setattr(gate, "get_request_context_hybrid", signed_in)
    monkeypatch.setattr("core.auth.workspace_permission.workspace_permission_granted",
                        lambda db, ctx, permission: granted.setdefault(permission, False))
    with pytest.raises(HTTPException) as viewer:
        _pass(_request())
    assert viewer.value.status_code == 403 and granted == {"agents:execute": False}
    monkeypatch.setattr("core.auth.workspace_permission.workspace_permission_granted", lambda db, ctx, p: True)
    assert _pass(_request()) is member


def test_a_server_key_starts_a_run_its_ticket_says_a_key_filed(new_session, workspace):
    from api.agent_execute import AgentRunRequest, execute_agent
    from core.models import BoardTask

    ws, agent_id = workspace
    ctx = _pass(_request(_mint(new_session, ws)["key"]))
    s = new_session()
    try:
        answer = execute_agent(agent_id, AgentRunRequest(input=ASKED), Response(), ctx=ctx, db=s)
        task = s.get(BoardTask, answer["task_id"])
        assert (str(task.workspace_id), task.created_by_type, task.created_by_id) == (ws, "api_key", ctx.user.id)
    finally:
        s.close()



class _Body:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def _grants(new_session, task_id):
    s = new_session()
    try:
        return s.execute(text("SELECT count(*) FROM approval_grants WHERE subject_type = 'board_task' AND subject_id = :t"), {"t": str(task_id)}).scalar()
    finally:
        s.close()


def test_a_server_key_files_a_ticket_and_its_ticket_says_a_key_filed(new_session, workspace):
    from api.board_tasks import create_task
    from core.models import BoardTask

    ws, agent_id = workspace
    ctx = _pass(_request(_mint(new_session, ws, permissions=(TASKS_WRITE,))["key"]), "missions:create", TASKS_WRITE)
    s = new_session()
    try:
        answer = asyncio.run(create_task(_Body({"title": "Renew Acme", "assigned_agent_id": agent_id}), ctx=ctx, db=s))
        task = s.get(BoardTask, answer["id"])
        assert (str(task.workspace_id), task.created_by_type, task.created_by_id) == (ws, "api_key", ctx.user.id)
        assert task.status == "assigned"
    finally:
        s.close()


def test_a_server_key_is_not_the_operator_so_its_ticket_gets_no_operator_consent(new_session, workspace):
    """On the local edition, filing and assigning a ticket is the operator's consent (PRD-234
    D16). A key's ticket meets the workspace's approval policy instead."""
    from api.board_task_create import file_ticket, new_ticket_columns
    from services.board_consent import creation_is_consent

    ws, agent_id = workspace
    assert creation_is_consent()  # the test edition is local
    key_ctx = _pass(_request(_mint(new_session, ws, permissions=(TASKS_WRITE,))["key"]), "missions:create", TASKS_WRITE)
    operator = SimpleNamespace(workspace_id=uuid.UUID(ws), auth_type="clerk",
                               user=SimpleNamespace(id="u-1", clerk_user_id="clerk_owner", email=None))
    filed = {}
    for name, ctx in (("key", key_ctx), ("operator", operator)):
        s = new_session()
        try:
            columns = new_ticket_columns(s, ws, {"title": f"Renew Acme ({name})", "assigned_agent_id": agent_id})
            filed[name] = file_ticket(s, ctx, columns).id
        finally:
            s.close()
    assert _grants(new_session, filed["key"]) == 0 and _grants(new_session, filed["operator"]) == 1


def test_the_named_routes_carry_the_gate_and_tasks_write_can_be_minted():
    from fastapi.routing import APIRoute, iter_route_contexts

    from api.api_keys import VALID_PERMISSIONS
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR
    from main import app

    routes = {(m, r.path): r for r in iter_route_contexts(app.routes)
              if isinstance(r.original_route, APIRoute) for m in r.methods}
    for key, permission, scope in ((("POST", "/api/agents/{agent_id}/execute"), "agents:execute", AGENTS_EXECUTE),
                                   (("POST", "/api/v1/tasks"), "missions:create", TASKS_WRITE)):
        calls = [d.call for d in routes[key].dependant.dependencies]
        marked = [(getattr(c, PERMISSION_MARKER_ATTR, None), getattr(c, gate.KEY_SCOPE_MARKER_ATTR, None))
                  for c in calls if getattr(c, gate.KEY_SCOPE_MARKER_ATTR, None)]
        assert marked == [(permission, scope)], key
    assert TASKS_WRITE in VALID_PERMISSIONS
