"""PRD-274 Wave 1 (§5.2): on the routes a server key reaches, the per-IP limit gives way to the key's.

With the policy plane on, slowapi limits every route per client IP. The routes a server key reaches
are exempted from it and limit inside their gate instead: a server key that authenticates meets its
own limits, never the IP's; anyone else meets the same per-IP limit for that route; and keys that
fail are counted per IP, so past the limit the next one is refused before it is looked up. With
the plane off, nothing here runs. Real keys on the real database; the window is a fake.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import text
from starlette.requests import Request

from core.auth import server_key_gate as gate
from core.auth import server_key_ip_limit as ip_limit
from core.auth.scopes import AGENTS_EXECUTE


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
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w1-ip-limit')"), {"id": ws})
    s.commit()
    s.close()
    yield ws
    s = new_session.sweep()
    s.execute(text("DELETE FROM sdk_api_keys WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


class _Window:
    """One shared fake window: check() counts and answers, recent() only reads."""

    def __init__(self):
        self.seen = {}

    def check(self, key_id, limit, window=None, *, quiet=False, label=""):
        count = self.seen.get(key_id, 0)
        self.seen[key_id] = count + 1
        return (count < limit, limit, max(0, limit - count - 1), 30)

    def recent(self, key_id, window=None):
        return self.seen.get(key_id, 0)


@pytest.fixture
def window(monkeypatch):
    shared = _Window()
    monkeypatch.setattr("api.widgets.rate_limit.RateLimitStore", lambda: shared)
    return shared


@pytest.fixture
def plane(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "POLICY_PLANE_ENABLED", True)
    monkeypatch.setattr(config, "APP_RATE_LIMIT_PER_MINUTE", 2)
    monkeypatch.setattr(config, "SERVER_KEY_REQUESTS_PER_MINUTE", 5)


def _request(token=None, ip="203.0.113.7"):
    headers = [(b"x-forwarded-for", ip.encode())]
    if token:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    return Request({"type": "http", "method": "POST", "path": "/api/agents/1/execute", "headers": headers,
                    "query_string": b"", "client": (ip, 1234)})


def _gate():
    return gate.require_workspace_permission_or_server_key("agents:execute", AGENTS_EXECUTE)


def _server_key(new_session, ws):
    from core.services.api_key_service import ApiKeyService

    s = new_session()
    try:
        return ApiKeyService.create_api_key(s, uuid.UUID(ws), "acme backend", key_type="server",
                                            permissions=[AGENTS_EXECUTE])["key"]
    finally:
        s.close()


def _status(call):
    try:
        call()
    except HTTPException as refused:
        return refused.status_code
    return 200


def test_a_server_key_meets_its_own_limit_not_the_ips(new_session, workspace, window, plane):
    token = _server_key(new_session, workspace)
    answers = [_status(lambda: asyncio.run(_gate()(_request(token), db=None))) for _ in range(6)]
    assert answers == [200, 200, 200, 200, 200, 429]  # five a minute for the key; the IP's two never applied


def test_anyone_else_meets_the_per_ip_limit_for_the_route(window, plane, monkeypatch):
    async def signed_in(request):
        raise HTTPException(status_code=401, detail="no session")  # the limit runs before this

    monkeypatch.setattr(gate, "get_request_context_hybrid", signed_in)
    answers = [_status(lambda: asyncio.run(_gate()(_request(), db=None))) for _ in range(3)]
    assert answers == [401, 401, 429]
    assert _status(lambda: asyncio.run(_gate()(_request(ip="198.51.100.9"), db=None))) == 401  # another address


def test_failed_keys_are_counted_per_address_and_then_refused_before_any_lookup(window, plane, monkeypatch):
    looked_up = []
    real = gate._sdk_key_context_in_own_session

    def counting(request, scope):
        looked_up.append(1)
        return real(request, scope)

    monkeypatch.setattr(gate, "_sdk_key_context_in_own_session", counting)
    made_up = "ak_srv_" + "0" * 40
    answers = [_status(lambda: asyncio.run(_gate()(_request(made_up), db=None))) for _ in range(4)]
    assert answers == [401, 401, 429, 429] and len(looked_up) == 2


def test_with_the_plane_off_nothing_here_runs(window, monkeypatch):
    from config import config

    monkeypatch.setattr(config, "POLICY_PLANE_ENABLED", False)
    ip_limit.limit_this_caller(_request())
    ip_limit.note_failed_key(_request())
    ip_limit.refuse_after_failed_keys(_request())
    assert window.seen == {}


def test_slowapi_exempts_exactly_the_routes_a_server_key_reaches():
    from main import limiter

    assert {"api.agent_execute.execute_agent", "api.board_tasks.create_task"} <= limiter._exempt_routes
    assert "api.board_tasks.list_tasks" not in limiter._exempt_routes
    assert "api.agents.list_agents" not in limiter._exempt_routes
