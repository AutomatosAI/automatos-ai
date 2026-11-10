"""PRD-274 Wave 1 (§5.2): a product's server key reads and stores the workspace's memory.

``GET /api/memory`` and ``GET /api/memory/search`` take a server key listing ``memory:read``;
``POST /api/memory`` takes one listing ``memory:write``. A key reads only facts no audience limits
(need-to-know) and none private to a person; a signed-in member reads as before. What a key stores
carries ``written_by: sdk:<key id>``, which no caller can set. Deleting and resuming stay with
people. Real keys on the real database; the memory store is a fake that records its calls.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import text
from starlette.requests import Request

from api import widget_memory
from core.auth import server_key_gate as gate
from core.auth.scopes import MEMORY_READ, MEMORY_WRITE

ROWS = [
    {"id": "plain", "memory": "Acme renews in March.", "metadata": {"scope": "workspace"}},
    {"id": "legacy", "memory": "Acme's account manager is Priya.", "metadata": {}},
    {"id": "limited", "memory": "Acme may be acquired.", "metadata": {"audience": ["finance"]}},
    {"id": "private", "memory": "Priya prefers email.", "metadata": {"scope": "private", "owner": "user:42"}},
]


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
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w1-memory-keys')"), {"id": ws})
    s.commit()
    s.close()
    yield ws
    s = new_session.sweep()
    s.execute(text("DELETE FROM sdk_api_keys WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


class _Store:
    def __init__(self):
        self.stored = []

    async def get_all_memories(self, workspace_id, limit):
        return [dict(r) for r in ROWS]

    async def search_long_term(self, workspace_id, query, limit):
        return [dict(r, score=0.9) for r in ROWS]

    async def store_long_term(self, workspace_id, content, metadata=None):
        self.stored.append(metadata)
        return {"success": True, "id": "m-1"}


@pytest.fixture
def store(monkeypatch):
    fake = _Store()
    monkeypatch.setattr(widget_memory, "_get_memory_service", lambda: fake)
    return fake


@pytest.fixture
def unlimited(monkeypatch):
    class _Open:
        def check(self, *args, **kwargs):
            return (True, 1, 0, 60)

    monkeypatch.setattr("api.widgets.rate_limit.RateLimitStore", _Open)


def _key(new_session, ws, permissions, key_type="server"):
    from core.services.api_key_service import ApiKeyService

    s = new_session()
    try:
        return ApiKeyService.create_api_key(s, uuid.UUID(ws), "acme backend", key_type=key_type,
                                            permissions=list(permissions))["key"]
    finally:
        s.close()


def _request(token):
    pairs = [(b"authorization", f"Bearer {token}".encode())]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": pairs, "query_string": b""})


def _key_ctx(new_session, ws, permissions, scope=MEMORY_READ):
    return asyncio.run(gate.request_context_or_server_key(scope)(_request(_key(new_session, ws, permissions))))


def _member(ws):
    return SimpleNamespace(workspace_id=uuid.UUID(ws), auth_type="clerk",
                           user=SimpleNamespace(id="user_1", clerk_user_id="user_1", email=None))


def test_a_key_lists_and_searches_only_what_no_audience_limits_and_nothing_private(new_session, workspace,
                                                                                   store, unlimited):
    ctx = _key_ctx(new_session, workspace, [MEMORY_READ])
    listed = asyncio.run(widget_memory.list_memories(ctx=ctx, limit=50))
    found = asyncio.run(widget_memory.search_memories(q="Acme", ctx=ctx, limit=20))
    assert [m.id for m in listed.memories] == ["plain", "legacy"] and listed.total == 2
    assert [m.id for m in found.results] == ["plain", "legacy"]
    assert str(ctx.workspace_id) == workspace


def test_a_member_reads_as_before(workspace, store):
    listed = asyncio.run(widget_memory.list_memories(ctx=_member(workspace), limit=50))
    assert [m.id for m in listed.memories] == ["plain", "legacy", "limited", "private"]


def test_what_a_key_stores_says_which_key_and_no_caller_can_say_so(new_session, workspace, store, unlimited):
    key_ctx = asyncio.run(gate.require_workspace_permission_or_server_key("knowledge:create", MEMORY_WRITE)(
        _request(_key(new_session, workspace, [MEMORY_WRITE])), db=None))
    body = widget_memory.MemoryCreate(content="Acme renews in March.",
                                      metadata={"written_by": "sdk:someone-else", "source": "crm"})
    asyncio.run(widget_memory.store_memory(body=body, ctx=key_ctx))
    asyncio.run(widget_memory.store_memory(body=body, ctx=_member(workspace)))
    by_key, by_member = store.stored
    assert by_key["written_by"] == key_ctx.user.id and key_ctx.user.id.startswith("sdk:")
    assert by_key["source"] == "crm" and "written_by" not in by_member


@pytest.mark.parametrize("permissions, key_type", [([MEMORY_WRITE], "server"), ([], "server"),
                                                   ([MEMORY_READ], "public")])
def test_reading_needs_a_server_key_with_memory_read(new_session, workspace, unlimited, permissions, key_type):
    token = _key(new_session, workspace, permissions, key_type=key_type)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(gate.request_context_or_server_key(MEMORY_READ)(_request(token)))
    assert refused.value.status_code == 403


def test_the_memory_routes_carry_the_scopes_and_delete_and_resume_take_no_key():
    from fastapi.routing import APIRoute, iter_route_contexts

    from api.api_keys import VALID_PERMISSIONS
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR
    from main import app

    routes = {(m, r.path): r for r in iter_route_contexts(app.routes)
              if isinstance(r.original_route, APIRoute) for m in r.methods}

    def scopes(key):
        calls = [d.call for d in routes[key].dependant.dependencies]
        return [(getattr(c, gate.KEY_SCOPE_MARKER_ATTR), getattr(c, PERMISSION_MARKER_ATTR, None))
                for c in calls if getattr(c, gate.KEY_SCOPE_MARKER_ATTR, None)]

    assert scopes(("GET", "/api/memory")) == [(MEMORY_READ, None)]
    assert scopes(("GET", "/api/memory/search")) == [(MEMORY_READ, None)]
    assert scopes(("POST", "/api/memory")) == [(MEMORY_WRITE, "knowledge:create")]
    assert scopes(("DELETE", "/api/memory/{memory_id}")) == [] and scopes(("GET", "/api/memory/resume")) == []
    assert {MEMORY_READ, MEMORY_WRITE} <= set(VALID_PERMISSIONS)
