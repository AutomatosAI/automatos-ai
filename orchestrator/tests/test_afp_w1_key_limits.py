"""PRD-274 Wave 1 (§5.2): a server key's per-key limits are enforced on the routes named for it.

Requests a minute: the key's own ``rate_limit_requests`` under the platform's ceiling
(``SERVER_KEY_REQUESTS_PER_MINUTE``), which a key's own limit can only lower. Tokens a minute: the
key's ``rate_limit_tokens`` against what its runs used in the last minute, from the usage ledger.
Over either is a 429 with ``Retry-After``. Real database, real keys.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import text
from starlette.requests import Request

from core.auth import server_key_gate as gate
from core.auth.scopes import AGENTS_EXECUTE
from services import key_limits


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
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w1-key-limits')"), {"id": ws})
    s.commit()
    s.close()
    yield ws
    s = new_session.sweep()
    for table in ("llm_usage", "board_tasks", "sdk_api_keys"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


@pytest.fixture
def ceiling(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "SERVER_KEY_REQUESTS_PER_MINUTE", 100)


class _Window:
    """A request window per bucket, as RateLimitStore answers: (allowed, limit, remaining, reset)."""

    def __init__(self):
        self.seen = {}

    def check(self, key_id, limit, window=None, *, quiet=False, label=""):
        count = self.seen.get(key_id, 0)
        self.seen[key_id] = count + 1
        return (count < limit, limit, max(0, limit - count - 1), 42)


def _mint(new_session, ws, **limits):
    from core.services.api_key_service import ApiKeyService

    s = new_session()
    try:
        return ApiKeyService.create_api_key(s, uuid.UUID(ws), "acme backend", key_type="server",
                                            permissions=[AGENTS_EXECUTE], **limits)
    finally:
        s.close()


def _ctx(new_session, ws, **limits):
    key = _mint(new_session, ws, **limits)
    pairs = [(b"authorization", f"Bearer {key['key']}".encode())]
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": pairs, "query_string": b""})
    from core.auth.hybrid import _sdk_key_context_in_own_session

    return _sdk_key_context_in_own_session(request, AGENTS_EXECUTE), key


def _ticket_with_usage(new_session, ws, filed_by, tokens, seconds_ago=0):
    s = new_session()
    task_id = s.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, status, created_by_type, created_by_id) "
        "VALUES (CAST(:ws AS uuid), 'run', 'done', 'api_key', :by) RETURNING id"), {"ws": ws, "by": filed_by}).scalar()
    s.execute(text(
        "INSERT INTO llm_usage (workspace_id, model_id, provider, tier, execution_id, input_tokens, output_tokens, "
        "total_tokens, input_cost, output_cost, total_cost, created_at) VALUES (CAST(:ws AS uuid), 'm', 'p', 'direct', "
        ":ex, 0, :t, :t, 0, 0, 0, now() - make_interval(secs => :ago))"),
        {"ws": ws, "ex": f"board_task:{task_id}", "t": tokens, "ago": seconds_ago})
    s.commit()
    s.close()


def _enforce(new_session, ctx, store):
    s = new_session()
    try:
        key_limits.enforce_key_limits(s, ctx, store=store)
    finally:
        s.close()


@pytest.mark.parametrize("own, expected", [(None, 100), (20, 20), (500, 100), (0, 100)])
def test_a_keys_own_request_limit_can_only_lower_the_ceiling(new_session, workspace, ceiling, own, expected):
    ctx, _ = _ctx(new_session, workspace, rate_limit_requests=own)
    s = new_session()
    assert key_limits.limits_for(s, ctx.api_key_id).requests == expected
    s.close()


def test_over_its_requests_a_minute_a_key_is_told_when_to_retry(new_session, workspace, ceiling):
    ctx, _ = _ctx(new_session, workspace, rate_limit_requests=2)
    window = _Window()
    _enforce(new_session, ctx, window)
    _enforce(new_session, ctx, window)
    with pytest.raises(HTTPException) as over:
        _enforce(new_session, ctx, window)
    assert over.value.status_code == 429 and over.value.headers["Retry-After"] == "42"
    assert "2 requests a minute" in over.value.detail


def test_over_its_tokens_a_minute_a_key_starts_no_new_work(new_session, workspace, ceiling):
    ctx, _ = _ctx(new_session, workspace, rate_limit_tokens=1000)
    other, _ = _ctx(new_session, workspace, rate_limit_tokens=1000)
    _ticket_with_usage(new_session, workspace, ctx.user.id, 600)
    _ticket_with_usage(new_session, workspace, ctx.user.id, 5000, seconds_ago=120)  # a minute ago and more
    _ticket_with_usage(new_session, workspace, other.user.id, 5000)                 # another key's run
    _enforce(new_session, ctx, _Window())
    _ticket_with_usage(new_session, workspace, ctx.user.id, 400)
    with pytest.raises(HTTPException) as over:
        _enforce(new_session, ctx, _Window())
    assert over.value.status_code == 429 and "1000 tokens a minute" in over.value.detail
    assert over.value.headers["Retry-After"] == str(key_limits.WINDOW_SECONDS)


def test_a_key_with_no_token_limit_is_not_counted(new_session, workspace, ceiling):
    ctx, _ = _ctx(new_session, workspace)
    _ticket_with_usage(new_session, workspace, ctx.user.id, 10**9)
    _enforce(new_session, ctx, _Window())


def test_the_gate_enforces_the_limits_on_a_server_key(new_session, workspace, ceiling, monkeypatch):
    _, key = _ctx(new_session, workspace, rate_limit_requests=1)
    window = _Window()
    monkeypatch.setattr("api.widgets.rate_limit.RateLimitStore", lambda: window)
    pairs = [(b"authorization", f"Bearer {key['key']}".encode())]
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": pairs, "query_string": b""})
    dep = gate.require_workspace_permission_or_server_key("agents:execute", AGENTS_EXECUTE)
    assert asyncio.run(dep(request, db=None)).api_key_id == key["id"]
    with pytest.raises(HTTPException) as over:
        asyncio.run(dep(request, db=None))
    assert over.value.status_code == 429
