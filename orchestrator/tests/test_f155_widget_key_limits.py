"""F155 (g) — a widget key's budget and its stored limit.

A session token (a signed JWT, minted only from a server key) was bucketed by
its own digest, so each new session started a fresh window and a key's
sessions shared no budget. A session now keeps its own window and also counts
against its key's bucket at the server limit. A key's stored per-minute limit
(sdk_api_keys.rate_limit_requests) had no reader: requests are now counted per
key, raw key and sessions together, and a request over it is logged once a
window, never refused (log-only).
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from types import SimpleNamespace as NS

import jwt
import pytest

import api.widgets.rate_limit as rl
from config import config
from tests.test_p2w2_widget_rate_limit import _App, _FakeRedis, _call, _status

SECRET = "f155-widget-session-secret"


@pytest.fixture
def signed(monkeypatch):
    import api.widgets.auth as widget_auth

    monkeypatch.setattr(widget_auth, "WIDGET_TOKEN_SECRET", SECRET)


def _session(key_id, workspace_id=None):
    return jwt.encode({"workspace_id": str(workspace_id or uuid.uuid4()), "api_key_id": str(key_id),
                       "permissions": ["chat"], "sid": uuid.uuid4().hex}, SECRET, algorithm="HS256")


def _statuses(tokens):
    fake = _FakeRedis()
    limiter = rl.WidgetRateLimitMiddleware(_App(), store=rl.RateLimitStore(redis_factory=lambda: fake))
    return [_status(asyncio.run(_call(limiter, "/api/widgets/docs/search", {"Authorization": f"Bearer {token}"})))
            for token in tokens]


def test_a_keys_sessions_share_its_budget(signed, monkeypatch):
    monkeypatch.setattr(config, "WIDGET_RATE_LIMIT_SERVER_PER_WINDOW", 2)
    monkeypatch.setattr(config, "WIDGET_RATE_LIMIT_PUBLIC_PER_WINDOW", 100)
    key, other = uuid.uuid4(), uuid.uuid4()
    assert _statuses([_session(key), _session(key), _session(key), _session(other)]) == [200, 200, 429, 200]


def test_each_session_keeps_its_own_window(signed, monkeypatch):
    monkeypatch.setattr(config, "WIDGET_RATE_LIMIT_SERVER_PER_WINDOW", 100)
    monkeypatch.setattr(config, "WIDGET_RATE_LIMIT_PUBLIC_PER_WINDOW", 1)
    token = _session(uuid.uuid4())
    assert _statuses([token, token]) == [200, 429]


def test_a_key_over_its_stored_limit_is_logged_and_still_served(signed, db_session, seed_workspace, monkeypatch,
                                                                 caplog):
    from api.widgets.auth import widget_auth
    from core.services.api_key_service import ApiKeyService

    fake = _FakeRedis()
    monkeypatch.setattr(rl, "_census_store", rl.RateLimitStore(redis_factory=lambda: fake), raising=False)
    monkeypatch.setattr(rl, "_census_logged", {}, raising=False)
    monkeypatch.setattr(rl, "_stored_limits", {}, raising=False)
    ws = uuid.UUID(seed_workspace())
    key = ApiKeyService.create_api_key(db_session, workspace_id=ws, name="shop backend", key_type="server",
                                       permissions=["chat"], rate_limit_requests=2)
    bearers = [key["key"], key["key"], _session(key["id"], ws), _session(key["id"], ws)]
    with caplog.at_level(logging.WARNING, logger="api.widgets.rate_limit"):
        served = [asyncio.run(widget_auth(NS(headers={"Authorization": f"Bearer {bearer}"}, state=NS()), db_session))
                  for bearer in bearers]
    assert {str(ctx.api_key_id) for ctx in served} == {key["id"]}
    over = [record for record in caplog.records if "over its limit of 2 requests a minute" in record.getMessage()]
    assert len(over) == 1
