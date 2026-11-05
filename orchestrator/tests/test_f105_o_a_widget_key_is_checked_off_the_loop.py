"""F105 (wave 2): a widget request's key is looked up off the event loop.

``api/widgets/auth.widget_auth`` backs every widget and SDK route. It stays async (it awaits
the key's rate-limit count, ``note_key_limit``), but it read the database on the loop: a
session token's stored limit (``stored_key_limit``), and for a raw key its validation
(``ApiKeyService.validate_api_key``, which commits the key's last use) and every check on
the row after it (its type, prefix, domains, workspace and permissions; after the commit,
the first of those reads loads the row again). Those now run in FastAPI's threadpool; the
count stays on the loop. Who is let in, every refusal and the context are unchanged.

DB-free: the lookups and every read of the key's row record the thread they run on.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace as NS
from uuid import uuid4

import jwt
import pytest
from fastapi import HTTPException
from starlette.requests import Request

import api.widgets.auth as auth
import api.widgets.rate_limit as rate_limit

SECRET = "f105-widget-session-secret"
ORIGIN = "https://shop.example.com"


class _Row:
    """An ``sdk_api_keys`` row whose every field read records its thread (a lazy load would be one)."""

    def __init__(self, reads, **fields):
        object.__setattr__(self, "_reads", reads)
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name):
        fields = object.__getattribute__(self, "_fields")
        if name not in fields:
            raise AttributeError(name)
        object.__getattribute__(self, "_reads").append(threading.get_ident())
        return fields[name]


def _row(reads, **overrides):
    fields = dict(id=uuid4(), workspace_id=uuid4(), key_type="public", key_prefix="ak_pub_f105",
                  allowed_domains=["shop.example.com"], permissions=["chat"], default_agent_id=7, team="hq",
                  rate_limit_requests=30)
    return _Row(reads, **{**fields, **overrides})


def _request(headers):
    scope = {"type": "http", "method": "POST", "path": "/api/widgets/chat", "query_string": b"",
             "headers": [(k.lower().encode(), str(v).encode()) for k, v in headers.items()]}
    return Request(scope)


@pytest.fixture
def seen(monkeypatch):
    seen = NS(threads={}, counted=[], reads=[])

    async def count(api_key_id, stored_limit):
        seen.counted.append((api_key_id, stored_limit, threading.get_ident()))

    monkeypatch.setattr(rate_limit, "note_key_limit", count)
    return seen


def _auth(headers):
    async def run():
        return threading.get_ident(), await auth.widget_auth(_request(headers), db="request-session")

    return asyncio.run(run())


def _key(seen, monkeypatch, row):
    def validate(db, token):
        seen.threads["key validated"] = threading.get_ident()
        return row

    monkeypatch.setattr(auth.ApiKeyService, "validate_api_key", validate)


def test_a_raw_key_is_validated_and_checked_off_the_loop(seen, monkeypatch):
    row = _row(seen.reads)
    _key(seen, monkeypatch, row)

    loop, ctx = _auth({"Authorization": "Bearer ak_pub_f105token", "Origin": ORIGIN,
                       "X-Workspace-ID": str(row._fields["workspace_id"])})

    fields = row._fields
    assert (ctx.workspace_id, ctx.api_key_id, ctx.permissions, ctx.default_agent_id, ctx.team) == (
        fields["workspace_id"], fields["id"], ["chat"], 7, "hq")
    assert seen.threads["key validated"] != loop
    assert seen.reads and loop not in seen.reads  # no read of the key's row on the loop
    assert [(key_id, limit) for key_id, limit, _ in seen.counted] == [(fields["id"], 30)]
    assert seen.counted[0][2] == loop  # the count stays on the loop


@pytest.mark.parametrize("row_fields,headers,status,detail", [
    ({}, {"Origin": "https://elsewhere.example.org"}, 403, "Origin not allowed for this API key"),
    ({}, {}, 403, "Origin required for public API keys"),
    ({}, {"Origin": ORIGIN, "X-Workspace-ID": "6a0f3c1e-2b4d-4e8f-9a1b-f10500a0c0ff"}, 403,
     "Workspace mismatch between API key and header"),
])
def test_a_raw_keys_refusals_are_unchanged(seen, monkeypatch, row_fields, headers, status, detail):
    _key(seen, monkeypatch, _row(seen.reads, **row_fields))

    with pytest.raises(HTTPException) as refused:
        _auth({"Authorization": "Bearer ak_pub_f105token", **headers})

    assert (refused.value.status_code, refused.value.detail) == (status, detail)
    assert seen.counted == []


def test_an_unknown_key_is_still_refused(seen, monkeypatch):
    _key(seen, monkeypatch, None)

    with pytest.raises(HTTPException) as refused:
        _auth({"Authorization": "Bearer ak_pub_nottheone", "Origin": ORIGIN})

    assert (refused.value.status_code, refused.value.detail) == (401, "Invalid or expired API key")


def test_a_session_tokens_stored_limit_is_read_off_the_loop(seen, monkeypatch):
    monkeypatch.setattr(auth, "WIDGET_TOKEN_SECRET", SECRET)
    workspace, key = uuid4(), uuid4()

    def stored(db, api_key_id):
        seen.threads["limit read"] = threading.get_ident()
        return 12

    monkeypatch.setattr(rate_limit, "stored_key_limit", stored)
    token = jwt.encode({"workspace_id": str(workspace), "api_key_id": str(key), "permissions": ["chat"],
                        "default_agent_id": "5", "team": "hq"}, SECRET, algorithm="HS256")

    loop, ctx = _auth({"Authorization": f"Bearer {token}"})

    assert (ctx.workspace_id, ctx.api_key_id, ctx.permissions, ctx.default_agent_id, ctx.team) == (
        workspace, key, ["chat"], 5, "hq")
    assert seen.threads["limit read"] != loop
    assert [(key_id, limit) for key_id, limit, _ in seen.counted] == [(key, 12)]


def test_a_malformed_session_token_is_still_refused(seen, monkeypatch):
    monkeypatch.setattr(auth, "WIDGET_TOKEN_SECRET", SECRET)
    token = jwt.encode({"workspace_id": "not-a-uuid", "api_key_id": str(uuid4())}, SECRET, algorithm="HS256")

    with pytest.raises(HTTPException) as refused:
        _auth({"Authorization": f"Bearer {token}"})

    assert refused.value.status_code == 401
    assert refused.value.detail.startswith("Malformed widget session token:")
