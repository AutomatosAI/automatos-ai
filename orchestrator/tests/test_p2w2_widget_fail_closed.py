"""PRD-194 S3 (P2-13, security §1.2.a F053 / §1.2.d F042) — widget auth fails closed.

The storefront widget is the only surface exposed to arbitrary browsers, and
its auth failed OPEN in two directions:

1. **Origin-absent bypass** — an ``ak_pub_`` key (ships in page HTML by
   design; its entire security model is the origin lock) validated from any
   client that simply omitted ``Origin``/``Referer``. Now: public key with
   no origin ⇒ **403**. Server keys are not origin-locked and are
   unaffected; ``check_domain``'s merchant opt-in (empty ``allowed_domains``
   = any origin) is unchanged.
2. **Empty-permission god-key** — with the policy plane OFF (the default),
   ``require_permission`` treated an empty permission list as unrestricted.
   Now: **empty = deny regardless of the policy flag** — the internet-facing
   plane is never the permissive one (P2-11 owns the platform-wide fix).

Rollout note (locked decision): audit-first — Gerard audits live ``ak_pub_``
usage for header-less callers before this deploys; an ``ak_srv_`` server key
is the sanctioned path for any legitimate non-browser caller found.

Pure: hand-built Starlette Requests, ``ApiKeyService.validate_api_key``
stubbed at the class boundary, no DB, no network.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio  # noqa: E402
import sys  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from uuid import uuid4  # noqa: E402

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from starlette.requests import Request  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import api.widgets.auth as widget_auth_mod  # noqa: E402
from api.widgets.auth import WidgetAuthContext, require_permission, widget_auth  # noqa: E402


# ---------------------------------------------------------------- helpers

def _make_request(headers: dict) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/widgets/chat",
        "headers": [(k.lower().encode(), str(v).encode()) for k, v in headers.items()],
        "query_string": b"",
    }

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    return Request(scope, receive)


def _run(coro):
    return asyncio.run(coro)


def _key_record(
    key_type="public",
    allowed_domains=None,
    permissions=None,
):
    return SimpleNamespace(
        id=uuid4(),
        workspace_id=uuid4(),
        key_type=key_type,
        key_prefix="ak_pub_a1b2" if key_type == "public" else "ak_srv_a1b2",
        allowed_domains=allowed_domains,
        permissions=permissions or [],
        default_agent_id=None,
        team=None,
    )


def _auth(monkeypatch, record, headers):
    """Run widget_auth with validate_api_key stubbed to return *record*."""
    monkeypatch.setattr(
        widget_auth_mod.ApiKeyService, "validate_api_key", lambda db, token: record
    )
    base = {"Authorization": "Bearer ak_pub_testtoken"}
    base.update(headers)
    return _run(widget_auth(request=_make_request(base), db=object()))


# ---------------------------------------------------------------- S3a: origin

def test_public_key_denied_when_origin_absent(monkeypatch):
    """An ak_pub_ key with no Origin/Referer header ⇒ 403 (was: validated)."""
    record = _key_record(key_type="public", allowed_domains=["good.com"])
    with pytest.raises(HTTPException) as ei:
        _auth(monkeypatch, record, {})
    assert ei.value.status_code == 403
    assert "Origin required" in ei.value.detail


def test_public_key_still_allowed_from_listed_origin(monkeypatch):
    """No regression for the real browser path: listed origin still passes."""
    record = _key_record(key_type="public", allowed_domains=["good.com"])
    ctx = _auth(monkeypatch, record, {"Origin": "https://good.com"})
    assert isinstance(ctx, WidgetAuthContext)
    assert ctx.workspace_id == record.workspace_id
    assert ctx.api_key_id == record.id


def test_public_key_from_unlisted_origin_denied(monkeypatch):
    """Existing deny direction unchanged: wrong origin is still a 403."""
    record = _key_record(key_type="public", allowed_domains=["good.com"])
    with pytest.raises(HTTPException) as ei:
        _auth(monkeypatch, record, {"Origin": "https://evil.com"})
    assert ei.value.status_code == 403


def test_public_key_empty_domains_with_origin_allowed(monkeypatch):
    """check_domain's merchant opt-in is NOT weakened: empty allowed_domains
    still admits any *present* origin — the new deny is origin-ABSENT only."""
    record = _key_record(key_type="public", allowed_domains=[])
    ctx = _auth(monkeypatch, record, {"Origin": "https://anywhere.example"})
    assert ctx.workspace_id == record.workspace_id


def test_server_key_unaffected_by_missing_origin(monkeypatch):
    """ak_srv_ keys are not origin-locked — no origin is fine for them."""
    record = _key_record(key_type="server", allowed_domains=["good.com"])
    ctx = _auth(monkeypatch, record, {})
    assert ctx.workspace_id == record.workspace_id


def test_unknown_key_type_treated_as_public(monkeypatch):
    """A null/legacy key_type fails CLOSED (treated as origin-locked public)."""
    record = _key_record(key_type=None)
    with pytest.raises(HTTPException) as ei:
        _auth(monkeypatch, record, {})
    assert ei.value.status_code == 403


def test_referer_counts_as_origin(monkeypatch):
    """_extract_origin accepts Referer — a real browser path that sends only
    Referer must keep working for public keys."""
    record = _key_record(key_type="public", allowed_domains=["good.com"])
    ctx = _auth(monkeypatch, record, {"Referer": "https://good.com/products/x"})
    assert ctx.workspace_id == record.workspace_id


# ---------------------------------------------------------------- S3b: perms

def _ctx(permissions):
    return WidgetAuthContext(
        workspace_id=uuid4(), api_key_id=uuid4(), permissions=permissions
    )


def test_widget_empty_permission_denied():
    """A widget key with permissions=[] is DENIED a permissioned route —
    with the policy plane in its default (OFF) state (was: allow-all)."""
    check = require_permission("chat")
    with pytest.raises(HTTPException) as ei:
        _run(check(auth=_ctx([])))
    assert ei.value.status_code == 403


def test_widget_granted_permission_allowed():
    check = require_permission("chat")
    ctx = _run(check(auth=_ctx(["chat"])))
    assert isinstance(ctx, WidgetAuthContext)


def test_widget_missing_permission_denied():
    check = require_permission("data:execute")
    with pytest.raises(HTTPException) as ei:
        _run(check(auth=_ctx(["chat"])))
    assert ei.value.status_code == 403


def test_widget_explicit_wildcard_grants():
    """An explicit '*' element is a deliberate grant (board-plane semantics),
    not an omission — honoured on the widget plane too."""
    check = require_permission("chat")
    ctx = _run(check(auth=_ctx(["*"])))
    assert isinstance(ctx, WidgetAuthContext)


def test_empty_permission_denied_even_without_policy_module(monkeypatch):
    """The deny does not depend on the policy package importing — empty is
    denied by the local fallback too (deny regardless of the flag/plane)."""
    monkeypatch.setitem(sys.modules, "modules.policy.roles", None)
    check = require_permission("chat")
    with pytest.raises(HTTPException) as ei:
        _run(check(auth=_ctx([])))
    assert ei.value.status_code == 403
    # and explicit grants still work through the fallback
    ctx = _run(check(auth=_ctx(["chat"])))
    assert isinstance(ctx, WidgetAuthContext)
