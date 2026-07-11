"""Tests for the board SDK-key auth unlock (PRD-09 Slice 2, Step 0).

Pure-function / monkeypatched style (no live DB), mirroring
``test_api_key_domain_check.py``. Covers three layers:

1. ``_sdk_key_has_scope`` — the least-privilege scope gate (the security crux:
   empty/None permissions must be DENIED — the rule the whole platform now
   shares: PRD-195 S1 collapsed ``ApiKeyService.check_permissions`` and the
   widget plane onto the same empty=deny semantic).
2. ``_resolve_sdk_key_context`` — the SDK-key → RequestContext resolver, incl.
   the cross-tenant guard (binds to the key's workspace, never env defaults).
3. ``require_task_context`` — the dependency factory: routes ``ak_*`` bearers to
   the SDK path and delegates everything else (Clerk / env key / anon / OPTIONS)
   to the untouched ``get_request_context_hybrid``.
"""

import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from core.auth import hybrid
from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import (
    _extract_origin,
    _resolve_sdk_key_context,
    _sdk_key_has_scope,
    require_task_context,
)
from core.auth.scopes import TASKS_READ

WS = UUID("11111111-1111-1111-1111-111111111111")
OTHER_WS = UUID("22222222-2222-2222-2222-222222222222")


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #

class _Headers:
    """Case-insensitive header mapping, like Starlette's ``Headers``."""

    def __init__(self, data: dict):
        self._data = {k.lower(): v for k, v in (data or {}).items()}

    def get(self, key, default=None):
        return self._data.get(key.lower(), default)


class _FakeRequest:
    def __init__(self, headers=None, method="GET"):
        self.headers = _Headers(headers)
        self.method = method
        self.query_params = _Headers({})
        self.state = SimpleNamespace()


def _key(*, workspace_id=WS, permissions, allowed_domains=None, key_id=None):
    return SimpleNamespace(
        id=key_id or uuid4(),
        workspace_id=workspace_id,
        permissions=permissions,
        allowed_domains=allowed_domains,
        key_prefix="ak_pub_test",
    )


def _patch_valid(monkeypatch, record):
    """Make validate_api_key return *record* and the workspace checks pass."""
    monkeypatch.setattr(hybrid.ApiKeyService, "validate_api_key", lambda db, token: record)
    monkeypatch.setattr(hybrid, "_workspace_exists", lambda db, ws: True)
    monkeypatch.setattr(hybrid, "_assert_workspace_usable", lambda db, ws, *, is_admin=False: None)


def _resolve(req):
    return _resolve_sdk_key_context(req, object(), required_scope=TASKS_READ)


# --------------------------------------------------------------------------- #
# 0. Shared origin extraction (used by both board + widget planes)
# --------------------------------------------------------------------------- #

class TestExtractOrigin:
    def test_origin_header_returns_hostname(self):
        assert _extract_origin(_FakeRequest({"Origin": "https://app.example.com"})) == "app.example.com"

    def test_referer_fallback_strips_path(self):
        assert _extract_origin(_FakeRequest({"Referer": "https://app.example.com/x/y"})) == "app.example.com"

    def test_none_when_no_origin_or_referer(self):
        assert _extract_origin(_FakeRequest({})) is None


# --------------------------------------------------------------------------- #
# 1. Scope gate — the least-privilege crux
# --------------------------------------------------------------------------- #

class TestSdkKeyHasScope:
    def test_none_permissions_denied(self):
        # SECURITY: empty/None must NOT mean "all" (the shared platform rule).
        assert _sdk_key_has_scope(None, TASKS_READ) is False

    def test_empty_permissions_denied(self):
        assert _sdk_key_has_scope([], TASKS_READ) is False

    def test_unrelated_scope_denied(self):
        # An existing chat/widget key cannot reach the board.
        assert _sdk_key_has_scope(["chat"], TASKS_READ) is False

    def test_exact_scope_granted(self):
        assert _sdk_key_has_scope(["tasks:read"], TASKS_READ) is True

    def test_scope_among_others_granted(self):
        assert _sdk_key_has_scope(["chat", "tasks:read"], TASKS_READ) is True


# --------------------------------------------------------------------------- #
# 2. Resolver
# --------------------------------------------------------------------------- #

class TestResolveSdkKeyContext:
    def test_valid_read_key_returns_sdk_context(self, monkeypatch):
        rec = _key(permissions=["tasks:read"])
        _patch_valid(monkeypatch, rec)
        ctx = _resolve(_FakeRequest({"Authorization": "Bearer ak_pub_abc"}))
        assert isinstance(ctx, RequestContext)
        assert ctx.auth_type == "sdk_key"
        assert ctx.workspace_id == WS
        assert ctx.api_key_id == str(rec.id)
        assert ctx.user.role == "service"
        assert ctx.user.id == f"sdk:{rec.id}"

    def test_missing_token_401(self, monkeypatch):
        # Defensive: resolver called with no bearer at all.
        with pytest.raises(HTTPException) as e:
            _resolve(_FakeRequest({}))
        assert e.value.status_code == 401

    def test_invalid_or_revoked_key_401(self, monkeypatch):
        monkeypatch.setattr(hybrid.ApiKeyService, "validate_api_key", lambda db, token: None)
        with pytest.raises(HTTPException) as e:
            _resolve(_FakeRequest({"Authorization": "Bearer ak_pub_bad"}))
        assert e.value.status_code == 401

    def test_missing_scope_403(self, monkeypatch):
        _patch_valid(monkeypatch, _key(permissions=["chat"]))
        with pytest.raises(HTTPException) as e:
            _resolve(_FakeRequest({"Authorization": "Bearer ak_pub_abc"}))
        assert e.value.status_code == 403

    def test_null_permissions_key_403(self, monkeypatch):
        # The common case: an existing publishable chat key with null perms.
        _patch_valid(monkeypatch, _key(permissions=None))
        with pytest.raises(HTTPException) as e:
            _resolve(_FakeRequest({"Authorization": "Bearer ak_pub_abc"}))
        assert e.value.status_code == 403

    def test_origin_not_allowed_403(self, monkeypatch):
        _patch_valid(monkeypatch, _key(permissions=["tasks:read"], allowed_domains=["good.com"]))
        with pytest.raises(HTTPException) as e:
            _resolve(_FakeRequest(
                {"Authorization": "Bearer ak_pub_abc", "Origin": "https://evil.com"}
            ))
        assert e.value.status_code == 403

    def test_origin_allowed_ok(self, monkeypatch):
        _patch_valid(monkeypatch, _key(permissions=["tasks:read"], allowed_domains=["good.com"]))
        ctx = _resolve(_FakeRequest(
            {"Authorization": "Bearer ak_pub_abc", "Origin": "https://good.com"}
        ))
        assert ctx.workspace_id == WS

    def test_no_origin_skips_domain_check(self, monkeypatch):
        # Desktop / server keys send no Origin -> domain gate must not block.
        _patch_valid(monkeypatch, _key(permissions=["tasks:read"], allowed_domains=["good.com"]))
        ctx = _resolve(_FakeRequest({"Authorization": "Bearer ak_pub_abc"}))
        assert ctx.workspace_id == WS

    def test_workspace_header_mismatch_403(self, monkeypatch):
        _patch_valid(monkeypatch, _key(permissions=["tasks:read"]))
        with pytest.raises(HTTPException) as e:
            _resolve(_FakeRequest({
                "Authorization": "Bearer ak_pub_abc",
                "X-Workspace-ID": str(OTHER_WS),
            }))
        assert e.value.status_code == 403

    def test_workspace_header_match_ok(self, monkeypatch):
        _patch_valid(monkeypatch, _key(permissions=["tasks:read"]))
        ctx = _resolve(_FakeRequest({
            "Authorization": "Bearer ak_pub_abc",
            "X-Workspace-ID": str(WS),
        }))
        assert ctx.workspace_id == WS

    def test_soft_deleted_workspace_400(self, monkeypatch):
        monkeypatch.setattr(hybrid.ApiKeyService, "validate_api_key",
                            lambda db, token: _key(permissions=["tasks:read"]))
        monkeypatch.setattr(hybrid, "_workspace_exists", lambda db, ws: False)
        monkeypatch.setattr(hybrid, "_assert_workspace_usable", lambda db, ws, *, is_admin=False: None)
        with pytest.raises(HTTPException) as e:
            _resolve(_FakeRequest({"Authorization": "Bearer ak_pub_abc"}))
        assert e.value.status_code == 400

    def test_paused_workspace_403(self, monkeypatch):
        monkeypatch.setattr(hybrid.ApiKeyService, "validate_api_key",
                            lambda db, token: _key(permissions=["tasks:read"]))
        monkeypatch.setattr(hybrid, "_workspace_exists", lambda db, ws: True)

        def _raise(db, ws, *, is_admin=False):
            raise HTTPException(status_code=403, detail="Workspace is disabled.")

        monkeypatch.setattr(hybrid, "_assert_workspace_usable", _raise)
        with pytest.raises(HTTPException) as e:
            _resolve(_FakeRequest({"Authorization": "Bearer ak_pub_abc"}))
        assert e.value.status_code == 403

    def test_env_default_workspace_is_ignored(self, monkeypatch):
        # CROSS-TENANT REGRESSION: a divergent DEFAULT_WORKSPACE_ID / WORKSPACE_ID
        # must NOT change the resolved workspace — binding is to the key only.
        _patch_valid(monkeypatch, _key(workspace_id=WS, permissions=["tasks:read"]))
        monkeypatch.setattr(hybrid.config, "DEFAULT_WORKSPACE_ID", str(OTHER_WS), raising=False)
        monkeypatch.setattr(hybrid.config, "WORKSPACE_ID", str(OTHER_WS), raising=False)
        ctx = _resolve(_FakeRequest({"Authorization": "Bearer ak_pub_abc"}))
        assert ctx.workspace_id == WS


# --------------------------------------------------------------------------- #
# 3. Dependency factory — routing + additive delegation (no regression)
# --------------------------------------------------------------------------- #

class TestRequireTaskContext:
    def test_sdk_bearer_routes_to_resolver(self, monkeypatch):
        rec = _key(permissions=["tasks:read"])
        _patch_valid(monkeypatch, rec)
        monkeypatch.setattr(hybrid, "SessionLocal",
                            lambda: SimpleNamespace(close=lambda: None))
        dep = require_task_context(TASKS_READ)
        ctx = asyncio.run(dep(_FakeRequest({"Authorization": "Bearer ak_pub_abc"})))
        assert ctx.auth_type == "sdk_key"
        assert ctx.workspace_id == WS

    def test_srv_key_prefix_also_routes_to_resolver(self, monkeypatch):
        rec = _key(permissions=["tasks:read"])
        _patch_valid(monkeypatch, rec)
        monkeypatch.setattr(hybrid, "SessionLocal",
                            lambda: SimpleNamespace(close=lambda: None))
        dep = require_task_context(TASKS_READ)
        ctx = asyncio.run(dep(_FakeRequest({"Authorization": "Bearer ak_srv_abc"})))
        assert ctx.auth_type == "sdk_key"

    def test_session_is_closed_even_on_error(self, monkeypatch):
        closed = {"v": False}
        monkeypatch.setattr(hybrid.ApiKeyService, "validate_api_key", lambda db, token: None)
        monkeypatch.setattr(
            hybrid, "SessionLocal",
            lambda: SimpleNamespace(close=lambda: closed.__setitem__("v", True)),
        )
        dep = require_task_context(TASKS_READ)
        with pytest.raises(HTTPException):
            asyncio.run(dep(_FakeRequest({"Authorization": "Bearer ak_pub_bad"})))
        assert closed["v"] is True

    def _delegating_dep(self, monkeypatch, sentinel):
        async def fake_hybrid(request):
            return sentinel
        monkeypatch.setattr(hybrid, "get_request_context_hybrid", fake_hybrid)
        return require_task_context(TASKS_READ)

    def test_clerk_bearer_delegates_to_hybrid(self, monkeypatch):
        sentinel = RequestContext(workspace_id=WS, user=UserContext(), auth_type="clerk")
        dep = self._delegating_dep(monkeypatch, sentinel)
        ctx = asyncio.run(dep(_FakeRequest({"Authorization": "Bearer eyJhbGciOi"})))
        assert ctx is sentinel

    def test_x_api_key_no_bearer_delegates(self, monkeypatch):
        sentinel = RequestContext(workspace_id=WS, user=UserContext(), auth_type="api_key")
        dep = self._delegating_dep(monkeypatch, sentinel)
        ctx = asyncio.run(dep(_FakeRequest({"X-Api-Key": "envkey"})))
        assert ctx is sentinel

    def test_options_delegates(self, monkeypatch):
        sentinel = RequestContext(workspace_id=WS, user=UserContext(), auth_type="anonymous")
        dep = self._delegating_dep(monkeypatch, sentinel)
        ctx = asyncio.run(dep(_FakeRequest({}, method="OPTIONS")))
        assert ctx is sentinel


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
