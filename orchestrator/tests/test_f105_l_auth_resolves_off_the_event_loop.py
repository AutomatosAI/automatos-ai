"""F105 (4 Nov): resolving who a request is runs off the event loop; who it is does not change.

``get_request_context_hybrid`` backs nearly every route. For every request without a
device token it checked the Clerk token and read the workspace, the membership and the
operator row on a session of its own, synchronously, on the event loop: a slow query or
a wait on the pool froze every request on the worker. That work now runs in the
threadpool (``_resolve_request_context``), as the device lane's lookup already did
(``core.auth.device``), and so does the board plane's SDK-key lookup
(``require_task_context``). The context, every refusal (status and detail) and the log
context set on the request's own context are unchanged; a CORS preflight still answers at
once, with no session.

DB-free: the lookups are patched to record the thread they run on.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException

from config import config
from core.auth import device, hybrid
from core.auth.dependencies import RequestContext, UserContext
from core.utils.logging_adapter import user_id_var, workspace_id_var

WS = UUID("6a0f3c1e-2b4d-4e8f-9a1b-f10500a0c001")
CLERK_USER = "user_2f105owner"
INFO = {"clerk_user_id": CLERK_USER, "email": "owner@f105.test", "role": "user"}


class _Seen:
    """Each lookup, in order, with the thread it ran on."""

    def __init__(self):
        self.steps = []

    def record(self, name, result=None):
        def _step(*args, **kwargs):
            self.steps.append((name, threading.get_ident()))
            return result
        return _step

    def names(self):
        return [name for name, _ in self.steps]

    def threads(self):
        return {thread for _, thread in self.steps}


class _Clerk:
    def __init__(self, seen, claims):
        self._seen, self._claims = seen, claims

    def verify_token(self, token):
        self._seen.record("token verified")()
        return self._claims

    def extract_user_info(self, claims):
        return dict(INFO)


@pytest.fixture
def seen(monkeypatch):
    seen = _Seen()
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "REQUIRE_AUTH", True)
    monkeypatch.setattr(hybrid, "SessionLocal", lambda: NS(close=seen.record("session closed")))
    monkeypatch.setattr(hybrid, "get_clerk_auth", lambda: _Clerk(seen, {"sub": CLERK_USER}))
    monkeypatch.setattr(hybrid, "_touch_last_seen", seen.record("last seen stamped"))
    monkeypatch.setattr(hybrid, "_workspace_exists", seen.record("workspace read", True))
    monkeypatch.setattr(hybrid, "_user_has_workspace_access", seen.record("membership read", True))
    monkeypatch.setattr(hybrid, "_assert_workspace_usable", seen.record("workspace usable"))
    return seen


def _request(headers=None, method="GET"):
    return NS(method=method, headers=dict(headers or {}), query_params={}, state=NS())


def _resolve(request, dependency=None):
    """(the loop's thread, the context, the log context the request then carries)."""
    dependency = dependency or hybrid.get_request_context_hybrid

    async def run():
        context = await dependency(request)
        return threading.get_ident(), context, (workspace_id_var.get(""), user_id_var.get(""))

    return asyncio.run(run())


def _refusal(request) -> HTTPException:
    with pytest.raises(HTTPException) as refused:
        _resolve(request)
    return refused.value


CLERK_REQUEST = {"authorization": "Bearer eyJhbGciOi.f105.sig", "x-workspace-id": str(WS)}


def test_a_clerk_request_resolves_off_the_loop_to_the_same_context(seen):
    loop_thread, context, log_context = _resolve(_request(CLERK_REQUEST))

    assert (context.auth_type, context.workspace_id, context.user.id) == ("clerk", WS, CLERK_USER)
    assert context.user.email == INFO["email"] and context.user.system_role == "user"
    assert seen.names() == ["token verified", "last seen stamped", "workspace read", "membership read",
                            "workspace read", "workspace usable", "session closed"]
    assert loop_thread not in seen.threads()
    assert log_context == (str(WS), CLERK_USER)  # set on the request's own context


def test_the_env_api_key_resolves_off_the_loop(seen, monkeypatch):
    monkeypatch.setattr(config, "ORCHESTRATOR_API_KEY", "f105-env-key")

    loop_thread, context, _ = _resolve(_request({"x-api-key": "f105-env-key", "x-workspace-id": str(WS)}))

    assert (context.auth_type, context.workspace_id, context.user.system_role) == ("api_key", WS, "admin")
    assert "workspace read" in seen.names() and loop_thread not in seen.threads()


def test_an_anonymous_request_resolves_off_the_loop_when_auth_is_off(seen, monkeypatch):
    monkeypatch.setattr(config, "REQUIRE_AUTH", False)

    loop_thread, context, _ = _resolve(_request({"x-workspace-id": str(WS)}))

    assert (context.auth_type, context.workspace_id) == ("anonymous", WS)
    assert seen.names() == ["workspace usable", "session closed"] and loop_thread not in seen.threads()


def test_the_refusals_are_unchanged(seen, monkeypatch):
    monkeypatch.setattr(config, "ORCHESTRATOR_API_KEY", "f105-env-key")
    monkeypatch.setattr(hybrid, "get_clerk_auth", lambda: _Clerk(seen, None))
    refused = _refusal(_request(CLERK_REQUEST))
    assert (refused.status_code, refused.detail) == (401, "Invalid or expired token")

    refused = _refusal(_request({"x-api-key": "not-the-key"}))
    assert (refused.status_code, refused.detail) == (401, "Invalid API key")

    refused = _refusal(_request())
    assert (refused.status_code, refused.detail) == (401, "Authentication required")

    monkeypatch.setattr(hybrid, "get_clerk_auth", lambda: _Clerk(seen, {"sub": CLERK_USER}))
    monkeypatch.setattr(hybrid, "_workspace_exists", seen.record("workspace read", False))
    refused = _refusal(_request(CLERK_REQUEST))
    assert (refused.status_code, refused.detail) == (400, "Invalid workspace_id")


def test_a_preflight_answers_at_once_with_no_session(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(hybrid, "SessionLocal", lambda: pytest.fail("a preflight opens no session"))

    _, context, _ = _resolve(_request(method="OPTIONS"))

    assert context.auth_type == "anonymous"
    assert context.workspace_id == UUID("00000000-0000-0000-0000-000000000001")


def test_a_board_sdk_key_resolves_off_the_loop(seen, monkeypatch):
    sdk = RequestContext(workspace_id=WS, user=UserContext(id="sdk:9", role="service", system_role="service"),
                         auth_type="sdk_key", api_key_id="9")

    def resolve_key(request, db, *, required_scope):
        seen.record(f"key checked for {required_scope}")()
        return sdk

    monkeypatch.setattr(hybrid, "_resolve_sdk_key_context", resolve_key)
    dependency = hybrid.require_task_context("tasks:read")

    loop_thread, context, log_context = _resolve(_request({"authorization": "Bearer ak_pub_f105"}), dependency)

    assert context is sdk
    assert seen.names() == ["key checked for tasks:read", "session closed"]
    assert loop_thread not in seen.threads()
    assert log_context == (str(WS), "sdk:9")


def test_a_paired_device_resolves_off_the_loop_and_never_reaches_the_clerk_lane(seen, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    paired = RequestContext(workspace_id=WS, user=UserContext(id="owner@f105.test"), auth_type=device.DEVICE_AUTH_TYPE)

    def lookup(request, token):
        seen.record("device looked up")()
        return paired

    monkeypatch.setattr(device, "_resolve_in_own_session", lookup)

    loop_thread, context, _ = _resolve(_request({"authorization": "Bearer amd_f105"}))

    assert context is paired
    assert seen.names() == ["device looked up"] and loop_thread not in seen.threads()
