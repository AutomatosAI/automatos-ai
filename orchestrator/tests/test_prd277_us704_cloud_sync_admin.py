"""PRD-277 US-704: who may run Sync with Automatos Cloud once an enterprise install can switch it on.

A Sync brings packages and items in and moves installs for the whole install. Where the person using the app is
not its operator (``operator_is_user`` false: enterprise), ``POST /api/marketplace/cloud-sync`` needs, after the
edition's check and ``workspace:manage``, an install admin (``caller_is_admin``) and then a fresh step-up
(``require_step_up``). No database, and nothing reaches Automatos Cloud: ``pull.sync_catalog`` is replaced by a
recorder that answers "already running" (409), so a 409 proves every gate passed. Proven:

* in enterprise with Cloud sync on, a workspace owner who is not an install admin gets 403
  ``SYNC_ADMIN_REQUIRED`` and is never asked to sign in again; an install admin whose sign-in is too old is sent
  to sign in again (``step_up_required``); an install admin with a fresh sign-in reaches the Sync;
* the status read is unchanged: ``workspace:manage`` alone;
* in ``local`` the operator, whatever its role, reaches the Sync as before: no install-admin check runs and the
  step-up is ``skip``;
* the route checks the edition first, then ``workspace:manage``, then the install admin, then the step-up;
  ``CLOUD_SYNC_STEP_UP_ROUTES`` is this route's share of the app-wide step-up pin in ``test_prd261_c1_wire.py``.
"""
from __future__ import annotations

import os
import time
import uuid

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.marketplace_cloud_sync as sync_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
from config import config  # noqa: E402
from core.auth import step_up  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.principal import AUTH_SESSION_CLAIM_SOURCE, LOCAL_OPERATOR_CLAIM_SOURCE  # noqa: E402
from core.auth.session_lane import SESSION_AUTH_TYPE  # noqa: E402
from core.database.database import get_db  # noqa: E402
from services.marketplace_cloud import pull  # noqa: E402

SYNC = "/api/marketplace/cloud-sync"
STATUS = SYNC + "/status"
CLOUD_SYNC_STEP_UP_ROUTES = {("POST", SYNC)}
WS = uuid.uuid4()
CLOUD = "https://cloud.example.test"
BUSY = "busy"


def _person(system_role: str, signed_in_minutes_ago: float) -> RequestContext:
    """A person signed in through the enterprise session lane, ``signed_in_minutes_ago`` minutes ago."""
    claims = {"source": AUTH_SESSION_CLAIM_SOURCE, "user_id": 7, "auth_time": time.time() - signed_in_minutes_ago * 60}
    user = UserContext(id="person-704", email="person@example.test", system_role=system_role, raw_claims=claims)
    return RequestContext(workspace_id=WS, user=user, auth_type=SESSION_AUTH_TYPE)


def _operator() -> RequestContext:
    """The local edition's operator as the existing Sync tests bind it: a plain ``user`` role, the owner."""
    user = UserContext(id="operator@local", email="operator@local", system_role="user",
                       raw_claims={"source": LOCAL_OPERATOR_CLAIM_SOURCE, "user_id": 1})
    return RequestContext(workspace_id=WS, user=user, auth_type="anonymous")


@pytest.fixture
def sync(monkeypatch):
    """The Sync router alone, the caller a workspace owner; ``ran`` lists each Sync that started."""
    ran = []
    state = {"ctx": _operator()}
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", CLOUD)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr(pull, "sync_catalog", lambda db, factory: ran.append(1) or pull.SyncOutcome(
        ok=False, error=BUSY, busy=True))
    monkeypatch.setattr(pull, "sync_status", lambda db: {"enabled": True})
    app = FastAPI()
    app.include_router(sync_api.router)
    step_up.install_step_up_handler(app)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state["ctx"]
    app.dependency_overrides[get_db] = lambda: None

    def call(method: str, path: str, ctx: RequestContext):
        state["ctx"] = ctx
        return TestClient(app).request(method, path)

    return call, ran


def _enterprise_with_sync_on(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "enterprise")
    monkeypatch.setattr(config, "ENTERPRISE_CALLS_THEME_REVOCATIONS", "true")
    monkeypatch.setattr(config, "ENTERPRISE_CALLS_CLOUD_SYNC", "true")
    monkeypatch.setattr(config, "ENTERPRISE_CALLS_CLOUD_ITEMS", "false")


def test_in_enterprise_a_workspace_owner_who_is_not_an_install_admin_is_refused(sync, monkeypatch):
    call, ran = sync
    _enterprise_with_sync_on(monkeypatch)
    answer = call("POST", SYNC, _person("user", signed_in_minutes_ago=0))
    assert answer.status_code == 403 and answer.json() == {"detail": sync_api.SYNC_ADMIN_REQUIRED}
    assert ran == []


def test_in_enterprise_an_install_admin_with_an_old_sign_in_is_asked_to_sign_in_again(sync, monkeypatch):
    call, ran = sync
    _enterprise_with_sync_on(monkeypatch)
    answer = call("POST", SYNC, _person("admin", signed_in_minutes_ago=config.STEP_UP_MAX_AGE_MINUTES + 5))
    assert answer.status_code == 403 and answer.json()["code"] == step_up.STEP_UP_CODE
    assert answer.json()["step_up"]["start"] == step_up.SIGN_IN_AGAIN_START
    assert ran == []


@pytest.mark.parametrize("role", ["admin", "super_admin"])
def test_in_enterprise_an_install_admin_with_a_fresh_sign_in_reaches_the_sync(sync, monkeypatch, role):
    call, ran = sync
    _enterprise_with_sync_on(monkeypatch)
    answer = call("POST", SYNC, _person(role, signed_in_minutes_ago=0))
    assert answer.status_code == 409 and answer.json() == {"detail": BUSY} and ran == [1]


def test_in_enterprise_the_status_still_needs_only_workspace_manage(sync, monkeypatch):
    call, _ = sync
    _enterprise_with_sync_on(monkeypatch)
    answer = call("GET", STATUS, _person("user", signed_in_minutes_ago=600))
    assert answer.status_code == 200 and answer.json() == {"enabled": True}


def test_in_local_the_operator_reaches_the_sync_as_before(sync, monkeypatch):
    call, ran = sync
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    answer = call("POST", SYNC, _operator())
    assert answer.status_code == 409 and answer.json() == {"detail": BUSY} and ran == [1]


def _calls(dependant):
    found = []
    for sub in dependant.dependencies:
        found += [sub.call] + _calls(sub)
    return found


def test_the_sync_checks_the_edition_then_the_workspace_then_the_install_admin_then_the_step_up():
    [route] = [r for r in sync_api.router.routes if isinstance(r, APIRoute) and "POST" in r.methods]
    calls = _calls(route.dependant)
    marked = [n for n, c in enumerate(calls) if getattr(c, step_up.STEP_UP_MARKER_ATTR, None) is not None]
    gate = next(n for n, c in enumerate(calls) if getattr(c, permission_mod.PERMISSION_MARKER_ATTR, None))
    assert calls[0] is sync_api.require_local_edition
    assert gate < calls.index(sync_api.require_sync_admin) < marked[0] and len(marked) == 1
    assert getattr(calls[marked[0]], step_up.STEP_UP_MARKER_ATTR) == config.STEP_UP_MAX_AGE_MINUTES
    [status] = [r for r in sync_api.router.routes if isinstance(r, APIRoute) and "GET" in r.methods]
    assert sync_api.require_sync_admin not in _calls(status.dependant)
