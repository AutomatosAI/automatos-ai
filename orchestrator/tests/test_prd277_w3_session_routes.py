"""PRD-277 Wave 3 (US-303): the routes that end a session, as declared, without a database.

* Where the edition issues no sessions, each route of ``api/auth_sessions.py`` answers its
  declared method with 404 before any authentication or database work (another method on the
  same path answers 405, as in every edition).
* Sign-out is authenticated by the session itself (its own credential, so it is in the boundary
  sweep's ``OWN_AUTH_ROUTES``) and never by the shared request context.
* The three admin changes carry step-up at ``STEP_UP_MAX_AGE_MINUTES`` and call
  ``_require_admin(ctx)`` in their handler (the sweep's ``ADMIN_GATED_IN_HANDLER``); the listing
  calls it too. ``SESSION_STEP_UP_ROUTES`` is the app-wide pin's share
  (``test_prd261_c1_wire.py``).
* Every route is in the committed ``route-manifest.json`` and the router in ``MANIFEST_ROUTERS``.
* ``_require_admin`` passes only a session-lane install admin: no key, device or member.
The behaviour against Postgres is in ``test_prd277_w3_sessions_realdb.py``.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

PREFIX = "/api/v1/auth"
SIGN_OUT = ("POST", PREFIX + "/sign-out")
LIST_SESSIONS = ("GET", PREFIX + "/people/{user_id}/sessions")
SESSION_STEP_UP_ROUTES = {
    ("POST", PREFIX + "/sessions/{session_id}/revoke"),
    ("POST", PREFIX + "/people/{user_id}/sessions/revoke"),
    ("POST", PREFIX + "/people/{user_id}/deactivate"),
}
SESSION_ROUTES = {SIGN_OUT, LIST_SESSIONS} | SESSION_STEP_UP_ROUTES

ORCH = Path(__file__).resolve().parents[1]


def _routes() -> dict:
    from fastapi.routing import APIRoute

    from api import auth_sessions

    return {(method, route.path): route for route in auth_sessions.router.routes if isinstance(route, APIRoute)
            for method in route.methods}


def _calls(dependant) -> list:
    found = [dependant.call]
    for sub in dependant.dependencies:
        found.extend(_calls(sub))
    return found


def test_the_router_declares_exactly_the_session_routes():
    assert set(_routes()) == SESSION_ROUTES


def test_the_admin_changes_carry_step_up_and_nothing_else_does():
    from config import config
    from core.auth import step_up

    for key, route in _routes().items():
        ages = [getattr(call, step_up.STEP_UP_MARKER_ATTR) for call in _calls(route.dependant)
                if getattr(call, step_up.STEP_UP_MARKER_ATTR, None) is not None]
        assert ages == ([config.STEP_UP_MAX_AGE_MINUTES] if key in SESSION_STEP_UP_ROUTES else []), key


def test_every_admin_route_checks_the_admin_in_its_handler():
    for key, route in _routes().items():
        source = inspect.getsource(route.endpoint)
        assert ("_require_admin(ctx" in source) == (key != SIGN_OUT), key


def test_sign_out_rides_its_own_credential_not_the_request_context():
    from core.auth.hybrid import get_request_context_hybrid

    route = _routes()[SIGN_OUT]
    assert get_request_context_hybrid not in _calls(route.dependant)
    assert "assert_cookie_request_allowed" in inspect.getsource(route.endpoint)


def test_the_routes_are_in_the_boundary_sweep_and_the_manifests():
    from router_manifest import MANIFEST_ROUTERS, RouterSpec
    from tests.test_p2w2_authz_boundary_sweep import ADMIN_GATED_IN_HANDLER, OWN_AUTH_ROUTES

    assert SIGN_OUT in OWN_AUTH_ROUTES and SIGN_OUT not in ADMIN_GATED_IN_HANDLER
    assert SESSION_STEP_UP_ROUTES <= ADMIN_GATED_IN_HANDLER
    assert not SESSION_STEP_UP_ROUTES & OWN_AUTH_ROUTES
    assert RouterSpec("api.auth_sessions") in MANIFEST_ROUTERS
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    listed = {(entry["method"], entry["path"]) for entry in manifest["routes"]}
    assert SESSION_ROUTES <= listed
    assert manifest["route_count"] == len(manifest["routes"])


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_every_declared_method_is_404_where_no_session_is_issued(monkeypatch, edition):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import auth_sessions
    from config import config
    from core.database.database import get_db

    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    app = FastAPI()
    app.include_router(auth_sessions.router)
    app.dependency_overrides[get_db] = lambda: pytest.fail("a closed route must not open a session")
    client = TestClient(app)
    token = "ams" + "_" + "x" * 43

    for method, path in SESSION_ROUTES:
        url = path.replace("{user_id}", "7").replace("{session_id}", "00000000-0000-0000-0000-000000000001")
        answer = client.request(method, url, headers={"authorization": "Bearer " + token})
        assert (answer.status_code, answer.json()) == (404, {"detail": "Not Found"}), (method, path)
    # A method a path does not declare is FastAPI's 405, as in every edition: the path is not hidden.
    assert client.get("/api/v1/auth/sign-out").status_code == 405


def _ctx(auth_type: str, role: str, claims: dict):
    return SimpleNamespace(auth_type=auth_type, user=SimpleNamespace(system_role=role, raw_claims=claims))


@pytest.mark.parametrize("ctx", [
    _ctx("session", "user", {"source": "auth_session", "user_id": 7}),
    _ctx("device", "super_admin", {"source": "mobile_device", "user_id": 7}),
    _ctx("api_key", "admin", {}),
    _ctx("clerk", "super_admin", {"source": "auth_session", "user_id": 7}),
    _ctx("session", "admin", {"source": "auth_session"}),
])
def test_only_a_session_lane_install_admin_passes(ctx):
    from fastapi import HTTPException

    from api import auth_sessions

    with pytest.raises(HTTPException) as refused:
        auth_sessions._require_admin(ctx)
    assert refused.value.status_code == 403


@pytest.mark.parametrize("role", ["admin", "super_admin"])
def test_a_session_lane_install_admin_is_named_by_users_id(role):
    from api import auth_sessions

    assert auth_sessions._require_admin(_ctx("session", role, {"source": "auth_session", "user_id": 7})) == 7
