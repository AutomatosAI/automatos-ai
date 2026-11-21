"""PRD-277 Wave 5 (US-501, US-502, US-504): the routes the web app adds, as declared, without a database.

* ``api/auth_web.py`` declares exactly ``GET /api/v1/auth/me`` and ``GET /api/v1/auth/sign-in-page``;
  ``api/team_waiting.py`` exactly ``GET`` and ``POST`` under ``/api/workspaces/{workspace_id}/team/waiting``.
* Where the edition issues no sessions (``local``, ``saas``) each answers its declared method with 404
  before any authentication or database work.
* ``/me`` rides its own credential (the session), never the shared request context, which refuses a
  person in no workspace; with no live session it is ``401 session_ended`` and clears the cookie.
* ``/sign-in-page`` is public and answers ``OIDC_DISPLAY_NAME`` only, never cached.
* Both waiting routes carry ``members:invite`` and check the install admin in their handler; only the
  POST carries step-up (``WAITING_STEP_UP_ROUTES``, the app-wide pin's share in
  ``test_prd261_c1_wire.py``).
* ``OIDC_DISPLAY_NAME`` defaults to the PRD's value, is in the config surface, and stops an enterprise
  start when it is not short plain text; ``local`` and ``saas`` never read it.
* Every route is in the committed ``route-manifest.json`` and each router in ``MANIFEST_ROUTERS``.
The behaviour against Postgres is in ``test_prd277_w5_web_realdb.py``.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ME = ("GET", "/api/v1/auth/me")
SIGN_IN_PAGE = ("GET", "/api/v1/auth/sign-in-page")
WEB_ROUTES = {ME, SIGN_IN_PAGE}
WAITING = "/api/workspaces/{workspace_id}/team/waiting"
LIST_WAITING = ("GET", WAITING)
GIVE_WORKSPACE = ("POST", WAITING + "/{user_id}")
WAITING_ROUTES = {LIST_WAITING, GIVE_WORKSPACE}
WAITING_STEP_UP_ROUTES = {GIVE_WORKSPACE}
WORKSPACE = "00000000-0000-0000-0000-0000000277f5"

ORCH = Path(__file__).resolve().parents[1]


def _routes(module) -> dict:
    from fastapi.routing import APIRoute

    return {(method, route.path): route for route in module.router.routes if isinstance(route, APIRoute)
            for method in route.methods}


def _calls(dependant) -> list:
    found = [dependant.call]
    for sub in dependant.dependencies:
        found.extend(_calls(sub))
    return found


def _app(*modules):
    from fastapi import FastAPI

    app = FastAPI()
    for module in modules:
        app.include_router(module.router)
    return app


def test_the_routers_declare_exactly_their_routes():
    from api import auth_web, team_waiting

    assert set(_routes(auth_web)) == WEB_ROUTES
    assert set(_routes(team_waiting)) == WAITING_ROUTES


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_every_declared_method_is_404_where_no_session_is_issued(monkeypatch, edition):
    from fastapi.testclient import TestClient

    from api import auth_web, team_waiting
    from config import config
    from core.database.database import get_db

    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    app = _app(auth_web, team_waiting)
    app.dependency_overrides[get_db] = lambda: pytest.fail("a closed route must not open a session")
    client = TestClient(app)
    token = "ams" + "_" + "x" * 43

    for method, path in WEB_ROUTES | WAITING_ROUTES:
        url = path.replace("{workspace_id}", WORKSPACE).replace("{user_id}", "7")
        answer = client.request(method, url, headers={"authorization": "Bearer " + token}, json={"role": "editor"})
        assert (answer.status_code, answer.json()) == (404, {"detail": "Not Found"}), (method, path)


def test_me_rides_its_own_credential_not_the_request_context():
    from api import auth_web
    from core.auth.hybrid import get_request_context_hybrid

    route = _routes(auth_web)[ME]
    assert get_request_context_hybrid not in _calls(route.dependant)
    assert "presented_session_token(request)" in inspect.getsource(route.endpoint)


def test_me_without_a_live_session_is_session_ended_and_clears_the_cookie(monkeypatch):
    from fastapi.testclient import TestClient

    from api import auth_web
    from config import config
    from core.database.database import get_db

    monkeypatch.setattr(config, "AUTH_EDITION", "enterprise")
    app = _app(auth_web)
    app.dependency_overrides[get_db] = lambda: object()  # no token: nothing is looked up
    answer = TestClient(app).get("/api/v1/auth/me")

    assert answer.status_code == 401
    assert answer.json()["detail"]["code"] == "session_ended"
    assert config.SESSION_COOKIE_NAME in answer.headers.get("set-cookie", "")


def test_the_sign_in_page_answers_the_display_name_only_and_is_never_cached(monkeypatch):
    from fastapi.testclient import TestClient

    from api import auth_web
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", "enterprise")
    monkeypatch.setattr(config, "OIDC_DISPLAY_NAME", "Example Corp")
    route = _routes(auth_web)[SIGN_IN_PAGE]
    assert not {"get_db", "get_request_context_hybrid"} & {getattr(c, "__name__", "") for c in _calls(route.dependant)}

    answer = TestClient(_app(auth_web)).get("/api/v1/auth/sign-in-page")

    assert (answer.status_code, answer.json()) == (200, {"display_name": "Example Corp"})
    assert answer.headers["cache-control"] == "no-store"


def test_the_waiting_routes_take_the_team_checks_and_only_the_post_steps_up():
    from api import team_waiting
    from config import config
    from core.auth import step_up

    for key, route in _routes(team_waiting).items():
        calls = _calls(route.dependant)
        ages = [getattr(c, step_up.STEP_UP_MARKER_ATTR) for c in calls
                if getattr(c, step_up.STEP_UP_MARKER_ATTR, None) is not None]
        assert ages == ([config.STEP_UP_MAX_AGE_MINUTES] if key in WAITING_STEP_UP_ROUTES else []), key
        assert any("require_workspace_permission" in getattr(c, "__qualname__", "") for c in calls), key
        assert "_require_admin(ctx)" in inspect.getsource(route.endpoint), key


def test_the_display_name_defaults_to_the_prds_and_is_in_the_surface():
    from config import config

    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"]
    assert "OIDC_DISPLAY_NAME" in surface and surface == sorted(surface)
    if "OIDC_DISPLAY_NAME" not in os.environ:
        assert config.OIDC_DISPLAY_NAME == "your organisation"


@pytest.mark.parametrize("value", ["", " padded ", "<b>Corp</b>", "Corp\nNext", "Corp\x7f", "x" * 65, "Tab\there"])
def test_a_display_name_that_is_not_short_plain_text_stops_enterprise(value):
    from core import edition_start_up
    from core.editions import capabilities_for
    from tests.test_prd277_w4b_access_rule import _settings

    with pytest.raises(edition_start_up.EditionNotReady) as refused:
        edition_start_up.check_start_up(capabilities_for("enterprise"), _settings(OIDC_DISPLAY_NAME=value))
    assert "OIDC_DISPLAY_NAME" in str(refused.value)
    assert value.strip() == "" or value not in str(refused.value)  # the value is never quoted


@pytest.mark.parametrize("value", ["your organisation", "Example Corp", "Entra ID (Example)", "x" * 64, "Société"])
def test_a_short_plain_display_name_starts(value):
    from core import edition_start_up
    from core.editions import capabilities_for
    from tests.test_prd277_w4b_access_rule import _settings

    assert edition_start_up.check_start_up(capabilities_for("enterprise"), _settings(OIDC_DISPLAY_NAME=value)) is None


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_local_and_saas_never_read_the_display_name(edition):
    from core import edition_start_up
    from core.editions import capabilities_for
    from tests.test_prd277_w4b_access_rule import _settings

    assert edition_start_up.check_start_up(capabilities_for(edition), _settings(OIDC_DISPLAY_NAME="<bad>")) is None


def test_the_routes_are_in_the_boundary_sweep_and_the_manifests():
    from router_manifest import MANIFEST_ROUTERS, RouterSpec
    from tests.test_p2w2_authz_boundary_sweep import ADMIN_GATED_IN_HANDLER, OWN_AUTH_ROUTES, PUBLIC_BY_DESIGN

    # The POST is classified by its require_workspace_permission gate, once; the GETs change nothing.
    assert not WAITING_ROUTES & (OWN_AUTH_ROUTES | ADMIN_GATED_IN_HANDLER | PUBLIC_BY_DESIGN)
    assert {RouterSpec("api.auth_web"), RouterSpec("api.team_waiting")} <= set(MANIFEST_ROUTERS)
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    listed = [(entry["method"], entry["path"]) for entry in manifest["routes"]]
    assert WEB_ROUTES | WAITING_ROUTES <= set(listed)
    assert manifest["route_count"] == len(manifest["routes"])
    assert listed == sorted(listed, key=lambda pair: (pair[1], pair[0]))
