"""PRD-261 B2-wire: step-up refusals on the real routes (A-07c applied; Gerard D1-d, #1157).

The real routers run on a mini app with the step-up handler installed, the REAL permission
gate (only the caller's workspace role is stubbed) and the REAL step-up dependency. Pins:

* in SaaS, import, opt-in, default, update, rollback and the settings PUT answer 403 with Clerk's
  reverification body at the top level when the first factor is missing, never verified or
  older than ``STEP_UP_MAX_AGE_MINUTES``, and no handler runs;
* a caller the permission gate refuses gets the plain 403, never a reverification prompt;
* rollback (it switches the running theme code, like update) runs with a fresh step-up in
  SaaS and with none in the local edition;
* the workspace role change carries the same step-up since PRD-261 A-07f (W-A2, Gerard D1-d,
  #1157): its refusals are ``test_prd261_a07f_account_step_up.py``.

The route table facts (which routes carry the dependency, in which order) are
``test_prd261_b2_wire.py``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ORCH = Path(__file__).resolve().parents[1]
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.themes_import as import_api  # noqa: E402
import api.themes_install as install_api  # noqa: E402
import api.themes_settings as settings_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
from config import config  # noqa: E402
from core.auth import step_up  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402

WS = UUID("00000000-0000-0000-0000-00000000b2a1")


# ---------------------------------------------------------------------------
# The theme routes
# ---------------------------------------------------------------------------


def _clerk(fva=None, system_role="user"):
    claims = {"sub": "clerk-b2", "iat": int(time.time())}
    if fva is not None:
        claims["fva"] = fva
    user = UserContext(id="clerk-b2", clerk_user_id="clerk-b2", system_role=system_role, raw_claims=claims)
    return RequestContext(workspace_id=WS, user=user, auth_type="clerk")


class _NoSettingsDb:
    """A session the gates may read (no settings row) but a handler must never write to."""

    def get(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        raise AssertionError(f"the handler ran (db.{name}) although a gate should have refused")


def _app(*routers, ctx, db):
    app = FastAPI()
    step_up.install_step_up_handler(app)
    for router in routers:
        app.include_router(router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app, raise_server_exceptions=False)


def _assert_reverification(resp):
    assert resp.status_code == 403, resp.text
    body = resp.json()
    level = body["clerk_error"]["metadata"]["reverification"]
    assert level == {"level": "first_factor", "afterMinutes": config.STEP_UP_MAX_AGE_MINUTES}
    assert body["code"] == step_up.STEP_UP_CODE


INSTALL_ID = "00000000-0000-0000-0000-00000000b2a2"
PACKAGE_ID = "00000000-0000-0000-0000-00000000b2a3"
STEP_UP_CALLS = [
    ("post", "/api/themes/import", {"files": {"file": ("x.autotheme", b"PK", "application/zip")}}),
    ("post", f"/api/themes/installs/{INSTALL_ID}/opt-in", {"json": {"packageId": PACKAGE_ID}}),
    ("put", "/api/themes/default", {"json": {"installId": None}}),
    ("post", f"/api/themes/installs/{INSTALL_ID}/update", {"json": {"packageId": PACKAGE_ID}}),
    ("post", f"/api/themes/installs/{INSTALL_ID}/rollback", {}),
    ("put", "/api/themes/settings", {"json": {"membersMayChoose": False}}),
]


@pytest.fixture
def saas_owner(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    # Uploads on, so the import route's private-upload gate (before the step-up) lets the caller through.
    monkeypatch.setattr(config, "THEMES_PRIVATE_UPLOAD", "on")
    role = SimpleNamespace(value="owner")
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: role.value)
    return role


@pytest.mark.parametrize("fva", [None, [-1, -1], [config.STEP_UP_MAX_AGE_MINUTES + 5, -1]])
@pytest.mark.parametrize("method,path,kwargs", STEP_UP_CALLS)
def test_a_theme_route_without_a_fresh_step_up_is_refused_in_saas(saas_owner, fva, method, path, kwargs):
    client = _app(import_api.router, install_api.router, settings_api.router, ctx=_clerk(fva), db=_NoSettingsDb())
    _assert_reverification(getattr(client, method)(path, **kwargs))


def test_a_viewer_is_refused_by_the_gate_before_any_reverification_prompt(saas_owner):
    saas_owner.value = "viewer"
    client = _app(install_api.router, ctx=_clerk(None), db=_NoSettingsDb())
    resp = client.put("/api/themes/default", json={"installId": None})
    assert resp.status_code == 403 and "clerk_error" not in resp.json()


class _RollbackDb:
    """The gates read no settings row; the handler's one write is a counted commit."""

    def __init__(self):
        self.commits = 0

    def get(self, *args, **kwargs):
        return None

    def commit(self):
        self.commits += 1

    def rollback(self):
        raise AssertionError("the rollback was refused")


def _local_operator():
    return RequestContext(workspace_id=WS, auth_type="anonymous",
                          user=UserContext(id="operator@local", email="operator@local", system_role="user"))


@pytest.mark.parametrize("edition", ["saas", "local"])
def test_rollback_runs_with_a_fresh_step_up_in_saas_and_with_none_locally(monkeypatch, edition):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    ctx = _clerk([1, -1]) if edition == "saas" else _local_operator()
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    rolled = []
    monkeypatch.setattr(install_api.updates, "rollback", lambda db, ws, install: rolled.append((ws, install)) or "row")
    monkeypatch.setattr(install_api, "_install_answer", lambda db, ctx, install: {"install": install})
    db = _RollbackDb()
    resp = _app(install_api.router, ctx=ctx, db=db).post(f"/api/themes/installs/{INSTALL_ID}/rollback")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"install": "row"} and rolled == [(WS, UUID(INSTALL_ID))] and db.commits == 1
