"""PRD-261 B2-wire: step-up refusals on the real routes (A-07c applied; Gerard D1-d, #1157).

The real routers run on a mini app with the step-up handler installed, the REAL permission
gate (only the caller's workspace role is stubbed) and the REAL step-up dependency. Pins:

* in SaaS, import, opt-in, default, update and the settings PUT answer 403 with Clerk's
  reverification body at the top level when the first factor is missing, never verified or
  older than ``THEME_STEP_UP_MAX_AGE_MINUTES``, and no handler runs;
* a caller the permission gate refuses gets the plain 403, never a reverification prompt;
* the workspace role change (``members:change_role``) is refused in SaaS without a fresh
  step-up and the role is unchanged; with one it is allowed; the local edition's one owner
  changes a role with no step-up.

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

import api.team as team_api  # noqa: E402
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
    assert level == {"level": "first_factor", "afterMinutes": config.THEME_STEP_UP_MAX_AGE_MINUTES}
    assert body["code"] == step_up.STEP_UP_CODE


INSTALL_ID = "00000000-0000-0000-0000-00000000b2a2"
PACKAGE_ID = "00000000-0000-0000-0000-00000000b2a3"
STEP_UP_CALLS = [
    ("post", "/api/themes/import", {"files": {"file": ("x.autotheme", b"PK", "application/zip")}}),
    ("post", f"/api/themes/installs/{INSTALL_ID}/opt-in", {"json": {"packageId": PACKAGE_ID}}),
    ("put", "/api/themes/default", {"json": {"installId": None}}),
    ("post", f"/api/themes/installs/{INSTALL_ID}/update", {"json": {"packageId": PACKAGE_ID}}),
    ("put", "/api/themes/settings", {"json": {"membersMayChoose": False}}),
]


@pytest.fixture
def saas_owner(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    role = SimpleNamespace(value="owner")
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: role.value)
    return role


@pytest.mark.parametrize("fva", [None, [-1, -1], [config.THEME_STEP_UP_MAX_AGE_MINUTES + 5, -1]])
@pytest.mark.parametrize("method,path,kwargs", STEP_UP_CALLS)
def test_a_theme_route_without_a_fresh_step_up_is_refused_in_saas(saas_owner, fva, method, path, kwargs):
    client = _app(import_api.router, install_api.router, settings_api.router, ctx=_clerk(fva), db=_NoSettingsDb())
    _assert_reverification(getattr(client, method)(path, **kwargs))


def test_a_viewer_is_refused_by_the_gate_before_any_reverification_prompt(saas_owner):
    saas_owner.value = "viewer"
    client = _app(install_api.router, ctx=_clerk(None), db=_NoSettingsDb())
    resp = client.put("/api/themes/default", json={"installId": None})
    assert resp.status_code == 403 and "clerk_error" not in resp.json()


# ---------------------------------------------------------------------------
# Behaviour: the workspace role change (D1-d, #1157)
# ---------------------------------------------------------------------------


class _RoleDb:
    """Just enough session for ``update_member_role``: one editor row, and commits counted."""

    def __init__(self):
        self.member = SimpleNamespace(id=5, role="editor")
        self.commits = 0

    def query(self, model):
        return self

    def filter(self, *conditions):
        return self

    def first(self):
        return self.member

    def commit(self):
        self.commits += 1


@pytest.fixture
def role_change(monkeypatch):
    audits = []
    monkeypatch.setattr(team_api, "_resolve_internal_user_id", lambda db, ctx: 1)
    monkeypatch.setattr(team_api, "AuditService", lambda db: SimpleNamespace(log=lambda **entry: audits.append(entry)))
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    db = _RoleDb()

    def change(ctx):
        client = _app(team_api.router, ctx=ctx, db=db)
        return client.patch(f"/api/workspaces/{WS}/team/members/5/role", json={"role": "admin"})

    return SimpleNamespace(change=change, db=db, audits=audits)


@pytest.mark.parametrize("fva", [None, [-1, -1], [30, -1]])
def test_a_role_change_without_a_fresh_step_up_is_refused_in_saas(role_change, monkeypatch, fva):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    _assert_reverification(role_change.change(_clerk(fva)))
    assert role_change.db.member.role == "editor" and role_change.db.commits == 0
    assert role_change.audits == []


def test_a_role_change_with_a_fresh_step_up_is_allowed_in_saas(role_change, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    resp = role_change.change(_clerk([1, -1]))
    assert resp.status_code == 200, resp.text
    assert role_change.db.member.role == "admin" and role_change.audits[0]["action"] == "member:role_changed"


def test_the_local_editions_owner_changes_a_role_with_no_step_up(role_change, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    operator = RequestContext(workspace_id=WS, auth_type="anonymous",
                              user=UserContext(id="operator@local", email="operator@local", system_role="user"))
    resp = role_change.change(operator)
    assert resp.status_code == 200, resp.text
    assert role_change.db.member.role == "admin"
