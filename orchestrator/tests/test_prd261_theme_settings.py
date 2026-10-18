"""PRD-261 A-07e — the workspace theme settings (PRD §5 admin controls, D1-b).

The real routers (``api/themes_settings.py``, and A-07b's ``api/themes_install.py`` for the
enforcement proofs) run on a mini FastAPI app over in-memory SQLite, with the REAL
workspace-permission gate (only the caller's role is stubbed) and the REAL ``resolve_user_pk``.
Pins (plan §2.6 A-07e): each default per edition, switch and member count (Q6); a
``members:read`` caller cannot PUT; a pinned install refuses update; members-may-choose off
-> ``PUT choice`` 403 and ``active`` returns the default; who may install only narrows and
only the owner moves it; another workspace's install is 404; the local anonymous owner works.
"""
from __future__ import annotations

import inspect
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from fastapi import Depends, FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.themes_install as install_api  # noqa: E402
import api.themes_settings as settings_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.models  # noqa: E402,F401  (registers every table the FKs name)
import services.themes.settings as settings_mod  # noqa: E402
from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.principal import LOCAL_OPERATOR_CLAIM_SOURCE  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.themes import (  # noqa: E402
    ThemeChoice,
    ThemeInstallation,
    ThemeLook,
    ThemeOptIn,
    ThemePackage,
    ThemeRevocation,
    ThemeWorkspaceSettings,
)
from core.workspaces.models import WorkspaceMember  # noqa: E402
from services.themes import records  # noqa: E402
from services.themes.records import PRIVATE_UPLOAD_SETTING  # noqa: E402

WS_A = uuid.uuid4()
WS_B = uuid.uuid4()
MEMBER_PK, OPERATOR_PK = 7, 1
SETTINGS = "/api/themes/settings"
TABLES = [m.__table__ for m in (
    ThemePackage, ThemeInstallation, ThemeChoice, ThemeLook, ThemeOptIn, ThemeWorkspaceSettings, ThemeRevocation,
)]
# A-07a's test's DDL: the model's postgresql UUID column has NUMERIC affinity on SQLite.
MEMBERS_DDL = (
    "CREATE TABLE workspace_members (id INTEGER PRIMARY KEY, workspace_id CHAR(32) NOT NULL, "
    "user_id INTEGER NOT NULL, role VARCHAR(50), invited_by INTEGER, invited_at DATETIME, "
    "joined_at DATETIME, is_active BOOLEAN)"
)
# ``workspaces.owner_id`` names an owner who may have no member row (records.people_count).
WORKSPACES_DDL = "CREATE TABLE workspaces (id CHAR(32) PRIMARY KEY, owner_id INTEGER)"


def _ctx(workspace_id=WS_A, clerk="clerk-7"):
    """A Clerk member who signed in a minute ago, so the step-up routes (A-07c) pass in SaaS."""
    claims = {"sub": clerk, "fva": [1, -1], "iat": int(time.time())}
    return RequestContext(
        workspace_id=workspace_id, user=UserContext(id=clerk, clerk_user_id=clerk, system_role="user", raw_claims=claims),
        auth_type="clerk",
    )


def _install_gated_probe(app):
    """An install route as A-07a/A-07b/C-04a wire it: the manage gate, then who may install."""

    @app.post("/probe/install")
    def probe(ctx=Depends(settings_api.MANAGE), _=Depends(settings_mod.require_install_role())):
        return {"ok": True}


@pytest.fixture
def api(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=TABLES)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY, clerk_user_id TEXT, email TEXT)"))
        conn.execute(sa.text("INSERT INTO users (id, clerk_user_id) VALUES (7, 'clerk-7'), (8, 'clerk-8')"))
        conn.execute(sa.text(MEMBERS_DDL))
        conn.execute(sa.text(WORKSPACES_DDL))
    session = sessionmaker(bind=engine)()
    state = SimpleNamespace(db=session, ctx=_ctx(), role="owner")
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: state.role)
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, PRIVATE_UPLOAD_SETTING, "auto", raising=False)
    app = FastAPI()
    app.include_router(settings_api.router)
    app.include_router(install_api.router)
    _install_gated_probe(app)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        session.close()
        engine.dispose()


def _members(db, workspace_id, count):
    db.add_all([WorkspaceMember(workspace_id=workspace_id, user_id=n + 1, role="viewer", is_active=True) for n in range(count)])
    db.commit()


def _package(db, workspace_id=None, version="1.0.0", theme_id="acme/ink", reviewed=None):
    owner = "workspace" if workspace_id else "marketplace"
    row = ThemePackage(
        theme_id=theme_id, version=version, owner_type=owner, workspace_id=workspace_id,
        digest=uuid.uuid4().hex * 2, manifest={"id": theme_id, "name": "Ink", "version": version, "base": "studio"},
        file_manifest={"dist/entry.js": {"type": "text/javascript", "size": 1, "sha256": "ab" * 32}},
        storage_prefix=f"themes/{theme_id}/{version}/",
        reviewed=(owner == "marketplace") if reviewed is None else reviewed,
        status="approved" if owner == "marketplace" else "pending",
    )
    db.add(row)
    db.commit()
    return row


def _install(db, package, workspace_id=WS_A, **extra):
    row = ThemeInstallation(workspace_id=workspace_id, theme_id=package.theme_id, package_id=package.id, **extra)
    db.add(row)
    db.commit()
    return row


def _get(api):
    resp = api.client.get(SETTINGS)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _put(api, **body):
    return api.client.put(SETTINGS, json=body)


# ---------------------------------------------------------------------------
# Defaults per edition, platform switch and member count (Q6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("edition, switch, members, default, effective", [
    ("local", "auto", 2, "on", "on"),
    ("saas", "auto", 1, "on", "on"),
    ("saas", "auto", 2, "off", "off"),
    ("saas", "on", 2, "on", "on"),
    ("saas", "off", 1, "off", "off"),
    ("saas", "admins", 1, "on", "admins"),
])
def test_each_default_per_edition_and_member_count(api, monkeypatch, edition, switch, members, default, effective):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(config, PRIVATE_UPLOAD_SETTING, switch, raising=False)
    _members(api.db, WS_A, members)
    body = _get(api)
    assert body["privateUploads"] == {
        "setting": None, "source": "default", "default": default, "platform": switch, "effective": effective,
    }
    assert (body["membersMayChoose"], body["installRole"], body["canInstall"]) == (True, "admins", True)
    assert body["installRoles"] == ["admins", "owner"]
    assert (body["defaultInstallId"], body["pins"], body["updatedAt"]) == (None, [], None)


def test_an_owner_without_a_member_row_and_one_member_are_two_people_so_auto_is_off(api):
    # S2: workspaces.owner_id names an owner who has no workspace_members row; with one
    # member that is two people, so SaaS ``auto`` defaults uploads off.
    api.db.execute(sa.text("INSERT INTO workspaces (id, owner_id) VALUES (:id, :owner)"), {"id": WS_A.hex, "owner": 99})
    api.db.commit()
    _members(api.db, WS_A, 1)
    assert records.people_count(api.db, WS_A) == 2
    assert _get(api)["privateUploads"]["default"] == "off"


def test_an_owner_who_is_also_the_member_counts_once(api):
    api.db.execute(sa.text("INSERT INTO workspaces (id, owner_id) VALUES (:id, :owner)"), {"id": WS_A.hex, "owner": 1})
    api.db.commit()
    _members(api.db, WS_A, 1)  # user_id 1
    assert records.people_count(api.db, WS_A) == 1
    assert _get(api)["privateUploads"]["default"] == "on"


def test_auto_is_a_default_an_admin_can_switch_on_and_null_puts_it_back(api):
    _members(api.db, WS_A, 2)
    api.role = "admin"
    resp = _put(api, privateUploads="on")
    assert resp.status_code == 200, resp.text
    assert resp.json()["privateUploads"] == {
        "setting": "on", "source": "workspace", "default": "off", "platform": "auto", "effective": "on",
    }
    row = api.db.get(ThemeWorkspaceSettings, WS_A)
    assert (row.private_uploads, row.updated_by) == ("on", MEMBER_PK)
    assert _put(api, privateUploads=None).json()["privateUploads"]["effective"] == "off"
    assert _put(api, privateUploads="off").json()["privateUploads"]["source"] == "workspace"


def test_the_platform_off_switch_is_a_ceiling(api, monkeypatch):
    monkeypatch.setattr(config, PRIVATE_UPLOAD_SETTING, "off", raising=False)
    resp = _put(api, privateUploads="on")
    assert resp.status_code == 409 and "turned off on this platform" in resp.json()["detail"]
    assert api.db.get(ThemeWorkspaceSettings, WS_A) is None
    assert _put(api, privateUploads="off").status_code == 200


def test_another_workspaces_settings_never_show(api):
    api.db.add(ThemeWorkspaceSettings(workspace_id=WS_B, members_may_choose=False, private_uploads="on"))
    api.db.commit()
    body = _get(api)
    assert (body["membersMayChoose"], body["privateUploads"]["setting"]) == (True, None)


# ---------------------------------------------------------------------------
# Who may PUT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["viewer", "editor"])
def test_a_members_read_caller_reads_but_cannot_put(api, role):
    api.role = role
    body = _get(api)
    assert body["canInstall"] is False
    resp = _put(api, membersMayChoose=False)
    assert resp.status_code == 403, resp.text
    assert api.db.get(ThemeWorkspaceSettings, WS_A) is None


def test_the_local_editions_anonymous_owner_changes_settings(api, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    api.ctx = RequestContext(
        workspace_id=WS_A, auth_type="anonymous",
        user=UserContext(id="operator@local", email="operator@local", system_role="super_admin",
                         raw_claims={"source": LOCAL_OPERATOR_CLAIM_SOURCE, "user_id": OPERATOR_PK}),
    )
    api.role = None  # the anonymous lane never reads a member row
    resp = _put(api, membersMayChoose=False, installRole="owner")
    assert resp.status_code == 200, resp.text
    assert (resp.json()["membersMayChoose"], resp.json()["installRole"]) == (False, "owner")
    assert api.db.get(ThemeWorkspaceSettings, WS_A).updated_by == OPERATOR_PK


@pytest.mark.parametrize("body, status", [
    ({}, 400),
    ({"privateUploads": "sometimes"}, 400),
    ({"installRole": "editors"}, 400),
    ({"membersMayChoose": "yes"}, 422),
    ({"defaultInstallId": None}, 422),
    ({"pins": [{"installId": str(uuid.uuid4()), "version": "latest"}]}, 400),
])
def test_a_bad_body_is_refused_with_nothing_stored(api, body, status):
    resp = api.client.put(SETTINGS, json=body)
    assert resp.status_code == status, resp.text
    assert api.db.get(ThemeWorkspaceSettings, WS_A) is None


def test_a_concurrent_first_change_updates_the_row_instead_of_a_500(api, monkeypatch):
    api.db.add(ThemeWorkspaceSettings(workspace_id=WS_A, members_may_choose=False))
    api.db.commit()
    api.db.expunge_all()  # the row exists in the database, not in this request's session
    real_row, calls = settings_mod.settings_row, []

    def first_read_misses(*args):
        calls.append(args)
        return None if len(calls) == 1 else real_row(*args)

    monkeypatch.setattr(settings_mod, "settings_row", first_read_misses)
    row = settings_mod._writable_row(api.db, WS_A, MEMBER_PK)
    assert row.members_may_choose is False and len(calls) == 2
    assert api.db.query(ThemeWorkspaceSettings).count() == 1


# ---------------------------------------------------------------------------
# A pinned version per install (A-07b refuses the update)
# ---------------------------------------------------------------------------


def test_a_pinned_install_refuses_update_until_unpinned(api):
    v1, v2 = _package(api.db), _package(api.db, version="1.1.0")
    install = _install(api.db, v1, pending_package_id=v2.id)
    resp = _put(api, pins=[{"installId": str(install.id), "version": "1.0.0"}])
    assert resp.status_code == 200, resp.text
    assert resp.json()["pins"] == [{
        "installId": str(install.id), "themeId": "acme/ink", "version": "1.0.0",
        "pinnedVersion": "1.0.0", "pinnableVersions": ["1.0.0", "1.1.0"],
    }]
    update = f"/api/themes/installs/{install.id}/update"
    refused = api.client.post(update, json={"packageId": str(v2.id)})
    assert refused.status_code == 409 and "pinned" in refused.json()["detail"]
    assert _put(api, pins=[{"installId": str(install.id), "version": None}]).status_code == 200
    assert api.client.post(update, json={"packageId": str(v2.id)}).status_code == 200


def test_a_pin_names_a_version_the_install_holds(api):
    install = _install(api.db, _package(api.db))
    resp = _put(api, pins=[{"installId": str(install.id), "version": "9.9.9"}])
    assert resp.status_code == 409 and "isn't one this install holds" in resp.json()["detail"]
    pin = {"installId": str(install.id), "version": "1.0.0"}
    assert _put(api, pins=[pin, pin]).status_code == 400
    api.db.refresh(install)
    assert install.pinned_version is None


def test_pinning_another_workspaces_install_is_404(api):
    theirs = _install(api.db, _package(api.db), workspace_id=WS_B)
    foreign = _package(api.db, workspace_id=WS_B, theme_id="acme/pen")
    forged = _install(api.db, foreign)  # ours, but pointing at another workspace's private package
    for install in (theirs, forged):
        resp = _put(api, pins=[{"installId": str(install.id), "version": "1.0.0"}])
        assert resp.status_code == 404, resp.text
        api.db.refresh(install)
        assert install.pinned_version is None
    assert _get(api)["pins"] == []


def test_a_failed_pin_stores_none_of_the_other_changes(api):
    resp = _put(api, membersMayChoose=False, pins=[{"installId": str(uuid.uuid4()), "version": "1.0.0"}])
    assert resp.status_code == 404
    assert _get(api)["membersMayChoose"] is True


# ---------------------------------------------------------------------------
# Members may choose (A-07b enforces it)
# ---------------------------------------------------------------------------


def test_members_may_choose_off_refuses_a_choice_and_active_returns_the_default(api):
    default = _install(api.db, _package(api.db))
    assert api.client.put("/api/themes/default", json={"installId": str(default.id)}).status_code == 200
    assert api.client.put("/api/themes/choice", json={"choice": "classic"}).status_code == 200
    assert _put(api, membersMayChoose=False).status_code == 200
    api.role = "viewer"
    resp = api.client.put("/api/themes/choice", json={"choice": "studio"})
    assert resp.status_code == 403, resp.text
    assert api.client.get("/api/themes/active").json()["choice"] == str(default.id)
    body = _get(api)
    assert (body["membersMayChoose"], body["defaultInstallId"]) == (False, str(default.id))


def test_only_a_reviewed_theme_can_be_the_default_the_settings_report(api):
    private = _install(api.db, _package(api.db, workspace_id=WS_A))
    assert api.client.put("/api/themes/default", json={"installId": str(private.id)}).status_code == 409
    assert _get(api)["defaultInstallId"] is None


# ---------------------------------------------------------------------------
# Who may install: it only narrows, and only the owner moves it
# ---------------------------------------------------------------------------


def test_an_admin_cannot_make_installs_owner_only(api):
    api.role = "admin"
    resp = _put(api, installRole="owner")
    assert resp.status_code == 403 and "owner" in resp.json()["detail"]
    assert _get(api)["installRole"] == "admins"


def test_owner_only_installs_lock_admins_out_of_install_pin_and_the_role(api):
    install = _install(api.db, _package(api.db))
    assert _put(api, installRole="owner").json()["installRole"] == "owner"
    assert api.client.post("/probe/install").status_code == 200
    api.role = "admin"
    assert _get(api)["canInstall"] is False
    refused = api.client.post("/probe/install")
    assert refused.status_code == 403 and "the workspace owner" in refused.json()["detail"]
    assert _put(api, installRole="admins").status_code == 403
    assert _put(api, pins=[{"installId": str(install.id), "version": "1.0.0"}]).status_code == 403
    assert _put(api, membersMayChoose=False).status_code == 200  # still an admin's setting
    api.role = "owner"
    assert _put(api, installRole=None).json()["installRole"] == "admins"
    api.role = "admin"
    assert api.client.post("/probe/install").status_code == 200


def test_an_unknown_stored_install_role_reads_as_owner_only(api):
    api.db.add(ThemeWorkspaceSettings(workspace_id=WS_A, install_role="members:read"))
    api.db.commit()
    api.role = "admin"
    assert _get(api)["installRole"] == "owner"
    assert api.client.post("/probe/install").status_code == 403


EXPECTED_GATES = {("GET", SETTINGS): "members:read", ("PUT", SETTINGS): "workspace:manage"}


def _gates(dependant):
    found = []
    for sub in dependant.dependencies:
        marker = getattr(sub.call, permission_mod.PERMISSION_MARKER_ATTR, None)
        found += ([marker] if marker else []) + _gates(sub)
    return found


def test_every_route_is_a_plain_def_with_the_planned_gate():
    seen = {}
    for route in settings_api.router.routes:
        assert isinstance(route, APIRoute)
        assert not inspect.iscoroutinefunction(route.endpoint), route.path
        for method in route.methods:
            seen[(method, route.path)] = _gates(route.dependant)
    assert seen == {key: [gate] for key, gate in EXPECTED_GATES.items()}
