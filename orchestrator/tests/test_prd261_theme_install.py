"""PRD-261 A-07b — installed, active, choice, opt-in, default, compare, update, rollback, uninstall.

The real router (``api/themes_install.py``) runs on a mini FastAPI app over in-memory
SQLite holding the theme tables and a small ``users`` table, with the REAL
workspace-permission gate (only the caller's workspace role is stubbed) and the REAL
``resolve_user_pk``. Pins (plan §2.6 A-07b and the W-B2 review carry):

* another workspace's install answers 404 on every route, and so does an install whose
  ``package_id`` points at another workspace's private package (no DB constraint stops
  that row), and so does every opt-in, update, compare and rollback naming one;
* Q6: a private theme can never be the workspace default; a new private version needs a
  fresh opt-in; an opt-in for a superseded package is 409;
* update then rollback returns to the old package and keeps looks; uninstall is soft,
  keeps looks and moves members to the default; a pinned install refuses update;
* a ``disable`` revocation makes ``active`` answer the base; a stale ``clonedFrom`` shows
  ``needsUpdate``; ``builtin:play`` round-trips; members-may-choose off answers 403;
* the answer carries what ``frontend/lib/theme-host/manifest.ts`` parseActiveInstall reads;
* the local edition's anonymous owner works; manage routes refuse a viewer.
"""
from __future__ import annotations

import base64
import inspect
import os
import sys
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
from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.themes_install as install_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.models  # noqa: E402,F401  (registers every table the FKs name)
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
import services.themes.choices as choices_mod  # noqa: E402
from services.themes.contract import core_page_hashes  # noqa: E402
from services.themes.optin import OPT_IN_WORDING  # noqa: E402

WS_A = uuid.uuid4()
WS_B = uuid.uuid4()
MEMBER_PK, OTHER_PK, OPERATOR_PK = 7, 8, 1
TABLES = [m.__table__ for m in (
    ThemePackage, ThemeInstallation, ThemeChoice, ThemeLook, ThemeOptIn, ThemeWorkspaceSettings, ThemeRevocation,
)]
ENTRY_SHA = "ab" * 32
WORDING = "runs with your access to all your workspaces"


def _ctx(workspace_id=WS_A, clerk="clerk-7"):
    return RequestContext(
        workspace_id=workspace_id, user=UserContext(id=clerk, clerk_user_id=clerk, system_role="user"), auth_type="clerk",
    )


@pytest.fixture
def api(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=TABLES)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY, clerk_user_id TEXT, email TEXT)"))
        conn.execute(sa.text("INSERT INTO users (id, clerk_user_id) VALUES (7, 'clerk-7'), (8, 'clerk-8')"))
    session = sessionmaker(bind=engine)()
    state = SimpleNamespace(db=session, ctx=_ctx(), role="owner")
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: state.role)
    app = FastAPI()
    app.include_router(install_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        session.close()
        engine.dispose()


def _package(db, workspace_id=None, version="1.0.0", theme_id="acme/ink", base="studio", pages=None, **extra):
    owner = "workspace" if workspace_id else "marketplace"
    manifest = {"id": theme_id, "name": "Ink", "version": version, "base": base, "htmlClass": "ink"}
    if pages is not None:
        manifest["pages"] = pages
    row = ThemePackage(
        theme_id=theme_id, version=version, owner_type=owner, workspace_id=workspace_id,
        digest=uuid.uuid4().hex * 2, manifest=manifest, storage_prefix=f"themes/{theme_id}/{version}/",
        file_manifest={"dist/entry.js": {"type": "text/javascript", "size": 1, "sha256": ENTRY_SHA},
                       "dist/theme.css": {"type": "text/css", "size": 1, "sha256": "cd" * 32}},
        reviewed=extra.pop("reviewed", owner == "marketplace"),
        status=extra.pop("status", "approved" if owner == "marketplace" else "pending"), **extra,
    )
    db.add(row)
    db.commit()
    return row


def _install(db, package, workspace_id=WS_A, **extra):
    row = ThemeInstallation(workspace_id=workspace_id, theme_id=package.theme_id, package_id=package.id, **extra)
    db.add(row)
    db.commit()
    return row


def _active(api):
    resp = api.client.get("/api/themes/active")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _choose(api, choice):
    return api.client.put("/api/themes/choice", json={"choice": choice})


def _route_call(api, route, install, package_id):
    calls = {
        "choice": lambda: _choose(api, str(install.id)),
        "opt-in": lambda: api.client.post(f"/api/themes/installs/{install.id}/opt-in", json={"packageId": str(package_id)}),
        "default": lambda: api.client.put("/api/themes/default", json={"installId": str(install.id)}),
        "compare": lambda: api.client.get(f"/api/themes/installs/{install.id}/compare/{package_id}"),
        "update": lambda: api.client.post(f"/api/themes/installs/{install.id}/update", json={"packageId": str(package_id)}),
        "rollback": lambda: api.client.post(f"/api/themes/installs/{install.id}/rollback"),
        "uninstall": lambda: api.client.delete(f"/api/themes/installs/{install.id}"),
    }
    return calls[route]()


ROUTES = ("choice", "opt-in", "default", "compare", "update", "rollback", "uninstall")


# ---------------------------------------------------------------------------
# Tenancy: 404, never 403, for anything of another workspace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
def test_another_workspaces_install_is_404(api, route):
    theirs = _package(api.db)
    newer = _package(api.db, version="1.1.0")
    install = _install(api.db, theirs, workspace_id=WS_B, previous_package_id=newer.id)
    resp = _route_call(api, route, install, newer.id)
    assert resp.status_code == 404, resp.text
    assert api.client.get("/api/themes/installed").json() == {"installs": []}


@pytest.mark.parametrize("route", ROUTES)
def test_an_install_pointing_at_another_workspaces_private_package_is_404(api, route):
    """The W-B2 carry: no DB constraint stops this row, so every route checks the package."""
    foreign = _package(api.db, workspace_id=WS_B)
    newer = _package(api.db, workspace_id=WS_B, version="1.1.0")
    forged = _install(api.db, foreign, previous_package_id=newer.id)
    resp = _route_call(api, route, forged, newer.id)
    assert resp.status_code == 404, resp.text
    api.db.add(ThemeChoice(workspace_id=WS_A, user_id=MEMBER_PK, choice=str(forged.id)))
    api.db.commit()
    assert _active(api) == {"choice": "studio", "theme": None}
    assert api.client.get("/api/themes/installed").json() == {"installs": []}


@pytest.mark.parametrize("route", ("opt-in", "compare", "update", "rollback"))
def test_moving_an_install_to_another_workspaces_private_package_is_404(api, route):
    ours = _package(api.db)
    foreign = _package(api.db, workspace_id=WS_B, version="1.1.0")
    install = _install(api.db, ours, previous_package_id=foreign.id)
    resp = _route_call(api, route, install, foreign.id)
    assert resp.status_code == 404, resp.text
    api.db.refresh(install)
    assert install.package_id == ours.id
    assert api.db.query(ThemeOptIn).count() == 0


# ---------------------------------------------------------------------------
# Active: choice -> default -> Studio, in the shape the host reads
# ---------------------------------------------------------------------------


def test_builtin_play_round_trips_as_a_choice(api):
    assert _active(api) == {"choice": "studio", "theme": None}
    resp = _choose(api, "builtin:play")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"choice": "builtin:play", "theme": None}
    assert _active(api) == {"choice": "builtin:play", "theme": None}
    assert _choose(api, "classic").json()["choice"] == "classic"
    for bad in ("play", "theme:builtin:play", str(uuid.uuid4()).upper()):
        assert _choose(api, bad).status_code == 400, bad


def test_active_falls_back_from_the_choice_to_the_default_then_studio(api):
    package = _package(api.db)
    install = _install(api.db, package)
    assert api.client.put("/api/themes/default", json={"installId": str(install.id)}).status_code == 200
    assert _active(api)["choice"] == str(install.id)
    assert _choose(api, "classic").status_code == 200
    assert _active(api)["choice"] == "classic"
    api.ctx = _ctx(clerk="clerk-8")  # another member of the same workspace never sees that choice
    assert _active(api)["choice"] == str(install.id)
    assert api.client.put("/api/themes/default", json={"installId": None}).json() == {"defaultInstallId": None}
    assert _active(api) == {"choice": "studio", "theme": None}


def test_active_answers_what_parse_active_install_reads(api):
    package = _package(api.db)
    install = _install(api.db, package)
    _choose(api, str(install.id))
    theme = _active(api)["theme"]
    assert theme["installId"] == str(install.id)
    assert theme["digest"] == package.digest and len(theme["digest"]) == 64
    assert theme["manifest"]["id"] == "acme/ink"
    assert (theme["themeId"], theme["version"], theme["reviewed"], theme["revoked"]) == ("acme/ink", "1.0.0", True, None)
    assert theme["optIn"] == {"required": False, "wording": OPT_IN_WORDING, "packageId": str(package.id)}
    url = f"/api/themes/files/{package.digest}/dist/entry.js"
    assert theme["integrity"] == {url: "sha256-" + base64.b64encode(bytes.fromhex(ENTRY_SHA)).decode()}


def test_a_concurrent_first_choice_ends_stored_not_500(api, monkeypatch):
    # Another tab saved this member's first choice between our read and our insert.
    api.db.add(ThemeChoice(workspace_id=WS_A, user_id=MEMBER_PK, choice="classic"))
    api.db.commit()
    real_row, calls = choices_mod._choice_row, []

    def first_read_misses(*args):
        calls.append(args)
        return None if len(calls) == 1 else real_row(*args)

    monkeypatch.setattr(choices_mod, "_choice_row", first_read_misses)
    resp = _choose(api, "builtin:play")
    assert resp.status_code == 200, resp.text
    assert resp.json()["choice"] == "builtin:play"
    assert len(calls) >= 2  # the insert collided, then the row the other tab wrote was updated
    assert api.db.query(ThemeChoice).filter_by(workspace_id=WS_A, user_id=MEMBER_PK).count() == 1


def test_members_may_not_choose_answers_403_and_active_shows_the_default(api):
    default = _install(api.db, _package(api.db))
    api.client.put("/api/themes/default", json={"installId": str(default.id)})
    _choose(api, "classic")
    api.db.add(ThemeWorkspaceSettings(workspace_id=WS_A, members_may_choose=False))
    api.db.commit()
    assert _choose(api, "studio").status_code == 403
    assert _active(api)["choice"] == str(default.id)


# ---------------------------------------------------------------------------
# Q6: private code — opt-in per member per version, never a default
# ---------------------------------------------------------------------------


def test_a_private_theme_cannot_be_the_default(api):
    private = _install(api.db, _package(api.db, workspace_id=WS_A))
    resp = api.client.put("/api/themes/default", json={"installId": str(private.id)})
    assert resp.status_code == 409 and "unreviewed" in resp.json()["detail"]
    api.db.refresh(private)
    assert private.is_default is False
    reviewed = _install(api.db, _package(api.db, theme_id="acme/pen"))
    assert api.client.put("/api/themes/default", json={"installId": str(reviewed.id)}).status_code == 200


def test_a_new_private_version_needs_a_fresh_opt_in(api):
    v1 = _package(api.db, workspace_id=WS_A)
    install = _install(api.db, v1)
    _choose(api, str(install.id))
    opt_in = _active(api)["theme"]["optIn"]
    assert opt_in["required"] is True and WORDING in opt_in["wording"] and opt_in["packageId"] == str(v1.id)
    resp = api.client.post(f"/api/themes/installs/{install.id}/opt-in", json={"packageId": str(v1.id)})
    assert resp.status_code == 200 and resp.json()["optIn"]["required"] is False
    assert _active(api)["theme"]["optIn"]["required"] is False
    v2 = _package(api.db, workspace_id=WS_A, version="1.1.0")
    install.pending_package_id = v2.id
    api.db.commit()
    assert api.client.post(f"/api/themes/installs/{install.id}/update", json={"packageId": str(v2.id)}).status_code == 200
    after = _active(api)["theme"]["optIn"]
    assert after == {"required": True, "wording": OPT_IN_WORDING, "packageId": str(v2.id)}
    api.ctx = _ctx(clerk="clerk-8")  # an opt-in is one member's, never the workspace's
    api.db.add(ThemeChoice(workspace_id=WS_A, user_id=OTHER_PK, choice=str(install.id)))
    api.db.commit()
    assert _active(api)["theme"]["optIn"]["required"] is True


def test_an_opt_in_for_a_superseded_package_is_409(api):
    v1 = _package(api.db, workspace_id=WS_A)
    v2 = _package(api.db, workspace_id=WS_A, version="1.1.0")
    install = _install(api.db, v2, previous_package_id=v1.id)
    resp = api.client.post(f"/api/themes/installs/{install.id}/opt-in", json={"packageId": str(v1.id)})
    assert resp.status_code == 409
    assert api.db.query(ThemeOptIn).count() == 0


def test_a_reviewed_marketplace_theme_needs_no_opt_in(api):
    package = _package(api.db)
    install = _install(api.db, package)
    resp = api.client.post(f"/api/themes/installs/{install.id}/opt-in", json={"packageId": str(package.id)})
    assert resp.json()["optIn"]["required"] is False
    assert api.db.query(ThemeOptIn).count() == 0


# ---------------------------------------------------------------------------
# Versions: update, rollback, pin, compare, uninstall
# ---------------------------------------------------------------------------


def _look(api, theme_id="acme/ink"):
    api.db.add(ThemeLook(workspace_id=WS_A, user_id=MEMBER_PK, theme_id=theme_id, looks={"a": {}}, active={"id": "a"}))
    api.db.commit()


def test_update_then_rollback_returns_to_the_old_package_and_keeps_looks(api):
    v1, v2 = _package(api.db), _package(api.db, version="1.1.0")
    install = _install(api.db, v1, pending_package_id=v2.id)
    _look(api)
    resp = api.client.post(f"/api/themes/installs/{install.id}/update", json={"packageId": str(v2.id)})
    assert resp.status_code == 200, resp.text
    row = resp.json()
    assert (row["packageId"], row["previousPackageId"], row["pendingPackageId"]) == (str(v2.id), str(v1.id), None)
    resp = api.client.post(f"/api/themes/installs/{install.id}/rollback")
    assert resp.status_code == 200 and resp.json()["packageId"] == str(v1.id)
    assert resp.json()["previousPackageId"] == str(v2.id)
    assert api.db.query(ThemeLook).count() == 1
    other = _package(api.db, theme_id="acme/pen")
    assert api.client.post(f"/api/themes/installs/{install.id}/update", json={"packageId": str(other.id)}).status_code == 409


def test_a_pinned_install_refuses_update(api):
    v1, v2 = _package(api.db), _package(api.db, version="1.1.0")
    install = _install(api.db, v1, pinned_version="1.0.0")
    resp = api.client.post(f"/api/themes/installs/{install.id}/update", json={"packageId": str(v2.id)})
    assert resp.status_code == 409 and "pinned" in resp.json()["detail"]
    api.db.refresh(install)
    assert install.package_id == v1.id


def test_a_default_never_moves_to_unreviewed_code(api):
    v1 = _package(api.db)
    private = _package(api.db, workspace_id=WS_A, version="1.1.0")
    unapproved = _package(api.db, version="1.2.0", status="pending", reviewed=False)
    install = _install(api.db, v1, is_default=True)
    resp = api.client.post(f"/api/themes/installs/{install.id}/update", json={"packageId": str(private.id)})
    assert resp.status_code == 409 and "unreviewed" in resp.json()["detail"]
    resp = api.client.post(f"/api/themes/installs/{install.id}/update", json={"packageId": str(unapproved.id)})
    assert resp.status_code == 409 and "approved" in resp.json()["detail"]
    api.db.refresh(install)
    assert install.package_id == v1.id


def test_compare_lists_changed_files_and_the_scan_diff(api):
    v1 = _package(api.db, scan_verdict="pass", scan_report={"contrast": "ok"})
    v2 = _package(api.db, version="1.1.0", scan_verdict="review_required", scan_report={"contrast": "low"})
    v2.file_manifest = {"dist/entry.js": {"sha256": "ef" * 32}, "dist/new.js": {"sha256": ENTRY_SHA}}
    api.db.commit()
    install = _install(api.db, v1)
    body = api.client.get(f"/api/themes/installs/{install.id}/compare/{v2.id}").json()
    assert (body["from"]["version"], body["to"]["version"]) == ("1.0.0", "1.1.0")
    assert body["files"] == {"added": ["dist/new.js"], "removed": ["dist/theme.css"], "changed": ["dist/entry.js"]}
    assert body["scan"]["verdict"] == {"from": "pass", "to": "review_required"}
    assert body["scan"]["changed"] == {"contrast": {"from": "ok", "to": "low"}}


def test_uninstall_is_soft_keeps_looks_and_moves_members_to_the_default(api):
    default = _install(api.db, _package(api.db, theme_id="acme/pen"))
    api.client.put("/api/themes/default", json={"installId": str(default.id)})
    mine = _install(api.db, _package(api.db, workspace_id=WS_A))
    _choose(api, str(mine.id))
    _look(api)
    assert api.client.delete(f"/api/themes/installs/{mine.id}").status_code == 200
    api.db.refresh(mine)
    assert mine.uninstalled_at is not None
    assert api.db.query(ThemeLook).count() == 1
    assert _active(api)["choice"] == str(default.id)
    assert [row["installId"] for row in api.client.get("/api/themes/installed").json()["installs"]] == [str(default.id)]


# ---------------------------------------------------------------------------
# Revocations and drift
# ---------------------------------------------------------------------------


def test_a_disable_revocation_makes_active_return_the_base(api):
    package = _package(api.db, base="classic")
    install = _install(api.db, package)
    _choose(api, str(install.id))
    api.db.add(ThemeRevocation(theme_id="acme/ink", version_range="^2.0.0", level="disable", reason="later only"))
    api.db.add(ThemeRevocation(digest=package.digest, level="warn", reason="slow"))
    api.db.commit()
    assert _active(api)["theme"]["revoked"] == {"level": "warn", "reason": "slow"}
    api.db.add(ThemeRevocation(theme_id="acme/ink", level="disable", reason="leaks data"))
    api.db.commit()
    body = _active(api)
    assert (body["choice"], body["theme"]) == ("classic", None)
    assert body["revoked"] == {"installId": str(install.id), "level": "disable", "reason": "leaks data"}


def test_a_stale_clone_shows_needs_update(api):
    path = "app/chat/page.tsx"
    current = core_page_hashes()[path]
    stale = _package(api.db, pages={"/chat": {"mode": "clone", "clonedFrom": {"path": path, "sha": "0" * 64}}})
    fresh = _package(api.db, theme_id="acme/pen", pages={"/chat": {"mode": "clone", "clonedFrom": {"path": path, "sha": current}}})
    stale_install, fresh_install = _install(api.db, stale), _install(api.db, fresh)
    rows = {row["installId"]: row for row in api.client.get("/api/themes/installed").json()["installs"]}
    assert rows[str(stale_install.id)]["needsUpdate"] is True
    assert rows[str(fresh_install.id)]["needsUpdate"] is False
    _choose(api, str(stale_install.id))
    assert _active(api)["theme"]["needsUpdate"] is True


# ---------------------------------------------------------------------------
# Gates and editions
# ---------------------------------------------------------------------------


def test_manage_routes_refuse_a_viewer_and_member_routes_serve_one(api):
    v1, v2 = _package(api.db), _package(api.db, version="1.1.0")
    install = _install(api.db, v1, previous_package_id=v2.id)
    api.role = "viewer"
    for route in ("default", "compare", "update", "rollback", "uninstall"):
        assert _route_call(api, route, install, v2.id).status_code == 403, route
    assert _choose(api, str(install.id)).status_code == 200
    assert _active(api)["theme"]["installId"] == str(install.id)


def test_the_local_editions_anonymous_owner_chooses_and_sees_its_theme(api):
    install = _install(api.db, _package(api.db))
    api.ctx = RequestContext(
        workspace_id=WS_A, auth_type="anonymous",
        user=UserContext(id="operator@local", email="operator@local", system_role="super_admin",
                         raw_claims={"source": LOCAL_OPERATOR_CLAIM_SOURCE, "user_id": OPERATOR_PK}),
    )
    api.role = None  # the anonymous lane never reads a member row
    assert _choose(api, str(install.id)).status_code == 200
    assert _active(api)["theme"]["installId"] == str(install.id)
    assert api.db.query(ThemeChoice).one().user_id == OPERATOR_PK
    assert api.client.put("/api/themes/default", json={"installId": str(install.id)}).status_code == 200


EXPECTED_GATES = {
    ("GET", "/api/themes/installed"): "members:read",
    ("GET", "/api/themes/active"): "members:read",
    ("PUT", "/api/themes/choice"): "members:read",
    ("POST", "/api/themes/installs/{install_id}/opt-in"): "members:read",
    ("PUT", "/api/themes/default"): "workspace:manage",
    ("GET", "/api/themes/installs/{install_id}/compare/{package_id}"): "workspace:manage",
    ("POST", "/api/themes/installs/{install_id}/update"): "workspace:manage",
    ("POST", "/api/themes/installs/{install_id}/rollback"): "workspace:manage",
    ("DELETE", "/api/themes/installs/{install_id}"): "workspace:manage",
}


def _gates(dependant):
    found = []
    for sub in dependant.dependencies:
        marker = getattr(sub.call, permission_mod.PERMISSION_MARKER_ATTR, None)
        found += ([marker] if marker else []) + _gates(sub)
    return found


def test_every_route_is_a_plain_def_with_the_planned_gate():
    seen = {}
    for route in install_api.router.routes:
        assert isinstance(route, APIRoute)
        assert not inspect.iscoroutinefunction(route.endpoint), route.path
        for method in route.methods:
            seen[(method, route.path)] = _gates(route.dependant)
    assert seen == {key: [gate] for key, gate in EXPECTED_GATES.items()}
