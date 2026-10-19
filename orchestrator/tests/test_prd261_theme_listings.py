"""PRD-261 C-04a — the Themes tab's listings: list, detail and install by reference.

The real routers (``api/marketplace_themes.py`` and A-07b's ``api/themes_install.py``) run on
a mini FastAPI app over in-memory SQLite holding the theme tables and a small ``users`` table,
with the REAL workspace-permission gate (only the caller's workspace role is stubbed), the REAL
"who may install" gate, the REAL step-up dependency and its handler. Pins (plan §2.6 C-04a):

* a second workspace installs the approved Play listing, and both installs reference one
  package row; update to a newer approved version (A-07b), then rollback;
* only approved, reviewed Marketplace versions are listed: never pending, rejected,
  deactivated, unreviewed, a workspace's private upload, or a ``disable``-revoked version;
* the detail carries previews per preset, publisher, version, ``uses`` and the versions map,
  and install takes the newest version this host can run;
* install answers the same install again (200), refuses another version or a private upload
  of the theme (409), and is gated by manage, "who may install", the switch and step-up.
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
from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.marketplace_themes as listings_api  # noqa: E402
import api.themes_install as install_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.models  # noqa: E402,F401  (registers every table the FKs name)
import services.themes.listings as listings_mod  # noqa: E402
from services.themes import review  # noqa: E402
from services.themes.listing_updates import offer_listed_update  # noqa: E402
from config import config  # noqa: E402
from core.auth import step_up  # noqa: E402
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
    ThemePublisher,
    ThemeReviewEvent,
    ThemeRevocation,
    ThemeWorkspaceSettings,
)
from services.themes.serve import module_url  # noqa: E402

WS_A, WS_B = uuid.uuid4(), uuid.uuid4()
MEMBER_PK, OPERATOR_PK = 7, 1
PLAY = "automatos/play"
PLAY_LISTING = "automatos.play"
TABLES = [m.__table__ for m in (
    ThemePackage, ThemeInstallation, ThemeChoice, ThemeLook, ThemeOptIn, ThemeWorkspaceSettings, ThemeRevocation,
    ThemePublisher, ThemeReviewEvent,
)]
ENGINES = {"automatos": "^1.0.0", "sdk": "^1.0.0"}
FUTURE_ENGINES = {"automatos": "^2.0.0", "sdk": "^1.0.0"}
PRESETS = ("ember", "ember-dark")
PREVIEWS = ("assets/previews/ember.png", "assets/previews/ember-chat.png",
            "assets/previews/ember-dark-chat.png", "assets/previews/team.png")


def _ctx(workspace_id=WS_A, fva=(1, -1)):
    """A Clerk member who signed in a minute ago (``fva`` None: never re-verified)."""
    claims = {"sub": "clerk-7", "iat": int(time.time())}
    if fva is not None:
        claims["fva"] = list(fva)
    user = UserContext(id="clerk-7", clerk_user_id="clerk-7", system_role="user", raw_claims=claims)
    return RequestContext(workspace_id=workspace_id, user=user, auth_type="clerk")


@pytest.fixture
def api(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=TABLES)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY, clerk_user_id TEXT, email TEXT)"))
        conn.execute(sa.text("INSERT INTO users (id, clerk_user_id) VALUES (7, 'clerk-7')"))
    session = sessionmaker(bind=engine)()
    state = SimpleNamespace(db=session, ctx=_ctx(), role="owner")
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: state.role)
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", "on")
    app = FastAPI()
    step_up.install_step_up_handler(app)
    app.include_router(listings_api.router)
    app.include_router(install_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        session.close()
        engine.dispose()


def _file(sha="ab"):
    return {"type": "text/javascript", "size": 1, "sha256": sha * 32}


def _package(db, version="1.0.0", theme_id=PLAY, status="approved", reviewed=None, workspace_id=None,
             engines=None, **manifest_extra):
    owner = "workspace" if workspace_id else "marketplace"
    manifest = {
        "id": theme_id, "name": "Play", "version": version, "base": "studio", "htmlClass": "play",
        "engines": engines or ENGINES, "tone": "from-look", "minWidth": 1024, "look": "ember",
        "uses": ["chat", "looks"], "previews": list(PREVIEWS), "description": "A playful theme.", **manifest_extra,
    }
    files = {"dist/entry.js": _file(), "README.md": _file("cd")}
    files.update({f"looks/presets/{preset}.look.json": _file("ef") for preset in PRESETS})
    files.update({path: _file("12") for path in PREVIEWS})
    row = ThemePackage(
        theme_id=theme_id, version=version, owner_type=owner, workspace_id=workspace_id,
        digest=uuid.uuid4().hex * 2, manifest=manifest, file_manifest=files,
        storage_prefix=f"themes/{theme_id}/{version}/", status=status,
        reviewed=(status == "approved" and owner == "marketplace") if reviewed is None else reviewed,
    )
    db.add(row)
    db.commit()
    return row


def _install(api, listing=PLAY_LISTING, **body):
    kwargs = {"json": {k: str(v) for k, v in body.items()}} if body else {}
    return api.client.post(f"/api/marketplace/themes/{listing}/install", **kwargs)


def _listings(api):
    resp = api.client.get("/api/marketplace/themes")
    assert resp.status_code == 200, resp.text
    return {row["listingId"]: row for row in resp.json()["listings"]}


# ---------------------------------------------------------------------------
# The plan's proofs
# ---------------------------------------------------------------------------


def test_a_second_workspace_installs_the_approved_play_listing_and_both_reference_one_package(api):
    play = _package(api.db)
    first = _install(api)
    assert first.status_code == 201, first.text
    api.ctx = _ctx(WS_B)
    second = _install(api)
    assert second.status_code == 201, second.text
    installs = api.db.query(ThemeInstallation).order_by(ThemeInstallation.workspace_id).all()
    assert {row.workspace_id for row in installs} == {WS_A, WS_B}
    assert {row.package_id for row in installs} == {play.id}
    assert api.db.query(ThemePackage).filter(ThemePackage.theme_id == PLAY).count() == 1
    body = second.json()
    assert body["created"] is True
    assert body["install"]["packageId"] == str(play.id) and body["install"]["ownerType"] == "marketplace"
    assert body["install"]["optIn"]["required"] is False
    assert api.client.get("/api/themes/installed").json()["installs"][0]["installId"] == body["install"]["installId"]


def test_update_to_a_newer_approved_version_then_rollback(api):
    old = _package(api.db, "1.0.0")
    new = _package(api.db, "1.1.0")
    resp = _install(api, packageId=old.id)
    assert resp.status_code == 201, resp.text
    install_id = resp.json()["install"]["installId"]
    installed = api.client.get(f"/api/marketplace/themes/{PLAY_LISTING}").json()["installed"]
    assert installed == {"installId": install_id, "packageId": str(old.id), "version": "1.0.0",
                         "fromListing": True, "updatePackageId": str(new.id)}
    moved = api.client.post(f"/api/themes/installs/{install_id}/update", json={"packageId": str(new.id)})
    assert moved.status_code == 200, moved.text
    assert moved.json()["version"] == "1.1.0" and moved.json()["previousPackageId"] == str(old.id)
    assert _listings(api)[PLAY_LISTING]["installed"]["updatePackageId"] is None
    back = api.client.post(f"/api/themes/installs/{install_id}/rollback")
    assert back.status_code == 200, back.text
    assert back.json()["version"] == "1.0.0" and back.json()["previousPackageId"] == str(new.id)
    assert api.db.query(ThemeInstallation).count() == 1


# ---------------------------------------------------------------------------
# "Update available" on the installed-themes list (pending_package_id)
# ---------------------------------------------------------------------------


def _installed(api):
    resp = api.client.get("/api/themes/installed")
    assert resp.status_code == 200, resp.text
    return {row["themeId"]: row for row in resp.json()["installs"]}


def test_installing_an_older_version_offers_the_newest_on_the_installed_list(api):
    old = _package(api.db, "1.0.0")
    new = _package(api.db, "1.1.0")
    _package(api.db, "1.2.0", engines=FUTURE_ENGINES)
    assert _install(api, packageId=old.id).status_code == 201
    row = _installed(api)[PLAY]
    assert row["pendingPackageId"] == str(new.id) and row["pendingVersion"] == "1.1.0"


def test_installing_the_newest_version_offers_nothing(api):
    _package(api.db, "1.0.0")
    _package(api.db, "1.1.0")
    assert _install(api).status_code == 201
    assert _installed(api)[PLAY]["pendingPackageId"] is None


def test_a_version_approved_later_is_offered_to_listing_installs_only(api):
    old = _package(api.db, "1.0.0")
    assert _install(api, packageId=old.id).status_code == 201
    private = _package(api.db, "0.5.0", workspace_id=WS_B)
    api.db.add(ThemeInstallation(workspace_id=WS_B, theme_id=PLAY, package_id=private.id))
    api.db.commit()
    new = _package(api.db, "1.1.0")
    assert offer_listed_update(api.db, new) == 1
    api.db.commit()
    assert _installed(api)[PLAY]["pendingPackageId"] == str(new.id)
    api.ctx = _ctx(WS_B)
    assert _installed(api)[PLAY]["pendingPackageId"] is None


@pytest.mark.parametrize("kind", ["pending", "unreviewed", "incompatible", "private"])
def test_a_version_that_is_not_listed_is_offered_to_no_one(api, kind):
    old = _package(api.db, "1.0.0")
    assert _install(api, packageId=old.id).status_code == 201
    extra = {
        "pending": {"status": "pending"},
        "unreviewed": {"reviewed": False},
        "incompatible": {"engines": FUTURE_ENGINES},
        "private": {"workspace_id": WS_A},
    }[kind]
    assert offer_listed_update(api.db, _package(api.db, "1.1.0", **extra)) == 0
    assert _installed(api)[PLAY]["pendingPackageId"] is None


def test_an_install_waiting_on_a_newer_version_keeps_it_and_a_newer_install_is_not_offered_an_older_one(api):
    old = _package(api.db, "1.0.0")
    newest = _package(api.db, "1.2.0")
    assert _install(api, packageId=old.id).status_code == 201
    api.ctx = _ctx(WS_B)
    assert _install(api, packageId=newest.id).status_code == 201
    middle = _package(api.db, "1.1.0")
    assert offer_listed_update(api.db, middle) == 0
    api.db.commit()
    assert _installed(api)[PLAY]["pendingPackageId"] is None
    api.ctx = _ctx(WS_A)
    assert _installed(api)[PLAY]["pendingPackageId"] == str(newest.id)


def _offered(api, package):
    offered = offer_listed_update(api.db, package)
    api.db.commit()
    return offered


def test_deactivating_an_offered_version_withdraws_the_offer_and_a_lower_fix_is_offered(api):
    old = _package(api.db, "1.0.0")
    assert _install(api, packageId=old.id).status_code == 201
    broken = _package(api.db, "1.1.0")
    assert _offered(api, broken) == 1
    review.deactivate(api.db, broken.id, "sends tokens off-site", OPERATOR_PK)
    api.db.commit()
    assert _installed(api)[PLAY]["pendingPackageId"] is None

    fix = _package(api.db, "1.0.1")
    assert _offered(api, fix) == 1
    row = _installed(api)[PLAY]
    assert row["pendingPackageId"] == str(fix.id) and row["pendingVersion"] == "1.0.1"


def test_deactivating_the_newest_offer_falls_back_to_the_newest_listed_one_still_newer(api):
    old = _package(api.db, "1.0.0")
    assert _install(api, packageId=old.id).status_code == 201
    middle = _package(api.db, "1.1.0")
    assert _offered(api, middle) == 1
    newest = _package(api.db, "1.2.0")
    assert _offered(api, newest) == 1
    review.deactivate(api.db, newest.id, "broken", OPERATOR_PK)
    api.db.commit()
    assert _installed(api)[PLAY]["pendingPackageId"] == str(middle.id)


def test_a_pending_version_no_longer_listed_never_blocks_a_lower_fix(api):
    old = _package(api.db, "1.0.0")
    assert _install(api, packageId=old.id).status_code == 201
    gone = _package(api.db, "1.1.0")
    assert _offered(api, gone) == 1
    gone.status = "deactivated"  # taken off without its offers withdrawn (data from before withdraw_offers)
    api.db.commit()

    fix = _package(api.db, "1.0.1")
    assert _offered(api, fix) == 1
    assert _installed(api)[PLAY]["pendingPackageId"] == str(fix.id)


# ---------------------------------------------------------------------------
# Approved versions only
# ---------------------------------------------------------------------------


def test_the_list_shows_approved_reviewed_marketplace_versions_only(api):
    _package(api.db, "1.0.0")
    _package(api.db, "2.0.0", status="pending")
    for theme_id, status, reviewed in (("acme/pending", "pending", None), ("acme/rejected", "rejected", None),
                                       ("acme/gone", "deactivated", None), ("acme/unreviewed", "approved", False)):
        _package(api.db, theme_id=theme_id, status=status, reviewed=reviewed)
    _package(api.db, theme_id="acme/private", workspace_id=WS_A)
    revoked = _package(api.db, theme_id="acme/revoked")
    api.db.add(ThemeRevocation(digest=revoked.digest, level="disable", reason="Withdrawn."))
    api.db.commit()
    listed = _listings(api)
    assert set(listed) == {PLAY_LISTING}
    assert listed[PLAY_LISTING]["version"] == "1.0.0"
    for listing in ("acme.pending", "acme.rejected", "acme.gone", "acme.unreviewed", "acme.private", "acme.revoked"):
        assert api.client.get(f"/api/marketplace/themes/{listing}").status_code == 404
        assert _install(api, listing).status_code == 404


@pytest.mark.parametrize("listing", ["submit", "automatos.play.x", "Automatos.play", "automatos.", "nope.nope"])
def test_a_listing_id_that_names_no_listing_is_404(api, listing):
    _package(api.db)
    assert api.client.get(f"/api/marketplace/themes/{listing}").status_code == 404
    assert _install(api, listing).status_code == 404


def test_installing_a_version_that_is_not_listed_by_id_is_404(api):
    _package(api.db, "1.0.0")
    pending = _package(api.db, "1.1.0", status="pending")
    private = _package(api.db, "1.2.0", workspace_id=WS_A)
    for package in (pending, private):
        resp = _install(api, packageId=package.id)
        assert resp.status_code == 404 and resp.json()["detail"] == listings_mod.VERSION_NOT_FOUND
    assert _install(api, packageId=uuid.uuid4()).status_code == 404
    assert api.db.query(ThemeInstallation).count() == 0


def test_a_warned_version_is_listed_with_its_reason(api):
    play = _package(api.db)
    api.db.add(ThemeRevocation(theme_id=PLAY, level="warn", reason="Contrast issue in one preset."))
    api.db.commit()
    detail = api.client.get(f"/api/marketplace/themes/{PLAY_LISTING}").json()
    assert detail["revoked"] == {"level": "warn", "reason": "Contrast issue in one preset."}
    assert detail["versions"]["1.0.0"]["revoked"]["level"] == "warn"
    assert _install(api, packageId=play.id).status_code == 201


# ---------------------------------------------------------------------------
# The detail and the versions map
# ---------------------------------------------------------------------------


def test_the_detail_shows_previews_per_preset_publisher_uses_and_versions(api):
    play = _package(api.db, "1.0.0")
    newer = _package(api.db, "1.1.0", pricing="Pro")
    detail = api.client.get(f"/api/marketplace/themes/{PLAY_LISTING}").json()
    assert detail["listingId"] == PLAY_LISTING and detail["themeId"] == PLAY
    assert detail["version"] == "1.1.0" and detail["packageId"] == str(newer.id) and detail["latest"] == "1.1.0"
    assert detail["publisher"] == {"name": "automatos", "official": True}
    assert detail["uses"] == ["chat", "looks"] and detail["pricing"] == "Pro" and detail["compatible"] is True
    url = lambda path: module_url(newer.digest, path)  # noqa: E731
    assert detail["presets"] == [
        {"id": "ember", "default": True, "previews": [{"screen": "chat", "url": url(PREVIEWS[1])},
                                                     {"screen": None, "url": url(PREVIEWS[0])}]},
        {"id": "ember-dark", "default": False, "previews": [{"screen": "chat", "url": url(PREVIEWS[2])}]},
    ]
    assert detail["otherPreviews"] == [url(PREVIEWS[3])]
    assert detail["readme"] == url("README.md") and detail["changelog"] is None
    assert detail["cover"] == url(PREVIEWS[1])
    assert set(detail["versions"]) == {"1.0.0", "1.1.0"}
    assert detail["versions"]["1.0.0"]["packageId"] == str(play.id)
    assert detail["versions"]["1.0.0"]["engines"] == ENGINES and detail["versions"]["1.0.0"]["compatible"] is True
    assert detail["installed"] is None


def test_an_unclaimed_publisher_is_not_official_and_the_price_label_defaults_to_free(api):
    _package(api.db, theme_id="acme/ink")
    api.db.add(ThemePublisher(publisher="acme", owner_user_id=MEMBER_PK))
    api.db.commit()
    card = _listings(api)["acme.ink"]
    assert card["publisher"] == {"name": "acme", "official": False}
    assert card["pricing"] == listings_mod.FREE_LABEL


def test_install_takes_the_newest_version_this_host_runs(api):
    runs = _package(api.db, "1.0.0")
    future = _package(api.db, "2.0.0", engines=FUTURE_ENGINES)
    detail = api.client.get(f"/api/marketplace/themes/{PLAY_LISTING}").json()
    assert detail["version"] == "1.0.0" and detail["latest"] == "1.0.0"
    assert detail["versions"]["2.0.0"]["compatible"] is False
    refused = _install(api, packageId=future.id)
    assert refused.status_code == 409 and "Choose a version marked compatible" in refused.json()["detail"]
    resp = _install(api)
    assert resp.status_code == 201 and resp.json()["install"]["packageId"] == str(runs.id)


def test_a_listing_with_no_version_this_host_runs_is_shown_but_not_installable(api):
    future = _package(api.db, "2.0.0", engines=FUTURE_ENGINES)
    card = _listings(api)[PLAY_LISTING]
    assert card["compatible"] is False and card["packageId"] == str(future.id)
    resp = _install(api)
    assert resp.status_code == 409 and resp.json()["detail"] == listings_mod.NO_COMPATIBLE


# ---------------------------------------------------------------------------
# Installing again, and a theme installed another way
# ---------------------------------------------------------------------------


def test_installing_the_same_version_again_answers_that_install(api):
    _package(api.db)
    first = _install(api).json()["install"]["installId"]
    again = _install(api)
    assert again.status_code == 200 and again.json() == {"created": False, "install": again.json()["install"]}
    assert again.json()["install"]["installId"] == first
    assert api.db.query(ThemeInstallation).count() == 1


def test_installing_another_version_of_an_installed_theme_points_at_update(api):
    old = _package(api.db, "1.0.0")
    _package(api.db, "1.1.0")
    assert _install(api, packageId=old.id).status_code == 201
    resp = _install(api)
    assert resp.status_code == 409 and "Update it from Settings" in resp.json()["detail"]


def test_a_workspace_running_its_own_upload_of_the_theme_must_uninstall_it_first(api):
    _package(api.db, theme_id="acme/ink")
    private = _package(api.db, "0.9.0", theme_id="acme/ink", workspace_id=WS_A)
    api.db.add(ThemeInstallation(workspace_id=WS_A, theme_id="acme/ink", package_id=private.id))
    api.db.commit()
    assert _listings(api)["acme.ink"]["installed"]["fromListing"] is False
    resp = _install(api, "acme.ink")
    assert resp.status_code == 409 and "Uninstall it first" in resp.json()["detail"]
    api.ctx = _ctx(WS_B)
    assert _listings(api)["acme.ink"]["installed"] is None


def test_an_install_that_lost_a_race_answers_the_winner(api, monkeypatch):
    play = _package(api.db)
    api.db.add(ThemeInstallation(workspace_id=WS_A, theme_id=PLAY, package_id=play.id))
    api.db.commit()
    real = listings_mod.active_installation
    calls = []

    def blind_first(db, workspace_id, theme_id, lock=False):
        calls.append(lock)
        return None if len(calls) == 1 else real(db, workspace_id, theme_id, lock)

    monkeypatch.setattr(listings_mod, "active_installation", blind_first)
    resp = _install(api)
    assert resp.status_code == 200, resp.text
    assert resp.json()["created"] is False and api.db.query(ThemeInstallation).count() == 1


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def test_a_viewer_reads_listings_but_cannot_install(api):
    _package(api.db)
    api.role = "viewer"
    assert api.client.get("/api/marketplace/themes").status_code == 200
    assert api.client.get(f"/api/marketplace/themes/{PLAY_LISTING}").status_code == 200
    resp = _install(api)
    assert resp.status_code == 403 and "clerk_error" not in resp.json()
    api.role = None
    assert api.client.get("/api/marketplace/themes").status_code == 403
    assert api.db.query(ThemeInstallation).count() == 0


def test_an_owner_only_workspace_refuses_an_admins_install(api):
    _package(api.db)
    api.db.add(ThemeWorkspaceSettings(workspace_id=WS_A, install_role="workspace:delete"))
    api.db.commit()
    api.role = "admin"
    resp = _install(api)
    assert resp.status_code == 403 and "only the workspace owner" in resp.json()["detail"]
    api.role = "owner"
    assert _install(api).status_code == 201


@pytest.mark.parametrize("switch", ["off", "maybe"])
def test_the_marketplace_switch_off_refuses_every_route(api, monkeypatch, switch):
    _package(api.db)
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", switch)
    for resp in (api.client.get("/api/marketplace/themes"), api.client.get(f"/api/marketplace/themes/{PLAY_LISTING}"),
                 _install(api)):
        assert resp.status_code == 403 and resp.json()["detail"] == listings_mod.MARKETPLACE_OFF
    assert api.db.query(ThemeInstallation).count() == 0


@pytest.mark.parametrize("fva", [None, (-1, -1), (config.STEP_UP_MAX_AGE_MINUTES + 5, -1)])
def test_a_saas_install_without_a_fresh_step_up_is_refused(api, monkeypatch, fva):
    _package(api.db)
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    api.ctx = _ctx(fva=fva)
    resp = _install(api)
    assert resp.status_code == 403, resp.text
    assert resp.json()["clerk_error"]["metadata"]["reverification"]["level"] == "first_factor"
    assert api.db.query(ThemeInstallation).count() == 0


def test_the_local_editions_anonymous_owner_installs_a_listing(api, monkeypatch):
    play = _package(api.db)
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    api.ctx = RequestContext(
        workspace_id=WS_A, auth_type="anonymous",
        user=UserContext(id="operator@local", email="operator@local", system_role="super_admin",
                         raw_claims={"source": LOCAL_OPERATOR_CLAIM_SOURCE, "user_id": OPERATOR_PK}),
    )
    api.role = None
    resp = _install(api)
    assert resp.status_code == 201, resp.text
    row = api.db.query(ThemeInstallation).one()
    assert row.package_id == play.id and row.installed_by == OPERATOR_PK


def test_the_route_table_gates_install_with_manage_who_installs_and_step_up_after_the_gate():
    def calls(dependant):
        found = []
        for sub in dependant.dependencies:
            found += [sub.call] + calls(sub)
        return found

    def index(found, predicate):
        return next((n for n, call in enumerate(found) if predicate(call)), None)

    table = {}
    for route in listings_api.router.routes:
        assert isinstance(route, APIRoute) and not inspect.iscoroutinefunction(route.endpoint)
        for method in route.methods - {"HEAD", "OPTIONS"}:
            table[(method, route.path)] = calls(route.dependant)
    install = ("POST", "/api/marketplace/themes/{listing_id}/install")
    assert set(table) == {("GET", "/api/marketplace/themes"), ("GET", "/api/marketplace/themes/{listing_id}"), install}
    for key, found in table.items():
        gate = index(found, lambda call: getattr(call, permission_mod.PERMISSION_MARKER_ATTR, None) is not None)
        stepped = index(found, lambda call: getattr(call, step_up.STEP_UP_MARKER_ATTR, None) is not None)
        role = index(found, lambda call: "require_install_role" in getattr(call, "__qualname__", ""))
        assert gate is not None, key
        assert (stepped is not None, role is not None) == (key == install, key == install), key
        if key == install:
            assert gate < role < stepped
            assert getattr(found[stepped], step_up.STEP_UP_MARKER_ATTR) == config.STEP_UP_MAX_AGE_MINUTES


@pytest.mark.parametrize("theme_id", ["automatos/play", "acme/ink-2", "a/b"])
def test_listing_ids_round_trip(theme_id):
    listing_id = listings_mod.listing_id_of(theme_id)
    assert "/" not in listing_id and listings_mod.theme_id_of(listing_id) == theme_id
