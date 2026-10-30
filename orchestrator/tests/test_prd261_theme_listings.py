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
  and install takes the newest version this host can run; previews keep theme.json's order, and
  the cover is the first preview it lists, not the first by name;
* install answers the same install again (200), refuses another version or a private upload
  of the theme (409), and is gated by manage, "who may install", the switch and step-up;
* (C-05b, QB-21) ``publisher.official`` is true only for a reserved publisher whose shown
  version's signature an active key verifies (the keys path points at an RFC 8032 public key,
  whose published seed signs here); a local operator's own approval of ``automatos/x`` is never
  Official; the card, the detail and every versions-map entry carry ``signature {state, keyId}``;
  a version whose key was retired is Official again once an active key's signature is pasted (QB-28);
* (C-05c, QB-28) a version whose signature fails, or whose key was revoked, is not listed at all
  (``tests/test_prd261_signature_on_load.py`` pins the rest of that disable); a retired key's is.
"""
from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

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
from services.themes import review, signature_verify, signing  # noqa: E402
from services.themes.digest import package_digest  # noqa: E402
from services.themes.listing_updates import offer_listed_update  # noqa: E402
from services.themes.review_stage import VERDICT_PASS  # noqa: E402
from config import config  # noqa: E402
from core.auth import step_up  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.principal import LOCAL_OPERATOR_CLAIM_SOURCE  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.themes import (  # noqa: E402
    THEME_SCAN_VERDICT_DEFAULT,
    ThemeChoice,
    ThemeInstallation,
    ThemeLook,
    ThemeOptIn,
    ThemePackage,
    ThemePublisher,
    ThemeReviewEvent,
    ThemeRevocation,
    ThemeRevocationSync,
    ThemeWorkspaceSettings,
)
from services.themes.serve import module_url  # noqa: E402

WS_A, WS_B = uuid.uuid4(), uuid.uuid4()
MEMBER_PK, OPERATOR_PK = 7, 1
PLAY = "automatos/play"
PLAY_LISTING = "automatos.play"
TABLES = [m.__table__ for m in (
    ThemePackage, ThemeInstallation, ThemeChoice, ThemeLook, ThemeOptIn, ThemeWorkspaceSettings, ThemeRevocation,
    ThemePublisher, ThemeReviewEvent, ThemeRevocationSync,  # C-05c reads the feed's revoked keys in the local edition
)]
ENGINES = {"automatos": "^1.0.0", "sdk": "^1.0.0"}
FUTURE_ENGINES = {"automatos": "^2.0.0", "sdk": "^1.0.0"}
PRESETS = ("ember", "ember-dark")
PREVIEWS = ("assets/previews/ember.png", "assets/previews/ember-chat.png",
            "assets/previews/ember-dark-chat.png", "assets/previews/team.png")
VECTOR = json.loads((_ORCH / "tests" / "fixtures" / "themes" / "signing-vector.json").read_text(encoding="utf-8"))
RFC_KEY, OTHER_RFC_KEY = VECTOR["ed25519"]["cases"][:2]
UNSIGNED = {"state": "unsigned", "keyId": None}


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
             engines=None, scan_verdict=THEME_SCAN_VERDICT_DEFAULT, **manifest_extra):
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
        storage_prefix=f"themes/{theme_id}/{version}/", status=status, scan_verdict=scan_verdict,
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
    assert detail["publisher"] == {"name": "automatos", "official": False}  # reserved, but unsigned (QB-21)
    assert detail["signature"] == UNSIGNED and detail["versions"]["1.0.0"]["signature"] == UNSIGNED
    assert detail["uses"] == ["chat", "looks"] and detail["pricing"] == "Pro" and detail["compatible"] is True
    url = lambda path: module_url(newer.digest, path)  # noqa: E731
    assert detail["presets"] == [
        {"id": "ember", "default": True, "previews": [{"screen": None, "url": url(PREVIEWS[0])},
                                                     {"screen": "chat", "url": url(PREVIEWS[1])}],
         "phonePreviews": []},
        {"id": "ember-dark", "default": False, "previews": [{"screen": "chat", "url": url(PREVIEWS[2])}],
         "phonePreviews": []},
    ]
    assert detail["otherPreviews"] == [url(PREVIEWS[3])]
    assert detail["readme"] == url("README.md") and detail["changelog"] is None
    assert detail["cover"] == url(PREVIEWS[0])
    assert set(detail["versions"]) == {"1.0.0", "1.1.0"}
    assert detail["versions"]["1.0.0"]["packageId"] == str(play.id)
    assert detail["versions"]["1.0.0"]["engines"] == ENGINES and detail["versions"]["1.0.0"]["compatible"] is True
    assert detail["installed"] is None


def test_the_cover_is_the_first_preview_theme_json_lists_not_the_first_by_name(api):
    # As Play lists ember.png first though aurora.png sorts before it: the cover is the theme's pick.
    order = [PREVIEWS[3], PREVIEWS[1], PREVIEWS[0]]
    play = _package(api.db, previews=order)
    url = lambda path: module_url(play.digest, path)  # noqa: E731
    assert _listings(api)[PLAY_LISTING]["cover"] == url(PREVIEWS[3])
    detail = api.client.get(f"/api/marketplace/themes/{PLAY_LISTING}").json()
    assert detail["cover"] == url(PREVIEWS[3])
    assert detail["presets"][0]["previews"] == [{"screen": "chat", "url": url(PREVIEWS[1])},
                                                {"screen": None, "url": url(PREVIEWS[0])}]
    assert detail["otherPreviews"] == [url(PREVIEWS[3])]


def test_an_unclaimed_publisher_is_not_official_and_the_price_label_defaults_to_free(api):
    _package(api.db, theme_id="acme/ink")
    api.db.add(ThemePublisher(publisher="acme", owner_user_id=MEMBER_PK))
    api.db.commit()
    card = _listings(api)["acme.ink"]
    assert card["publisher"] == {"name": "acme", "official": False}
    assert card["pricing"] == listings_mod.FREE_LABEL


# ---------------------------------------------------------------------------
# Official needs a verified signature (C-05b, QB-21)
# ---------------------------------------------------------------------------


def _keys(monkeypatch, tmp_path, status="active"):
    """Point the verifier at a keys file holding the RFC 8032 TEST 1 public key."""
    entry = {"keyId": RFC_KEY["keyId"], "spki": RFC_KEY["spki"], "status": status, "added": "2026-10-20"}
    path = tmp_path / f"signing-keys-{status}.json"
    path.write_text(json.dumps({"schema": 1, "keys": [entry]}), encoding="utf-8")
    monkeypatch.setattr(signature_verify, "KEYS_PATH", path)
    return RFC_KEY["keyId"]


def _signable(db, version="1.0.0", theme_id=PLAY, **extra):
    """A version whose file hashes add up to its digest, so it can be signed (and approved)."""
    row = _package(db, version, theme_id, **extra)
    entry = {**row.file_manifest["dist/entry.js"], "sha256": hashlib.sha256(f"{theme_id}@{version}".encode()).hexdigest()}
    row.file_manifest = {**row.file_manifest, "dist/entry.js": entry}
    row.digest = package_digest({path: item["sha256"] for path, item in row.file_manifest.items()})
    db.commit()
    return row


def _sign(db, package, case=None):
    """Store the RFC seed's signature of the version's canonical text, as the paste route stores it."""
    seed = Ed25519PrivateKey.from_private_bytes(bytes.fromhex((case or RFC_KEY)["seed"]))
    text = signing.canonical_text(signing.signing_payload(package))
    package.signature = base64.b64encode(seed.sign(text.encode("ascii"))).decode("ascii")
    db.commit()


def test_a_reserved_publisher_is_official_only_while_the_shown_version_verifies(api, monkeypatch, tmp_path):
    key_id = _keys(monkeypatch, tmp_path)
    signed = _signable(api.db, "1.0.0")
    _sign(api.db, signed)
    verified = {"state": "verified", "keyId": key_id}
    card = _listings(api)[PLAY_LISTING]
    assert card["publisher"] == {"name": "automatos", "official": True} and card["signature"] == verified
    detail = api.client.get(f"/api/marketplace/themes/{PLAY_LISTING}").json()
    assert detail["publisher"]["official"] is True and detail["signature"] == verified
    assert detail["versions"]["1.0.0"]["signature"] == verified

    _signable(api.db, "1.1.0")  # newer and unsigned: the card shows it, so it isn't Official
    detail = api.client.get(f"/api/marketplace/themes/{PLAY_LISTING}").json()
    assert detail["version"] == "1.1.0" and detail["publisher"]["official"] is False
    assert detail["signature"] == UNSIGNED
    assert {v: e["signature"]["state"] for v, e in detail["versions"].items()} == {"1.0.0": "verified", "1.1.0": "unsigned"}


@pytest.mark.parametrize("why", ["another key", "retired key", "revoked key", "tampered files"])
def test_a_signature_that_does_not_verify_is_never_official(api, monkeypatch, tmp_path, why):
    _keys(monkeypatch, tmp_path)
    play = _signable(api.db)
    _sign(api.db, play, OTHER_RFC_KEY if why == "another key" else None)
    key_out_of_use = why in ("retired key", "revoked key")
    if key_out_of_use:
        _keys(monkeypatch, tmp_path, status=why.split()[0])
    if why == "tampered files":
        play.file_manifest = {**play.file_manifest, "README.md": _file("99")}
        api.db.commit()
    verdict = signature_verify.package_verdict(play).view()  # what the listing shows of the signature
    assert verdict == {"state": "invalid", "keyId": RFC_KEY["keyId"] if key_out_of_use else None}  # the key that signed
    card = _listings(api).get(PLAY_LISTING)
    if why == "retired key":  # still its signer's: listed, never Official
        assert card["publisher"]["official"] is False and card["signature"] == verdict
    else:  # C-05c (QB-28): a signature that fails, or a revoked key's, disables the version, so it isn't listed
        assert card is None
        assert api.client.get(f"/api/marketplace/themes/{PLAY_LISTING}").status_code == 404


def test_a_publisher_marked_reserved_needs_the_signature_too_and_an_unreserved_one_never_is_official(
        api, monkeypatch, tmp_path):
    key_id = _keys(monkeypatch, tmp_path)
    api.db.add(ThemePublisher(publisher="acme", owner_user_id=None, reserved=True))
    api.db.commit()
    ink = _signable(api.db, theme_id="acme/ink")
    assert _listings(api)["acme.ink"]["publisher"] == {"name": "acme", "official": False}
    _sign(api.db, ink)
    assert _listings(api)["acme.ink"]["publisher"] == {"name": "acme", "official": True}
    nib = _signable(api.db, theme_id="nibco/nib")
    _sign(api.db, nib)
    card = _listings(api)["nibco.nib"]
    assert card["publisher"] == {"name": "nibco", "official": False}
    assert card["signature"] == {"state": "verified", "keyId": key_id}


def test_a_keys_file_that_does_not_load_fails_closed_and_the_listings_still_answer(api, monkeypatch, tmp_path):
    _keys(monkeypatch, tmp_path)
    play = _signable(api.db)
    _sign(api.db, play)
    assert _listings(api)[PLAY_LISTING]["publisher"]["official"] is True
    broken = tmp_path / "signing-keys-broken.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(signature_verify, "KEYS_PATH", broken)
    card = _listings(api)[PLAY_LISTING]  # 200, not a 500 for every workspace
    assert card["publisher"]["official"] is False and card["signature"] == {"state": "invalid", "keyId": None}
    detail = api.client.get(f"/api/marketplace/themes/{PLAY_LISTING}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["publisher"]["official"] is False
    assert detail.json()["versions"]["1.0.0"]["signature"]["state"] == "invalid"


def _key_file(monkeypatch, tmp_path, *held):
    """Point the verifier at a keys file holding each ``(case, status)`` of ``held``."""
    keys = [{"keyId": case["keyId"], "spki": case["spki"], "status": status, "added": "2026-10-20"}
            for case, status in held]
    path = tmp_path / "signing-keys-held.json"
    path.write_text(json.dumps({"schema": 1, "keys": keys}), encoding="utf-8")
    monkeypatch.setattr(signature_verify, "KEYS_PATH", path)


def test_a_version_whose_key_was_retired_is_official_again_once_an_active_key_re_signs_it(
        api, monkeypatch, tmp_path):
    _keys(monkeypatch, tmp_path)
    play = _signable(api.db)
    request_id = uuid.UUID(signing.request_signature(api.db, play, OPERATOR_PK)["requestId"])
    _sign(api.db, play)  # key A
    assert _listings(api)[PLAY_LISTING]["publisher"]["official"] is True
    with pytest.raises(signing.SigningError) as refused:  # A still verifies, so nothing replaces it
        review.sign(api.db, play.id, request_id, play.signature, OPERATOR_PK)
    assert refused.value.status_code == 409

    _key_file(monkeypatch, tmp_path, (RFC_KEY, "retired"), (OTHER_RFC_KEY, "active"))  # A retired, B added
    assert _listings(api)[PLAY_LISTING]["publisher"]["official"] is False
    seed = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(OTHER_RFC_KEY["seed"]))
    text = signing.canonical_text(signing.signing_payload(play)).encode("ascii")
    stored = review.sign(api.db, play.id, request_id, base64.b64encode(seed.sign(text)).decode("ascii"), OPERATOR_PK)
    api.db.commit()
    assert stored["signature"]["keyId"] == OTHER_RFC_KEY["keyId"]
    card = _listings(api)[PLAY_LISTING]
    assert card["publisher"]["official"] is True
    assert card["signature"] == {"state": "verified", "keyId": OTHER_RFC_KEY["keyId"]}


def test_a_local_operator_who_approves_automatos_x_unsigned_is_not_official(api, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    # as the operator's submission is once the submit route's automated review passed it (approve refuses not_run)
    submitted = _signable(api.db, theme_id="automatos/x", status="pending", scan_verdict=VERDICT_PASS)
    approved = review.approve(api.db, submitted.id, OPERATOR_PK)  # the local edition needs no isolated job
    api.db.commit()
    assert approved["status"] == "approved" and approved["signatureRequest"]["requestId"]
    card = _listings(api)["automatos.x"]
    assert card["publisher"] == {"name": "automatos", "official": False} and card["signature"] == UNSIGNED
    detail = api.client.get("/api/marketplace/themes/automatos.x").json()
    assert detail["publisher"]["official"] is False and detail["versions"]["1.0.0"]["signature"] == UNSIGNED


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
