"""PRD-261 C-06c — the official listing, end to end through the real routers (the CI half of C-06).

What the hosted owner run (C-06d) then checks only by hand: here a platform admin submits an
``automatos/`` theme (the review stage mocked to pass), pastes a passing isolated-job note naming
its digest, approves it, signs the approval's canonical text with an RFC 8032 test key (the keys
path points at that key's public half; no other key is ever trusted here) and stores the
signature. Pins (plan §2.6 C-06c, QB-21):

* the listing is Official, with ``signature {state: verified, keyId}``;
* a second workspace installs it by reference (one package row, two installs) and
  ``GET /api/themes/active`` there answers the theme;
* a signature from another key is a 400 and the listing is not Official; a publisher that is not
  reserved is never Official, even with a signature that verifies;
* a ``disable`` revocation by theme id (a row inserted directly) after the installs makes
  ``active`` answer the base in both workspaces;
* (C-05b, local mode) the local edition's operator who submits and approves ``automatos/x``
  unsigned is never Official.

The routers are the real ones on a test app over SQLite (``create_all``), with a fake S3 client
in ``services/themes/storage.py``; the workspace role is stubbed to owner, every hosted caller is
a Clerk principal who signed in a minute ago (the step-ups pass) and resolves from a small
``users`` table, and the local-mode caller is the local edition's operator.
"""
from __future__ import annotations

import base64
import io
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
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.admin_themes as admin_api  # noqa: E402
import api.marketplace_themes as listings_api  # noqa: E402
import api.marketplace_themes_submit as submit_api  # noqa: E402
import api.themes_install as install_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.models  # noqa: E402,F401  (registers every table the FKs name)
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
from services.themes import signature_verify, signing, storage, submissions  # noqa: E402
from services.themes.submissions import THEME_SUBMIT_ROUTE  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

OFFICIAL_ID, OFFICIAL_LISTING = "automatos/hello-official", "automatos.hello-official"
COMMUNITY_ID, COMMUNITY_LISTING = "acme-e2e/hello-signed", "acme-e2e.hello-signed"
LOCAL_ID, LOCAL_LISTING = "automatos/x", "automatos.x"
OPERATOR_PK = 1
SOURCE_FILE, SOURCE_BYTES = "source/src/index.tsx", b"export default {}\n"  # a submission ships source/ (C-06p4)
ADMIN, MEMBER = "clerk-7", "clerk-8"
WS_A, WS_B = uuid.uuid4(), uuid.uuid4()
ADMIN_BASE, LISTINGS = "/api/admin/themes", "/api/marketplace/themes"
VECTOR = json.loads((_ORCH / "tests" / "fixtures" / "themes" / "signing-vector.json").read_text(encoding="utf-8"))
RFC_KEY, OTHER_RFC_KEY = VECTOR["ed25519"]["cases"][:2]
TABLES = (ThemePackage, ThemePublisher, ThemeReviewEvent, ThemeInstallation, ThemeWorkspaceSettings, ThemeRevocation,
          ThemeChoice, ThemeLook, ThemeOptIn)
NOTE_TAIL = "isolated job: build reproducible, digests equal, JS rules clean"


class _FakeS3:
    """Objects by key, as ``services/themes/storage.py`` writes them."""

    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803 (boto3's names)
        self.objects[Key] = (Body, ContentType)

    def get_object(self, Bucket, Key):  # noqa: N803
        return {"Body": io.BytesIO(self.objects[Key][0])}

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.objects.pop(Key, None)


def _ctx(clerk, workspace_id=WS_A):
    """A Clerk caller who signed in a minute ago (``fva``), so every step-up passes."""
    role = "admin" if clerk == ADMIN else "user"
    claims = {"sub": clerk, "iat": int(time.time()), "fva": [1, -1]}
    user = UserContext(id=clerk, clerk_user_id=clerk, email=f"{clerk}@example.com", system_role=role, raw_claims=claims)
    return RequestContext(workspace_id=workspace_id, user=user, auth_type="clerk")


def _local_operator():
    """The local edition's anonymous operator (a super admin, its user id on the claims)."""
    user = UserContext(id="operator@local", email="operator@local", system_role="super_admin",
                       raw_claims={"source": LOCAL_OPERATOR_CLAIM_SOURCE, "user_id": OPERATOR_PK})
    return RequestContext(workspace_id=WS_A, user=user, auth_type="anonymous")


def _passing_stage(checked, package_bytes, token_budget=None):
    return {"schema": "automatos-theme-scan/v1", "verdict": "pass", "checks": {}}


def _keys_file(tmp_path):
    entry = {"keyId": RFC_KEY["keyId"], "spki": RFC_KEY["spki"], "status": "active", "added": "2026-10-20"}
    path = tmp_path / "signing-keys.json"
    path.write_text(json.dumps({"schema": 1, "keys": [entry]}), encoding="utf-8")
    return path


@pytest.fixture
def env(monkeypatch, tmp_path):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=[model.__table__ for model in TABLES])
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY, clerk_user_id TEXT, email TEXT)"))
        conn.execute(sa.text("INSERT INTO users (id, clerk_user_id) VALUES (7, 'clerk-7'), (8, 'clerk-8')"))
    fake = _FakeS3()
    monkeypatch.setattr(storage, "get_s3_client", lambda: fake)
    monkeypatch.setattr(storage, "ensure_bucket", lambda bucket: None)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr(signature_verify, "KEYS_PATH", _keys_file(tmp_path))
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")  # the stage is mocked; never the Anthropic API
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", "on", raising=False)
    monkeypatch.setattr(config, submissions.ONE_OPEN_SUBMISSION_SETTING, False, raising=False)
    sessions = sessionmaker(bind=engine)
    state = SimpleNamespace(sessions=sessions, ctx=_ctx(ADMIN))
    state.client = _client(state)
    try:
        yield state
    finally:
        engine.dispose()


def _client(state):
    app = FastAPI()
    for router in (submit_api.router, listings_api.router, install_api.router, admin_api.router):
        app.include_router(router)
    step_up.install_step_up_handler(app)

    def _db():
        session = state.sessions()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[submit_api.submission_review_stage] = lambda: _passing_stage
    return TestClient(app)


def _query(env, work):
    session = env.sessions()
    try:
        return work(session)
    finally:
        session.close()


def _ok(resp, status=200):
    assert resp.status_code == status, resp.text
    return resp.json()


# --- the admin's steps ----------------------------------------------------------------------


def _submit(env, theme_id):
    """Submit the hello package (with one source/ file) as ``theme_id``: the pending submission."""
    package = build.build_package(replace={SOURCE_FILE: SOURCE_BYTES, build.MANIFEST: build.changed_manifest({"id": theme_id})})
    return _ok(env.client.post(THEME_SUBMIT_ROUTE, files={"file": ("hello.autotheme", package, "application/zip")}), 201)


def _submit_and_approve(env, theme_id, submitter=ADMIN):
    """Submit as ``submitter``, note a passing isolated job and approve as the admin: the signature request."""
    env.ctx = _ctx(submitter)
    submitted = _submit(env, theme_id)
    row, digest = submitted["package"]["id"], submitted["package"]["digest"]
    env.ctx = _ctx(ADMIN)
    summary = (f"Automatos theme review (automatos-theme-review/v1): {theme_id} 1.0.0\nOutcome: pass\n"
               f"Expected digest: {digest}\nPackage digest:  {digest}\n\n{NOTE_TAIL}")
    _ok(env.client.post(f"{ADMIN_BASE}/{row}/report-note", json={"summary": summary, "outcome": "pass"}))
    approved = _ok(env.client.post(f"{ADMIN_BASE}/{row}/approve"))
    assert approved["status"] == "approved"
    return row, approved["signatureRequest"]


def _signature_of(request, case):
    """The request's canonical text signed with an RFC 8032 seed, as the kit's sign workflow prints it."""
    assert request["canonical"] == signing.canonical_text(request["payload"])
    seed = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(case["seed"]))
    return base64.b64encode(seed.sign(request["canonical"].encode("ascii"))).decode("ascii")


def _paste(env, row, request, case):
    body = {"requestId": request["requestId"], "signature": _signature_of(request, case)}
    return env.client.post(f"{ADMIN_BASE}/{row}/signature", json=body)


def _listing(env, listing_id):
    return _ok(env.client.get(f"{LISTINGS}/{listing_id}"))


# --- a member's steps -----------------------------------------------------------------------


def _install_and_choose(env, listing_id, workspace_id):
    env.ctx = _ctx(MEMBER, workspace_id)
    install = _ok(env.client.post(f"{LISTINGS}/{listing_id}/install"), 201)["install"]
    _ok(env.client.put("/api/themes/choice", json={"choice": install["installId"]}))
    return install


def _active(env, workspace_id):
    env.ctx = _ctx(MEMBER, workspace_id)
    return _ok(env.client.get("/api/themes/active"))


# --- the run --------------------------------------------------------------------------------


def test_the_official_listing_end_to_end(env):
    row, request = _submit_and_approve(env, OFFICIAL_ID)
    assert request["payload"]["id"] == OFFICIAL_ID
    unsigned = _listing(env, OFFICIAL_LISTING)
    assert unsigned["publisher"] == {"name": "automatos", "official": False}
    assert unsigned["signature"] == {"state": "unsigned", "keyId": None}

    refused = _paste(env, row, request, OTHER_RFC_KEY)  # a key the keys file doesn't hold
    assert refused.status_code == 400 and refused.json()["detail"] == signing.NOT_VERIFIED
    assert _query(env, lambda db: db.get(ThemePackage, uuid.UUID(row)).signature) is None
    assert _listing(env, OFFICIAL_LISTING)["publisher"]["official"] is False

    stored = _ok(_paste(env, row, request, RFC_KEY))["signature"]
    assert (stored["verified"], stored["keyId"]) == (True, RFC_KEY["keyId"])
    official = _listing(env, OFFICIAL_LISTING)
    assert official["publisher"] == {"name": "automatos", "official": True}
    assert official["signature"] == {"state": "verified", "keyId": RFC_KEY["keyId"]}
    assert official["versions"]["1.0.0"]["signature"]["state"] == "verified"
    report = _ok(env.client.get(f"{ADMIN_BASE}/{row}/report"))
    assert (report["signatureVerified"], report["keyId"]) == (True, RFC_KEY["keyId"])

    first = _install_and_choose(env, OFFICIAL_LISTING, WS_A)
    second = _install_and_choose(env, OFFICIAL_LISTING, WS_B)
    installs = _query(env, lambda db: db.query(ThemeInstallation).all())
    assert {(i.workspace_id, i.package_id) for i in installs} == {(WS_A, uuid.UUID(row)), (WS_B, uuid.UUID(row))}
    assert _query(env, lambda db: db.query(ThemePackage).filter(ThemePackage.theme_id == OFFICIAL_ID).count()) == 1
    for workspace_id, install in ((WS_B, second), (WS_A, first)):
        theme = _active(env, workspace_id)["theme"]
        assert (theme["themeId"], theme["packageId"], theme["installId"]) == (OFFICIAL_ID, row, install["installId"])
        assert theme["reviewed"] is True and theme["optIn"]["required"] is False

    def revoke(db):
        db.add(ThemeRevocation(theme_id=OFFICIAL_ID, level="disable", reason="Withdrawn in the C-06c run."))
        db.commit()

    _query(env, revoke)
    for workspace_id, install in ((WS_A, first), (WS_B, second)):
        active = _active(env, workspace_id)
        assert (active["choice"], active["theme"]) == ("studio", None)  # the hello theme's base
        assert active["revoked"]["installId"] == install["installId"] and active["revoked"]["level"] == "disable"


def test_a_publisher_that_is_not_reserved_is_never_official_even_signed(env):
    row, request = _submit_and_approve(env, COMMUNITY_ID, submitter=MEMBER)
    stored = _ok(_paste(env, row, request, RFC_KEY))["signature"]
    assert stored["verified"] is True
    listing = _listing(env, COMMUNITY_LISTING)
    assert listing["publisher"] == {"name": "acme-e2e", "official": False}
    assert listing["signature"] == {"state": "verified", "keyId": RFC_KEY["keyId"]}


def test_a_local_operator_who_submits_and_approves_automatos_x_unsigned_is_never_official(env, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    env.ctx = _local_operator()
    submitted = _submit(env, LOCAL_ID)["package"]
    approved = _ok(env.client.post(f"{ADMIN_BASE}/{submitted['id']}/approve"))  # local: no isolated job, no step-up
    assert approved["status"] == "approved" and approved["signatureRequest"]["requestId"]
    listing = _listing(env, LOCAL_LISTING)
    assert listing["publisher"] == {"name": "automatos", "official": False}
    assert listing["signature"] == {"state": "unsigned", "keyId": None}
    assert listing["versions"]["1.0.0"]["signature"]["state"] == "unsigned"
