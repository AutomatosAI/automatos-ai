"""PRD-261 C-03a — the admin review queue: pending, report, report-note, approve, reject,
deactivate, signature.

The real router (``api/admin_themes.py``) runs on a mini FastAPI app over in-memory SQLite
holding the theme tables and a small ``users`` table, with the REAL ``caller_is_admin`` and
``resolve_user_pk``. Pins (plan §2.6 C-03a, D3 19 Oct):

* a blocked scan cannot be approved, nor a not-run one; ``review_required`` (the LLM stage did
  not run) can be;
* approve writes the ``approved`` event and records a signature REQUEST whose payload is
  ``{id, version, digest, files}``; it never signs and the package has no signature after it;
* the signature route stores a pasted base64 Ed25519 signature (64 bytes) for the open request
  only; the reviewer's isolated-job note is stored and gates approval on the hosted edition;
* reject needs a reason; deactivate writes a ``disable`` revocation of the digest;
* every route refuses a non-admin, and a workspace's private upload is 404 here;
* auto-publish never applies; no signing key in config.
"""
from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
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

import api.admin_themes as admin_api  # noqa: E402
import core.models  # noqa: E402,F401  (registers every table the FKs name)
from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.themes import ThemeInstallation, ThemePackage, ThemeReviewEvent, ThemeRevocation  # noqa: E402
from services.themes import revocations, signing  # noqa: E402
from services.themes.digest import package_digest  # noqa: E402

ADMIN_PK, MEMBER_PK = 7, 8
TABLES = [m.__table__ for m in (ThemePackage, ThemeReviewEvent, ThemeRevocation, ThemeInstallation)]
GOOD_SIGNATURE = base64.b64encode(bytes(range(64))).decode("ascii")
PASSING_NOTE = {"summary": "isolated job: build reproducible, digests equal, JS rules clean", "outcome": "pass"}
BASE = "/api/admin/themes"


def _ctx(role="admin", clerk="clerk-7"):
    user = UserContext(id=clerk, clerk_user_id=clerk, system_role=role, raw_claims={"sub": clerk})
    return RequestContext(workspace_id=uuid.uuid4(), user=user, auth_type="clerk")


@pytest.fixture
def api(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=TABLES)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY, clerk_user_id TEXT, email TEXT)"))
        conn.execute(sa.text("INSERT INTO users (id, clerk_user_id) VALUES (7, 'clerk-7'), (8, 'clerk-8')"))
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    session = sessionmaker(bind=engine)()
    state = SimpleNamespace(db=session, ctx=_ctx())
    app = FastAPI()
    app.include_router(admin_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        session.close()
        engine.dispose()


def _files(version="1.0.0"):
    """Two files whose hashes differ per version, so each version has its own digest (unique in the Marketplace)."""
    return {
        "dist/entry.js": {"type": "text/javascript", "size": 1, "sha256": hashlib.sha256(version.encode()).hexdigest()},
        "dist/theme.css": {"type": "text/css", "size": 1, "sha256": "cd" * 32},
    }


def _package(db, verdict="pass", status="pending", workspace_id=None, theme_id="acme/ink", version="1.0.0",
             age_minutes=0, files=None, digest=None):
    files = files if files is not None else _files(version)
    row = ThemePackage(
        theme_id=theme_id, version=version, owner_type="workspace" if workspace_id else "marketplace",
        workspace_id=workspace_id, manifest={"id": theme_id, "name": "Ink", "version": version},
        file_manifest=files, storage_prefix=f"themes/{theme_id}/{version}/", status=status,
        digest=digest or package_digest({p: r["sha256"] for p, r in files.items()}),
        scan_verdict=verdict, reviewed=status == "approved", submitted_by=MEMBER_PK,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=age_minutes),
    )
    db.add(row)
    db.commit()
    return row


def _events(db, package, event=None):
    query = db.query(ThemeReviewEvent).filter(ThemeReviewEvent.package_id == package.id)
    rows = query.filter(ThemeReviewEvent.event == event).all() if event else query.all()
    return rows


def _note(api, package, body=None):
    return api.client.post(f"{BASE}/{package.id}/report-note", json=body or PASSING_NOTE)


def _approve(api, package, with_note=True):
    if with_note:
        assert _note(api, package).status_code == 200
    return api.client.post(f"{BASE}/{package.id}/approve")


# --- approve -----------------------------------------------------------------------------


@pytest.mark.parametrize("verdict", ["blocked", "not_run"])
def test_a_blocked_or_not_run_scan_cannot_be_approved(api, verdict):
    package = _package(api.db, verdict=verdict)
    resp = _approve(api, package)
    assert resp.status_code == 409
    assert "can't be approved" in resp.json()["detail"]
    api.db.refresh(package)
    assert (package.status, package.reviewed, package.signature) == ("pending", False, None)
    assert _events(api.db, package, "approved") == []
    assert _events(api.db, package, signing.EVENT_SIGNATURE_REQUESTED) == []


def test_approve_writes_the_event_and_requests_a_signature(api):
    package = _package(api.db, verdict="review_required")  # "LLM stage did not run" is approvable
    resp = _approve(api, package)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    api.db.refresh(package)
    assert (package.status, package.reviewed, package.signature) == ("approved", True, None)
    approved = _events(api.db, package, "approved")
    assert len(approved) == 1
    assert (approved[0].from_status, approved[0].to_status, approved[0].actor_user_id) == (
        "pending", "approved", ADMIN_PK,
    )
    request = body["signatureRequest"]
    payload = {"id": "acme/ink", "version": "1.0.0", "digest": package.digest,
               "files": {p: r["sha256"] for p, r in _files().items()}}
    assert request["payload"] == payload
    assert json.loads(request["canonical"]) == payload
    assert request["canonical"] == json.dumps(payload, sort_keys=True, separators=(",", ":"))
    requested = _events(api.db, package, signing.EVENT_SIGNATURE_REQUESTED)
    assert [str(row.id) for row in requested] == [request["requestId"]]


def test_approve_refuses_a_version_that_is_not_pending_and_a_revoked_digest(api):
    approved = _package(api.db, status="approved", version="0.9.0")
    assert _approve(api, approved, with_note=False).status_code == 409
    revoked = _package(api.db)
    api.db.add(ThemeRevocation(digest=revoked.digest, level="disable", reason="malware"))
    api.db.commit()
    resp = _approve(api, revoked)
    assert resp.status_code == 409 and "revoked" in resp.json()["detail"]


def test_approve_refuses_files_that_do_not_add_up_to_the_digest(api):
    package = _package(api.db, digest="ef" * 32)
    assert _approve(api, package).status_code == 409
    api.db.refresh(package)
    assert package.status == "pending"
    assert _events(api.db, package, "approved") == []


def test_hosted_edition_needs_a_passing_isolated_job_note(api):
    package = _package(api.db)
    resp = _approve(api, package, with_note=False)
    assert resp.status_code == 409 and "isolated review job" in resp.json()["detail"]
    assert _note(api, package, {"summary": "digest differs after rebuild", "outcome": "fail"}).status_code == 200
    resp = api.client.post(f"{BASE}/{package.id}/approve")
    assert resp.status_code == 409 and "failed" in resp.json()["detail"]
    assert _approve(api, package).status_code == 200  # the latest note wins


def test_local_edition_approves_without_the_isolated_stage(api, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    package = _package(api.db)
    assert _approve(api, package, with_note=False).status_code == 200


# --- auto-publish ------------------------------------------------------------------------


def test_auto_publish_never_applies(api):
    for index in range(6):  # more than marketplace.py's five-approval "trusted" threshold
        _package(api.db, status="approved", version=f"0.{index}.0")
    fresh = _package(api.db, version="2.0.0")
    api.db.refresh(fresh)
    assert fresh.status == "pending" and fresh.reviewed is False
    pending = api.client.get(f"{BASE}/pending").json()["pending"]
    assert [row["id"] for row in pending] == [str(fresh.id)]
    api.ctx = _ctx(role="user", clerk="clerk-8")
    assert _approve(api, fresh, with_note=False).status_code == 403
    for path in ("services/themes/review.py", "services/themes/signing.py", "api/admin_themes.py"):
        source = (_ORCH / path).read_text()
        assert "is_trusted" not in source and "approved_count" not in source


# --- queue and report --------------------------------------------------------------------


def test_pending_lists_marketplace_pending_versions_oldest_first(api):
    newer = _package(api.db, version="1.1.0", age_minutes=1)
    older = _package(api.db, version="1.0.0", age_minutes=5, verdict="blocked")
    _package(api.db, version="0.5.0", status="rejected")
    _package(api.db, workspace_id=uuid.uuid4(), version="1.0.0")  # a private upload: never queued
    rows = api.client.get(f"{BASE}/pending").json()["pending"]
    assert [row["id"] for row in rows] == [str(older.id), str(newer.id)]
    assert rows[0]["approval"] == {"allowed": False, "reason": rows[0]["approval"]["reason"]}
    assert "blocked" in rows[0]["approval"]["reason"]
    assert rows[1]["publisher"] == "acme" and rows[1]["reportNote"] is None


def test_report_shows_note_history_and_request(api):
    package = _package(api.db, verdict="review_required")
    package.scan_report = {"checks": {"llm": {"status": "review_required", "reason": "LLM stage did not run"}}}
    api.db.commit()
    _approve(api, package)
    report = api.client.get(f"{BASE}/{package.id}/report").json()
    assert report["scanReport"]["checks"]["llm"]["reason"] == "LLM stage did not run"
    assert report["reportNote"]["outcome"] == "pass"
    events = [row["event"] for row in report["history"]]
    assert events[0] == "report_note"  # its own request, before the approval
    assert sorted(events[1:]) == ["approved", "signature_requested"]  # one flush, same instant possible
    assert report["signatureRequest"]["payload"]["digest"] == package.digest
    assert report["signatureVerified"] is False


def test_a_private_upload_or_unknown_id_is_404(api):
    private = _package(api.db, workspace_id=uuid.uuid4())
    for package_id in (private.id, uuid.uuid4()):
        assert api.client.get(f"{BASE}/{package_id}/report").status_code == 404
        assert api.client.post(f"{BASE}/{package_id}/approve").status_code == 404
        assert api.client.post(f"{BASE}/{package_id}/reject", json={"reason": "x"}).status_code == 404


def test_report_note_refuses_nul_and_blank_and_non_pending(api):
    package = _package(api.db)
    assert _note(api, package, {"summary": "bad\x00text", "outcome": "pass"}).status_code == 400
    assert _note(api, package, {"summary": "   ", "outcome": "pass"}).status_code == 400
    assert _note(api, package, {"summary": "ok", "outcome": "maybe"}).status_code == 422
    done = _package(api.db, status="approved", version="0.1.0")
    assert _note(api, done).status_code == 409


# --- reject and deactivate ---------------------------------------------------------------


def test_reject_needs_a_reason_and_writes_the_event(api):
    package = _package(api.db)
    assert api.client.post(f"{BASE}/{package.id}/reject", json={}).status_code == 422
    assert api.client.post(f"{BASE}/{package.id}/reject", json={"reason": "  "}).status_code == 400
    resp = api.client.post(f"{BASE}/{package.id}/reject", json={"reason": "fetch() in source/app.ts"})
    assert resp.status_code == 200 and resp.json()["status"] == "rejected"
    [event] = _events(api.db, package, "rejected")
    assert (event.reason, event.from_status, event.to_status) == ("fetch() in source/app.ts", "pending", "rejected")
    assert _approve(api, package, with_note=False).status_code == 409


def test_deactivate_writes_a_disable_revocation(api):
    pending = _package(api.db, version="0.1.0")
    assert api.client.post(f"{BASE}/{pending.id}/deactivate", json={"reason": "x"}).status_code == 409
    package = _package(api.db)
    assert _approve(api, package).status_code == 200
    resp = api.client.post(f"{BASE}/{package.id}/deactivate", json={"reason": "reported: exfiltrates chat"})
    assert resp.status_code == 200 and resp.json()["revocation"] == "disable"
    api.db.refresh(package)
    assert package.status == "deactivated"
    [revocation] = api.db.query(ThemeRevocation).filter(ThemeRevocation.digest == package.digest).all()
    assert (revocation.level, revocation.created_by) == ("disable", ADMIN_PK)
    assert revocations.is_disabled(api.db, package)
    assert len(_events(api.db, package, "deactivated")) == 1


# --- signature ---------------------------------------------------------------------------


def _signature(api, package, request_id, signature=GOOD_SIGNATURE):
    return api.client.post(f"{BASE}/{package.id}/signature", json={"requestId": str(request_id), "signature": signature})


def test_signature_is_stored_for_the_open_request_only(api):
    package = _package(api.db)
    assert _signature(api, package, uuid.uuid4()).status_code == 409  # not approved yet
    request_id = _approve(api, package).json()["signatureRequest"]["requestId"]
    assert _signature(api, package, uuid.uuid4()).status_code == 409
    assert _signature(api, package, request_id, "not base64 !").status_code == 400
    short = base64.b64encode(b"x" * 32).decode("ascii")
    assert _signature(api, package, request_id, short).status_code == 400
    urlsafe = base64.urlsafe_b64encode(bytes(range(64))).decode("ascii").rstrip("=")
    resp = _signature(api, package, request_id, f"  {urlsafe}\n")
    assert resp.status_code == 200, resp.text
    assert resp.json()["signature"]["verified"] is False
    api.db.refresh(package)
    assert package.signature == GOOD_SIGNATURE
    assert len(_events(api.db, package, signing.EVENT_SIGNED)) == 1
    assert _signature(api, package, request_id).status_code == 409  # already signed


# --- gates, shape and the no-key rule ----------------------------------------------------


EXPECTED_ROUTES = {
    ("GET", f"{BASE}/pending"), ("GET", f"{BASE}/{{package_id}}/report"),
    ("POST", f"{BASE}/{{package_id}}/report-note"), ("POST", f"{BASE}/{{package_id}}/approve"),
    ("POST", f"{BASE}/{{package_id}}/reject"), ("POST", f"{BASE}/{{package_id}}/deactivate"),
    ("POST", f"{BASE}/{{package_id}}/signature"),
}


def _routes():
    return [route for route in admin_api.router.routes if isinstance(route, APIRoute)]


def test_routes_are_plain_def_and_assert_admin_in_the_handler():
    seen = {(method, route.path) for route in _routes() for method in route.methods}
    assert seen == EXPECTED_ROUTES
    for route in _routes():
        assert not inspect.iscoroutinefunction(route.endpoint), route.path
        # tests/authz_sweep_probe.py classifies ADMIN_GATED_IN_HANDLER by this text.
        assert "_require_admin(ctx" in inspect.getsource(route.endpoint), route.path


@pytest.mark.parametrize("signed_in", [True, False])
def test_every_route_refuses_a_non_admin(api, signed_in):
    package = _package(api.db)
    anonymous = RequestContext(workspace_id=uuid.uuid4(), user=None, auth_type="anonymous")
    api.ctx = _ctx(role="user", clerk="clerk-8") if signed_in else anonymous
    bodies = {"report-note": PASSING_NOTE, "reject": {"reason": "x"}, "deactivate": {"reason": "x"},
              "signature": {"requestId": str(uuid.uuid4()), "signature": GOOD_SIGNATURE}}
    assert api.client.get(f"{BASE}/pending").status_code == 403
    assert api.client.get(f"{BASE}/{package.id}/report").status_code == 403
    for action in ("report-note", "approve", "reject", "deactivate", "signature"):
        assert api.client.post(f"{BASE}/{package.id}/{action}", json=bodies.get(action)).status_code == 403
    api.db.refresh(package)
    assert package.status == "pending" and _events(api.db, package) == []


def test_a_super_admin_may_review(api):
    api.ctx = _ctx(role="super_admin")
    assert api.client.get(f"{BASE}/pending").status_code == 200


KEY_NAME = re.compile(r"SIGNING_KEY|SIGN_KEY|ED25519|THEME\w*(SIGN|PRIVATE_KEY|SECRET)", re.IGNORECASE)


def test_no_signing_key_in_config():
    names = [name for name in dir(config) if KEY_NAME.search(name)]
    assert names == [], f"a signing key setting reached the orchestrator: {names}"
    surface = json.loads((_ORCH / "reports" / "config-surface.json").read_text())["settings"]
    assert [name for name in surface if KEY_NAME.search(name)] == []
    for path in ("services/themes/signing.py", "services/themes/review.py", "api/admin_themes.py"):
        source = (_ORCH / path).read_text()
        for banned in ("cryptography", "nacl", "PrivateKey", "private_key", "httpx", "requests", "github"):
            assert not re.search(rf"^\s*(import|from)\s+[^\n]*{banned}", source, re.MULTILINE), (path, banned)
    # signing reads no setting at all: there is no key to read
    assert not re.search(r"^\s*(from config import|import config)", inspect.getsource(signing), re.MULTILINE)
