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
  only, and only when an active key of the keys file verifies it (C-05b: the keys path points at
  an RFC 8032 public key, whose published seed signs the request's canonical text here); one no
  active key verifies is a 400 and nothing is stored; the ``signed`` event and the report carry
  ``verified`` and the keyId from ``signature_verify``; a stored signature an active key still
  verifies refuses another (409), one that no longer does takes an active key's (QB-28); a keys
  file that does not load reads ``invalid`` in the report (200) and the paste is a 503 with its
  plain message, nothing stored;
* the reviewer's isolated-job note is stored and gates approval on the hosted edition;
* reject needs a reason; deactivate writes a ``disable`` revocation of the digest;
* every route refuses a non-admin, and a workspace's private upload is 404 here;
* report-note, approve, deactivate and signature need a fresh step-up after the admin check
  (a non-admin gets the plain 403, never a reverification prompt); a ``pass`` note must carry
  the kit's ``Package digest:  <digest>`` line for this version and ``Outcome: pass``;
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
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import time  # noqa: E402

import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.admin_themes as admin_api  # noqa: E402
import core.models  # noqa: E402,F401  (registers every table the FKs name)
from config import config  # noqa: E402
from core.auth import step_up  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.themes import ThemeInstallation, ThemePackage, ThemeReviewEvent, ThemeRevocation  # noqa: E402
from services.themes import revocations, signature_verify, signing  # noqa: E402
from services.themes.digest import package_digest  # noqa: E402

ADMIN_PK, MEMBER_PK = 7, 8
TABLES = [m.__table__ for m in (ThemePackage, ThemeReviewEvent, ThemeRevocation, ThemeInstallation)]
GOOD_SIGNATURE = base64.b64encode(bytes(range(64))).decode("ascii")
NOTE_TAIL = "isolated job: build reproducible, digests equal, JS rules clean"
FAILED_NOTE = {"summary": "Outcome: fail\ndigest differs after rebuild", "outcome": "fail"}
BASE = "/api/admin/themes"
VECTOR = json.loads((_ORCH / "tests" / "fixtures" / "themes" / "signing-vector.json").read_text(encoding="utf-8"))
RFC_KEY, OTHER_RFC_KEY = VECTOR["ed25519"]["cases"][:2]


def _ctx(role="admin", clerk="clerk-7", fresh=True):
    """A Clerk caller; ``fresh`` signed in a minute ago, so the admin routes' step-up passes."""
    claims = {"sub": clerk, "iat": int(time.time())}
    if fresh:
        claims["fva"] = [1, -1]
    user = UserContext(id=clerk, clerk_user_id=clerk, system_role=role, raw_claims=claims)
    return RequestContext(workspace_id=uuid.uuid4(), user=user, auth_type="clerk")


def _passing_note(package):
    """The kit's summary for this version (automatos-play scripts/review/summary.ts header lines)."""
    summary = (f"Automatos theme review (automatos-theme-review/v1): {package.theme_id} {package.version}\n"
               f"Outcome: pass\nExpected digest: {package.digest}\nPackage digest:  {package.digest}\n\n{NOTE_TAIL}")
    return {"summary": summary, "outcome": "pass"}


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
    step_up.install_step_up_handler(app)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        session.close()
        engine.dispose()


def _keys(monkeypatch, tmp_path, status="active", case=None):
    """Point the verifier at a keys file holding one RFC 8032 public key (TEST 1 unless ``case``)."""
    case = case or RFC_KEY
    entry = {"keyId": case["keyId"], "spki": case["spki"], "status": status, "added": "2026-10-20"}
    path = tmp_path / f"signing-keys-{status}.json"
    path.write_text(json.dumps({"schema": 1, "keys": [entry]}), encoding="utf-8")
    monkeypatch.setattr(signature_verify, "KEYS_PATH", path)
    return case["keyId"]


def _signed(canonical, case=None):
    """The request's canonical text signed with an RFC seed, as the kit's sign workflow prints it."""
    seed = Ed25519PrivateKey.from_private_bytes(bytes.fromhex((case or RFC_KEY)["seed"]))
    return base64.b64encode(seed.sign(canonical.encode("ascii"))).decode("ascii")


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
    return api.client.post(f"{BASE}/{package.id}/report-note", json=body or _passing_note(package))


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
    assert _note(api, package, FAILED_NOTE).status_code == 200
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
    assert (report["signatureVerified"], report["keyId"], report["signed"]) == (False, None, False)


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


def test_signature_is_stored_for_the_open_request_only(api, monkeypatch, tmp_path):
    key_id = _keys(monkeypatch, tmp_path)
    package = _package(api.db)
    assert _signature(api, package, uuid.uuid4()).status_code == 409  # not approved yet
    request = _approve(api, package).json()["signatureRequest"]
    request_id, good = request["requestId"], _signed(request["canonical"])
    assert _signature(api, package, uuid.uuid4(), good).status_code == 409
    assert _signature(api, package, request_id, "not base64 !").status_code == 400
    short = base64.b64encode(b"x" * 32).decode("ascii")
    assert _signature(api, package, request_id, short).status_code == 400
    urlsafe = base64.urlsafe_b64encode(base64.b64decode(good)).decode("ascii").rstrip("=")
    resp = _signature(api, package, request_id, f"  {urlsafe}\n")
    assert resp.status_code == 200, resp.text
    assert (resp.json()["signature"]["verified"], resp.json()["signature"]["keyId"]) == (True, key_id)
    api.db.refresh(package)
    assert package.signature == good
    [signed] = _events(api.db, package, signing.EVENT_SIGNED)
    assert (signed.detail["verified"], signed.detail["keyId"], signed.detail["requestId"]) == (True, key_id, request_id)
    assert _signature(api, package, request_id, good).status_code == 409  # already signed


def _refused_and_nothing_stored(api, package, request_id, signature):
    resp = _signature(api, package, request_id, signature)
    assert resp.status_code == 400 and resp.json()["detail"] == signing.NOT_VERIFIED, resp.text
    api.db.refresh(package)
    assert package.signature is None and _events(api.db, package, signing.EVENT_SIGNED) == []


def test_a_signature_no_active_key_verifies_is_400_and_nothing_is_stored(api, monkeypatch, tmp_path):
    package = _package(api.db)
    request = _approve(api, package).json()["signatureRequest"]
    request_id, canonical = request["requestId"], request["canonical"]
    _refused_and_nothing_stored(api, package, request_id, _signed(canonical))  # the shipped file: no key
    _keys(monkeypatch, tmp_path)
    _refused_and_nothing_stored(api, package, request_id, GOOD_SIGNATURE)  # 64 bytes, no key's signature
    _refused_and_nothing_stored(api, package, request_id, _signed(canonical, OTHER_RFC_KEY))  # another key
    other = json.loads(canonical)
    other["version"] = "9.9.9"
    _refused_and_nothing_stored(api, package, request_id, _signed(signing.canonical_text(other)))  # other text
    for status in ("retired", "revoked"):
        _keys(monkeypatch, tmp_path, status=status)
        _refused_and_nothing_stored(api, package, request_id, _signed(canonical))
    _keys(monkeypatch, tmp_path)
    assert _signature(api, package, request_id, _signed(canonical)).status_code == 200  # the request stays open


def test_the_report_reads_signature_verified_and_key_id_from_signature_verify(api, monkeypatch, tmp_path):
    key_id = _keys(monkeypatch, tmp_path)
    package = _package(api.db)
    request = _approve(api, package).json()["signatureRequest"]
    report = api.client.get(f"{BASE}/{package.id}/report").json()
    assert (report["signatureVerified"], report["keyId"]) == (False, None)  # approved, not signed yet
    assert _signature(api, package, request["requestId"], _signed(request["canonical"])).status_code == 200
    report = api.client.get(f"{BASE}/{package.id}/report").json()
    assert (report["signatureVerified"], report["keyId"], report["signed"]) == (True, key_id, True)
    _keys(monkeypatch, tmp_path, status="retired")  # the same stored signature, its key retired since
    report = api.client.get(f"{BASE}/{package.id}/report").json()
    assert (report["signatureVerified"], report["keyId"], report["signed"]) == (False, key_id, True)
    _keys(monkeypatch, tmp_path, case=OTHER_RFC_KEY)  # a keys file without the signer's key
    report = api.client.get(f"{BASE}/{package.id}/report").json()
    assert (report["signatureVerified"], report["keyId"]) == (False, None)


def _key_file(monkeypatch, tmp_path, *held):
    """Point the verifier at a keys file holding each ``(case, status)`` of ``held``."""
    keys = [{"keyId": case["keyId"], "spki": case["spki"], "status": status, "added": "2026-10-20"}
            for case, status in held]
    path = tmp_path / "signing-keys-held.json"
    path.write_text(json.dumps({"schema": 1, "keys": keys}), encoding="utf-8")
    monkeypatch.setattr(signature_verify, "KEYS_PATH", path)


def _report_check(api, package):
    """The report's check on the stored signature: ``(signatureVerified, keyId)``."""
    report = api.client.get(f"{BASE}/{package.id}/report").json()
    return report["signatureVerified"], report["keyId"]


def test_a_signature_whose_key_was_retired_takes_an_active_keys_and_one_that_verifies_never(
        api, monkeypatch, tmp_path):
    first, second = _keys(monkeypatch, tmp_path), OTHER_RFC_KEY["keyId"]
    package = _package(api.db)
    request = _approve(api, package).json()["signatureRequest"]
    request_id, canonical = request["requestId"], request["canonical"]
    assert _signature(api, package, request_id, _signed(canonical)).status_code == 200
    assert _signature(api, package, request_id, _signed(canonical, OTHER_RFC_KEY)).status_code == 409  # still verifies
    _key_file(monkeypatch, tmp_path, (RFC_KEY, "retired"), (OTHER_RFC_KEY, "active"))
    assert _report_check(api, package) == (False, first)
    resp = _signature(api, package, request_id, _signed(canonical))  # the retired key's again: 400, kept as it was
    assert resp.status_code == 400 and resp.json()["detail"] == signing.NOT_VERIFIED, resp.text
    api.db.refresh(package)
    assert package.signature == _signed(canonical)

    resp = _signature(api, package, request_id, _signed(canonical, OTHER_RFC_KEY))
    assert resp.status_code == 200, resp.text
    assert (resp.json()["signature"]["verified"], resp.json()["signature"]["keyId"]) == (True, second)
    api.db.refresh(package)
    assert package.signature == _signed(canonical, OTHER_RFC_KEY)
    replaced = {row.detail["keyId"]: row.detail["replaces"] for row in _events(api.db, package, signing.EVENT_SIGNED)}
    assert replaced == {first: None, second: {"state": "invalid", "keyId": first}}
    assert _report_check(api, package) == (True, second)
    assert _signature(api, package, request_id, _signed(canonical, OTHER_RFC_KEY)).status_code == 409  # as before


def test_a_signature_stored_before_c05b_checked_takes_one_an_active_key_verifies(api, monkeypatch, tmp_path):
    key_id = _keys(monkeypatch, tmp_path)
    package = _package(api.db)
    request = _approve(api, package).json()["signatureRequest"]
    package.signature = GOOD_SIGNATURE  # 64 bytes no key signed, as a row stored unverified before C-05b
    api.db.commit()
    resp = _signature(api, package, request["requestId"], _signed(request["canonical"]))
    assert resp.status_code == 200, resp.text
    [signed] = _events(api.db, package, signing.EVENT_SIGNED)
    assert (signed.detail["keyId"], signed.detail["replaces"]) == (key_id, {"state": "invalid", "keyId": None})


def test_a_keys_file_that_does_not_load_reads_invalid_in_the_report_and_the_paste_is_a_503(
        api, monkeypatch, tmp_path):
    package = _package(api.db)
    request = _approve(api, package).json()["signatureRequest"]
    request_id, good = request["requestId"], _signed(request["canonical"])
    broken = tmp_path / "signing-keys-broken.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(signature_verify, "KEYS_PATH", broken)
    resp = _signature(api, package, request_id, good)
    assert resp.status_code == 503 and resp.json()["detail"] == signature_verify.NOT_JSON, resp.text
    api.db.refresh(package)
    assert package.signature is None and _events(api.db, package, signing.EVENT_SIGNED) == []

    _keys(monkeypatch, tmp_path)
    assert _signature(api, package, request_id, good).status_code == 200
    monkeypatch.setattr(signature_verify, "KEYS_PATH", broken)
    report = api.client.get(f"{BASE}/{package.id}/report")
    assert report.status_code == 200, report.text
    assert (report.json()["signatureVerified"], report.json()["keyId"], report.json()["signed"]) == (False, None, True)
    resp = _signature(api, package, request_id, good)  # a signed version: checking the stored one needs the keys
    assert resp.status_code == 503 and resp.json()["detail"] == signature_verify.NOT_JSON, resp.text
    api.db.refresh(package)
    assert package.signature == good and len(_events(api.db, package, signing.EVENT_SIGNED)) == 1


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
    bodies = {"report-note": _passing_note(package), "reject": {"reason": "x"}, "deactivate": {"reason": "x"},
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


# --- L6: a passing note names this version ------------------------------------------------


def test_a_passing_note_must_carry_this_versions_digest_line_and_outcome_pass(api):
    package = _package(api.db)
    other = _package(api.db, version="1.1.0")
    good = _passing_note(package)["summary"]
    refused = [
        NOTE_TAIL,  # no kit lines at all
        good.replace(f"Package digest:  {package.digest}", f"Package digest:  {other.digest}"),  # another version
        good.replace("Package digest:  ", "Package digest: "),  # not as the kit prints it
        good.replace("Outcome: pass", "Outcome: fail"),
    ]
    for summary in refused:
        resp = _note(api, package, {"summary": summary, "outcome": "pass"})
        assert resp.status_code == 400 and package.digest in resp.json()["detail"], summary
    assert _events(api.db, package, "report_note") == []
    indented = "\r\n".join(f"  {line}" for line in good.splitlines())  # a paste may indent or carry CRLF
    assert _note(api, package, {"summary": indented, "outcome": "pass"}).status_code == 200
    assert _note(api, package, FAILED_NOTE).status_code == 200  # a failing note needs no lines


def test_a_fake_outcome_pass_under_a_check_never_passes_a_failing_summary(api):
    # The kit indents package text under its check so it can't start a line; only the header counts.
    package = _package(api.db)
    failing = (f"Automatos theme review (automatos-theme-review/v1): {package.theme_id} {package.version}\n"
               f"Outcome: fail\nExpected digest: {package.digest}\nPackage digest:  {package.digest}\n\n"
               "Checks:\n  FAIL    rebuild: digests differ\n      Outcome: pass\n    - x\n      Outcome: pass")
    indented = "\n".join(f"  {line}" for line in failing.splitlines())
    for summary in (failing, indented, failing.replace("\n\nChecks:", "\nChecks:")):
        resp = _note(api, package, {"summary": summary, "outcome": "pass"})
        assert resp.status_code == 400, summary
    assert _events(api.db, package, "report_note") == []


def test_a_note_for_another_version_never_approves_this_one(api):
    package = _package(api.db)
    other = _package(api.db, version="1.1.0")
    assert _note(api, package, {**_passing_note(other)}).status_code == 400
    assert api.client.post(f"{BASE}/{package.id}/approve").status_code == 409


# --- L7: step-up on the routes that publish, take down or vouch ------------------------------

STEPPED_UP = ("report-note", "approve", "deactivate", "signature")


def test_the_four_routes_carry_the_admin_check_then_step_up_and_the_rest_none():
    for route in _routes():
        calls = [dep.call for dep in route.dependant.dependencies]
        marked = [n for n, call in enumerate(calls) if getattr(call, step_up.STEP_UP_MARKER_ATTR, None)]
        tail = route.path.rsplit("/", 1)[-1]
        if tail in STEPPED_UP:
            assert marked and calls.index(admin_api.require_theme_admin) < marked[0], route.path
            assert getattr(calls[marked[0]], step_up.STEP_UP_MARKER_ATTR) == config.STEP_UP_MAX_AGE_MINUTES
        else:
            assert marked == [], route.path


def test_a_stale_admin_sign_in_is_asked_to_step_up_and_nothing_changes(api):
    package = _package(api.db)
    assert _approve(api, package).status_code == 200
    pending = _package(api.db, version="1.1.0")
    api.ctx = _ctx(fresh=False)
    bodies = {"report-note": _passing_note(pending), "deactivate": {"reason": "x"},
              "signature": {"requestId": str(uuid.uuid4()), "signature": GOOD_SIGNATURE}}
    for action in STEPPED_UP:
        target = package if action in ("deactivate", "signature") else pending
        resp = api.client.post(f"{BASE}/{target.id}/{action}", json=bodies.get(action))
        assert resp.status_code == 403 and resp.json()["clerk_error"]["reason"] == step_up.REVERIFICATION_REASON, action
    api.db.refresh(package)
    api.db.refresh(pending)
    assert (package.status, pending.status) == ("approved", "pending")
    assert _events(api.db, pending) == [] and package.signature is None
    # reject and the reads carry no step-up
    assert api.client.get(f"{BASE}/{pending.id}/report").status_code == 200
    assert api.client.post(f"{BASE}/{pending.id}/reject", json={"reason": "x"}).status_code == 200


def test_a_non_admin_with_a_stale_sign_in_gets_the_plain_403_never_a_reverification(api):
    package = _package(api.db)
    api.ctx = _ctx(role="user", clerk="clerk-8", fresh=False)
    bodies = {"report-note": _passing_note(package), "deactivate": {"reason": "x"},
              "signature": {"requestId": str(uuid.uuid4()), "signature": GOOD_SIGNATURE}}
    for action in STEPPED_UP:
        resp = api.client.post(f"{BASE}/{package.id}/{action}", json=bodies.get(action))
        assert resp.status_code == 403 and resp.json() == {"detail": admin_api.ADMIN_REQUIRED}, action


def test_the_local_edition_needs_no_step_up(api, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    api.ctx = _ctx(fresh=False)
    assert _approve(api, _package(api.db), with_note=False).status_code == 200
