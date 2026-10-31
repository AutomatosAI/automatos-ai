"""Shared set-up for the theme publish endpoint's tests (``tests/test_theme_publish_*.py``).

The publish router runs on a test app over SQLite (``create_all`` for the theme tables, a small
``users`` table by its own DDL), with a fake S3 client in ``services/themes/storage.py``. Its
sessions don't autoflush, as the app's ``SessionLocal`` doesn't, so a step that reads what an
earlier one wrote without a flush fails here as it would in production. The signing keys file
points at the RFC 8032 public key of the shared vector (``tests/fixtures/themes/signing-vector.json``);
its published seed, held in memory, makes the envelopes. The keys below are made up for the
tests, built at import time, and never a real key. Core's review stage is a stub with a set verdict.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ORCH = Path(__file__).resolve().parents[1]
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

import sqlalchemy as sa  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.marketplace_themes_publish as publish_api  # noqa: E402
import core.models  # noqa: E402,F401  (registers every table the FKs name)
from config import config  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.themes import (  # noqa: E402
    ThemeChoice, ThemeInstallation, ThemeLook, ThemeOptIn, ThemePackage, ThemePublisher, ThemeReviewEvent,
    ThemeRevocation, ThemeRevocationSync, ThemeWorkspaceSettings,
)
from services.themes import signature_verify, signed_upload, storage  # noqa: E402
from services.themes.package import read_theme_package  # noqa: E402
from services.themes.publish import THEME_PUBLISH_ROUTE  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

KEY = "5e" * 32  # a made-up publish key: 64 hex characters, as the kit's 32 random bytes are
NEXT_KEY = "a7" * 32
KEY_HASH = hashlib.sha256(KEY.encode("utf-8")).hexdigest()
NEXT_KEY_HASH = hashlib.sha256(NEXT_KEY.encode("utf-8")).hexdigest()
ACTOR_PK, PERSON_PK = 50, 7
THEME_ID, OFFICIAL_ID, UNLISTED_ID = "acme-e2e/hello", "automatos/hello-publish", "other-e2e/hello"
PUBLISHERS = "automatos, acme-e2e"
SOURCE_FILE, SOURCE_BYTES = "source/src/index.tsx", b"export default {}\n"
VECTOR = json.loads((ORCH / "tests" / "fixtures" / "themes" / "signing-vector.json").read_text(encoding="utf-8"))
SIGNER, OTHER_SIGNER = VECTOR["ed25519"]["cases"][:2]
TABLES = (ThemePackage, ThemePublisher, ThemeReviewEvent, ThemeInstallation, ThemeRevocation, ThemeRevocationSync,
          ThemeWorkspaceSettings, ThemeChoice, ThemeLook, ThemeOptIn)
USERS_DDL = (
    "CREATE TABLE users (id INTEGER PRIMARY KEY, username VARCHAR(255) NOT NULL UNIQUE, "
    "email VARCHAR(255) NOT NULL UNIQUE, clerk_user_id VARCHAR(255) UNIQUE, name VARCHAR(255), is_active BOOLEAN)"
)
SEED_USERS = "INSERT INTO users (id, username, email, is_active) VALUES (:id, :name, :email, :active)"
SCAN_SCHEMA = "automatos-theme-scan/v1"


class FakeS3:
    """Objects by key, as ``services/themes/storage.py`` writes them; ``fail`` makes every write raise."""

    def __init__(self) -> None:
        self.objects: Dict[str, Any] = {}
        self.fail = False

    def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803 (boto3's names)
        if self.fail:
            raise ConnectionError("storage is down")
        self.objects[Key] = (Body, ContentType)

    def get_object(self, Bucket, Key):  # noqa: N803
        return {"Body": io.BytesIO(self.objects[Key][0])}

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.objects.pop(Key, None)


def stage(verdict: str = "pass"):
    """Core's review stage, stubbed: answers ``verdict`` and records the digest of each package it reviews."""
    calls = []

    def run(checked, package_bytes, token_budget=None):
        calls.append(checked.digest)
        return {"schema": SCAN_SCHEMA, "verdict": verdict, "checks": {}}

    run.calls = calls
    return run


def keys_file(tmp_path: Path) -> Path:
    """A keys file holding the RFC key ``SIGNER`` as the one active key."""
    entry = {"keyId": SIGNER["keyId"], "spki": SIGNER["spki"], "status": "active", "added": "2026-10-20"}
    path = tmp_path / "signing-keys.json"
    path.write_text(json.dumps({"schema": 1, "keys": [entry]}), encoding="utf-8")
    return path


def configure(monkeypatch, hashes: str = KEY_HASH, publishers: str = PUBLISHERS, actor: str = str(ACTOR_PK)) -> None:
    """The three publish settings."""
    monkeypatch.setattr(config, "THEME_PUBLISH_KEY_SHA256", hashes)
    monkeypatch.setattr(config, "THEME_PUBLISH_PUBLISHERS", publishers)
    monkeypatch.setattr(config, "THEME_PUBLISH_ACTOR", actor)


def _engine():
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=[model.__table__ for model in TABLES])
    with engine.begin() as conn:
        conn.execute(sa.text(USERS_DDL))
        conn.execute(sa.text(SEED_USERS), [
            {"id": ACTOR_PK, "name": "theme-publisher", "email": "theme-publisher@automatos.invalid", "active": True},
            {"id": PERSON_PK, "name": "person", "email": "person@example.com", "active": True},
        ])
    return engine


def publish_env(monkeypatch, tmp_path: Path) -> Iterator[SimpleNamespace]:
    """The test app and its database, configured for the publish key; Q9 on (it must not apply to the key)."""
    engine = _engine()
    fake = FakeS3()
    monkeypatch.setattr(storage, "get_s3_client", lambda: fake)
    monkeypatch.setattr(storage, "ensure_bucket", lambda bucket: None)
    monkeypatch.setattr(signature_verify, "KEYS_PATH", keys_file(tmp_path))
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")  # the stage is a stub; never the Anthropic API
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", "on", raising=False)
    monkeypatch.setattr(config, "THEMES_ONE_OPEN_SUBMISSION", True, raising=False)
    monkeypatch.setattr(config, "THEME_PUBLISH_PER_DAY", 10)
    configure(monkeypatch)
    state = SimpleNamespace(sessions=sessionmaker(bind=engine, autoflush=False), s3=fake, stage=stage())
    state.client = client(state)
    try:
        yield state
    finally:
        engine.dispose()


def client(state: SimpleNamespace, raise_server_exceptions: bool = True, routers=()) -> TestClient:
    """The publish router (and ``routers``) on a test app; the review stage is ``state.stage`` at each request."""
    app = FastAPI()
    for router in (publish_api.router, *routers):
        app.include_router(router)

    def _db():
        session = state.sessions()
        try:
            yield session
        finally:
            session.rollback()
            session.close()

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[publish_api.publish_review_stage] = lambda: state.stage
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def package(theme_id: str = THEME_ID, version: str = "1.0.0", source: bytes = SOURCE_BYTES) -> bytes:
    """The hello package as ``theme_id`` at ``version``, with one ``source/`` file (``source`` changes its bytes)."""
    manifest = build.changed_manifest({"id": theme_id, "version": version})
    return build.build_package(replace={SOURCE_FILE: source, build.MANIFEST: manifest})


def digest_of(data: bytes) -> str:
    return read_theme_package(data).digest


def envelope(data: bytes, case: Dict[str, str] = SIGNER, **changes: Any) -> str:
    """The kit's ``.sig.json`` for these package bytes, signed with ``case``'s RFC seed, with ``changes``."""
    checked = read_theme_package(data)
    hashes = {path: row["sha256"] for path, row in checked.file_manifest.items()}
    payload = signature_verify.payload_for(checked.manifest["id"], checked.manifest["version"], hashes)
    seed = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(case["seed"]))
    signature = base64.b64encode(seed.sign(signature_verify.canonical_bytes(payload))).decode("ascii")
    body = {
        "form": signed_upload.ENVELOPE_FORM, "algorithm": "ed25519", "keyId": case["keyId"],
        "id": payload["id"], "version": payload["version"], "digest": payload["digest"], "signature": signature,
    }
    return json.dumps({**body, **changes})


def summary(digest: str, outcome: str = "pass", theme_id: str = THEME_ID) -> str:
    """The kit review's summary (``scripts/review/summary.ts``) for a package with ``digest``."""
    return (
        f"Automatos theme review (automatos-theme-review/v1): {theme_id}\nOutcome: {outcome}\n"
        f"Expected digest: {digest}\nPackage digest:  {digest}\n\nChecks:\n  build: reproducible, digests equal\n"
    )


def post(client_: TestClient, data: bytes, signature: Optional[str] = None, review_summary: Optional[str] = None,
         key: Optional[str] = KEY, headers: Optional[Dict[str, str]] = None, as_files: bool = False):
    """POST a publish: the package as a file part; the envelope and summary as text fields (or files)."""
    sent = {"Authorization": f"Bearer {key}"} if key else {}
    sent.update(headers or {})
    signature = envelope(data) if signature is None else signature
    review_summary = summary(digest_of(data)) if review_summary is None else review_summary
    files = {"package": ("acme-e2e.hello.autotheme", data, "application/zip")}
    if as_files:
        files.update({"signature": ("pack.sig.json", signature.encode(), "application/json"),
                      "review_summary": ("summary.txt", review_summary.encode(), "text/plain")})
        return client_.post(THEME_PUBLISH_ROUTE, headers=sent, files=files)
    return client_.post(THEME_PUBLISH_ROUTE, headers=sent, files=files,
                        data={"signature": signature, "review_summary": review_summary})


def rows(state: SimpleNamespace, model) -> list:
    session = state.sessions()
    try:
        return session.query(model).all()
    finally:
        session.close()


def snapshot(state: SimpleNamespace) -> Dict[str, Any]:
    """What a publish can write: every package, event and publisher row, and every stored file."""
    return {
        "packages": sorted((row.theme_id, row.version, row.status, bool(row.signature)) for row in rows(state, ThemePackage)),
        "events": sorted((str(row.package_id), row.event) for row in rows(state, ThemeReviewEvent)),
        "publishers": sorted((row.publisher, row.owner_user_id) for row in rows(state, ThemePublisher)),
        "files": sorted(state.s3.objects),
    }


def nothing_stored(state: SimpleNamespace) -> bool:
    return snapshot(state) == {"packages": [], "events": [], "publishers": [], "files": []}


def refused(response, status: int, code: str) -> Dict[str, Any]:
    """The refusal's detail, after checking its status, its code and the Marketplace error shape."""
    assert response.status_code == status, response.text
    detail = response.json()["detail"]
    assert detail["code"] == code and isinstance(detail["message"], str) and detail["errors"], detail
    assert all(set(error) == {"code", "path", "message"} for error in detail["errors"]), detail
    return detail


def add(state: SimpleNamespace, *objects) -> None:
    session = state.sessions()
    try:
        session.add_all(objects)
        session.commit()
    finally:
        session.close()
