"""PRD-261 A-08a — theme files by digest and an install's first-paint data, to exactly the right callers.

``api/themes_files.py`` is mounted on a test app (B2-wire registers it in ``main.py``), over
SQLite: the theme tables from the models, and the three tables the membership query reads
(``users``, ``workspaces``, ``workspace_members``) with only the columns it reads. Stored files
live in memory behind the route's reader dependency. The package is the real hello-clip test
package (``tests/fixtures/themes/build.py``) with a prepaint table, read by the real reader.

Pinned (plan §2.6 A-08a):

* the serving headers: nosniff, a sandboxed CSP, ``Cache-Control: private, max-age, immutable``,
  the stored type, ``Accept-Ranges``; never ``Service-Worker-Allowed``;
* one Range → 206 with ``Content-Range``; past the end → 416; a malformed one → 200;
* a member of two workspaces whose install is in the non-default one gets 200 for files and boot;
* a non-member gets 404 for both; a member who has not opted in to a private package gets 404
  for files and the base boot; another workspace's private package referenced from an install
  serves nothing; an unreviewed Marketplace package is private code too (the rule ``active``
  reports), so it is no approved listing and an install runs it only after the member's opt-in;
* tampered stored bytes → 404 and a logged error; ``Service-Worker: script`` → 403;
* the integrity value is the base64 form of the stored hex; a ``disable`` revocation → 404 and
  the base boot; the local edition's operator counts as owner of every workspace.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
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
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers the tables the theme FKs name)
from api import themes_files  # noqa: E402
from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.themes import ThemeInstallation, ThemeOptIn, ThemePackage, ThemeRevocation  # noqa: E402
from services.themes import serve  # noqa: E402
from services.themes.package import read_theme_package  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

PREPAINT = {
    "knobs": {"ground": {"paper": {"--hello-bg": "#fafafa"}, "ink": {"--hello-bg": "#101010"}}},
    "pairs": [], "numeric": [], "defaults": {"ground": "paper"},
}
ENTRY, CLIP, PREPAINT_PATH = "dist/entry.js", "assets/mascot/idle.mp4", "boot/prepaint.json"
ALICE, BOB, CAROL, OPERATOR = 1, 2, 3, 9
HOME, TEAM, ELSEWHERE, PAUSED = (uuid.UUID(int=n) for n in (0xA1, 0xA2, 0xA3, 0xA4))
RAW_TABLES = (
    "CREATE TABLE users (id INTEGER PRIMARY KEY, clerk_user_id VARCHAR, email VARCHAR)",
    "CREATE TABLE workspaces (id VARCHAR PRIMARY KEY, owner_id INTEGER, paused_at TIMESTAMP, deleted_at TIMESTAMP)",
    "CREATE TABLE workspace_members (workspace_id VARCHAR, user_id INTEGER, role VARCHAR, is_active BOOLEAN)",
)
THEME_TABLES = [model.__table__ for model in (ThemePackage, ThemeInstallation, ThemeOptIn, ThemeRevocation)]


@pytest.fixture(scope="module")
def package():
    data = build.build_package("hello-clip", replace={PREPAINT_PATH: json.dumps(PREPAINT).encode("utf-8")})
    return read_theme_package(data)


def _seed_people(db) -> None:
    """Alice owns HOME and is a member of TEAM; Carol owns TEAM, ELSEWHERE and PAUSED; Bob belongs nowhere."""
    for pk, name in ((ALICE, "alice"), (BOB, "bob"), (CAROL, "carol"), (OPERATOR, "operator")):
        db.execute(sa.text("INSERT INTO users VALUES (:id, :cid, :email)"),
                   {"id": pk, "cid": f"user_{name}", "email": f"{name}@example.com"})
    for ws, owner, paused in ((HOME, ALICE, None), (TEAM, CAROL, None), (ELSEWHERE, CAROL, None), (PAUSED, CAROL, "2026-10-01")):
        db.execute(sa.text("INSERT INTO workspaces VALUES (:id, :owner, :paused, NULL)"),
                   {"id": str(ws), "owner": owner, "paused": paused})
    for ws, active in ((TEAM, True), (PAUSED, True), (ELSEWHERE, False)):
        db.execute(sa.text("INSERT INTO workspace_members VALUES (:ws, :uid, 'viewer', :active)"),
                   {"ws": str(ws), "uid": ALICE, "active": active})


@pytest.fixture
def world(package):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        for ddl in RAW_TABLES:
            conn.execute(sa.text(ddl))
    ThemePackage.metadata.create_all(engine, tables=THEME_TABLES)
    db = sessionmaker(bind=engine)()
    _seed_people(db)
    db.commit()
    stored = {}
    try:
        yield SimpleNamespace(db=db, stored=stored, package=package, reader=lambda prefix, path: stored.get((prefix, path)))
    finally:
        db.close()
        engine.dispose()


def _store(world, prefix: str) -> None:
    world.stored.update({(prefix, path): item.data for path, item in world.package.files.items()})


def _row(world, owner_type: str, workspace=None, status="pending", file_manifest=None, reviewed=False) -> ThemePackage:
    pkg = world.package
    where = f"_workspace/{workspace}/" if workspace else ""
    prefix = f"themes/{where}{pkg.manifest['id']}/{pkg.manifest['version']}/"
    row = ThemePackage(
        theme_id=pkg.manifest["id"], version=pkg.manifest["version"], owner_type=owner_type, workspace_id=workspace,
        digest=pkg.digest, manifest=pkg.manifest, file_manifest=file_manifest or pkg.file_manifest,
        storage_prefix=prefix, status=status, reviewed=reviewed,
    )
    world.db.add(row)
    world.db.flush()
    _store(world, prefix)
    return row


def _install(world, workspace, package_row, opted_in=(), uninstalled=False) -> ThemeInstallation:
    install = ThemeInstallation(workspace_id=workspace, theme_id=package_row.theme_id, package_id=package_row.id)
    if uninstalled:
        install.uninstalled_at = datetime.now(timezone.utc)
    world.db.add(install)
    world.db.add_all([ThemeOptIn(workspace_id=workspace, user_id=user, package_id=package_row.id) for user in opted_in])
    world.db.commit()
    return install


def _private(world, workspace=TEAM, opted_in=(ALICE,)):
    row = _row(world, "workspace", workspace)
    return row, _install(world, workspace, row, opted_in)


def _person(name: str, workspace=HOME) -> RequestContext:
    user = UserContext(id=f"user_{name}", email=f"{name}@example.com", clerk_user_id=f"user_{name}")
    return RequestContext(workspace_id=workspace, user=user, auth_type="clerk")


def _client(world, ctx: RequestContext) -> TestClient:
    app = FastAPI()
    app.include_router(themes_files.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[themes_files.get_theme_file_reader] = lambda: world.reader

    def _db():
        yield world.db

    app.dependency_overrides[get_db] = _db
    return TestClient(app, raise_server_exceptions=False)


def _file(world, ctx, path=ENTRY, digest=None, headers=None):
    return _client(world, ctx).get(f"/api/themes/files/{digest or world.package.digest}/{path}", headers=headers or {})


def _boot(world, ctx, install_id):
    return _client(world, ctx).get(f"/api/themes/boot/{install_id}")


def _assert_serving_headers(resp) -> None:
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in resp.headers["content-security-policy"]
    assert "service-worker-allowed" not in resp.headers


# --- the files route -------------------------------------------------------------------------


def test_an_opted_in_member_gets_the_file_with_the_serving_headers(world):
    _private(world)
    resp = _file(world, _person("alice"))
    assert resp.status_code == 200
    assert resp.content == world.package.files[ENTRY].data
    assert resp.headers["content-type"].startswith("text/javascript")
    cache = resp.headers["cache-control"]
    assert cache.startswith("private, ") and "immutable" in cache and f"max-age={serve.file_cache_seconds()}" in cache
    assert resp.headers["accept-ranges"] == "bytes"
    assert resp.headers["etag"] == f'"{world.package.file_manifest[ENTRY]["sha256"]}"'
    _assert_serving_headers(resp)


def test_one_range_is_a_206_with_its_content_range(world):
    _private(world)
    clip = world.package.files[CLIP].data
    size = len(clip)
    for header, start, end in (("bytes=0-9", 0, 9), ("bytes=-16", size - 16, size - 1), (f"bytes=100-{size * 2}", 100, size - 1)):
        resp = _file(world, _person("alice"), CLIP, headers={"Range": header})
        assert resp.status_code == 206, header
        assert resp.headers["content-range"] == f"bytes {start}-{end}/{size}"
        assert resp.content == clip[start:end + 1]
        assert resp.headers["content-type"] == "video/mp4"
        _assert_serving_headers(resp)


def test_a_range_past_the_end_is_416_and_a_malformed_one_serves_the_whole_file(world):
    _private(world)
    size = len(world.package.files[CLIP].data)
    for header in (f"bytes={size}-", "bytes=-0"):
        resp = _file(world, _person("alice"), CLIP, headers={"Range": header})
        assert resp.status_code == 416, header
        assert resp.headers["content-range"] == f"bytes */{size}"
        _assert_serving_headers(resp)
    huge = "9" * 5000  # S1: past Python's int() digit limit; no Range, never a 500
    for header in ("bytes=0-1,4-5", "bytes=9-2", "lines=1-2", f"bytes={huge}-", f"bytes=0-{huge}", f"bytes=-{huge}"):
        resp = _file(world, _person("alice"), CLIP, headers={"Range": header})
        assert (resp.status_code, len(resp.content)) == (200, size), header


def test_a_matching_etag_is_304(world):
    _private(world)
    etag = f'"{world.package.file_manifest[ENTRY]["sha256"]}"'
    resp = _file(world, _person("alice"), headers={"If-None-Match": f'"other", W/{etag}'})
    assert resp.status_code == 304
    assert resp.content == b""


def test_a_member_whose_install_is_in_the_non_default_workspace_gets_files_and_boot(world):
    _, install = _private(world, TEAM)
    ctx = _person("alice", workspace=HOME)  # the header-less request resolves HOME, not TEAM
    assert _file(world, ctx).status_code == 200
    boot = _boot(world, ctx, install.id)
    assert boot.status_code == 200
    body = boot.json()
    assert body["addClasses"] == ["studio", "hello"]
    assert body["removeClasses"] == ["play"]
    assert body["tone"] == {"policy": "user", "value": None}
    assert body["prepaint"] == PREPAINT
    assert boot.headers["cache-control"] == "private, no-store"


def test_a_non_member_gets_404_for_files_and_boot(world):
    _, install = _private(world, TEAM, opted_in=(BOB,))
    bob = _person("bob")
    assert _file(world, bob).status_code == 404
    assert _boot(world, bob, install.id).status_code == 404
    assert _boot(world, bob, "not-a-uuid").status_code == 404
    assert _boot(world, _person("alice"), uuid.uuid4()).status_code == 404


def test_an_inactive_membership_or_a_paused_workspace_serves_nothing(world):
    for workspace in (ELSEWHERE, PAUSED):
        _, install = _private(world, workspace)
        assert _file(world, _person("alice")).status_code == 404, workspace
        assert _boot(world, _person("alice"), install.id).status_code == 404, workspace
        world.db.query(ThemeOptIn).delete()
        world.db.query(ThemeInstallation).delete()
        world.db.query(ThemePackage).delete()
        world.db.commit()


def test_a_member_who_has_not_opted_in_gets_404_for_files_and_the_base_boot(world):
    _, install = _private(world, TEAM, opted_in=())
    assert _file(world, _person("alice")).status_code == 404
    boot = _boot(world, _person("alice"), install.id)
    assert boot.status_code == 200
    assert boot.json() == {
        "addClasses": ["studio"], "removeClasses": ["hello", "play"], "tone": {"policy": "user", "value": None},
        "prepaint": None, "lookStorage": None, "lookGuard": None, "integrity": None,
    }


def test_an_uninstalled_install_serves_nothing(world):
    row = _row(world, "workspace", TEAM)
    install = _install(world, TEAM, row, opted_in=(ALICE,), uninstalled=True)
    assert _file(world, _person("alice")).status_code == 404
    assert _boot(world, _person("alice"), install.id).status_code == 404


def test_another_workspaces_private_package_referenced_from_an_install_serves_nothing(world):
    foreign = _row(world, "workspace", ELSEWHERE)  # Carol's upload; Alice is not an active member there
    install = _install(world, TEAM, foreign, opted_in=(ALICE,))
    assert _file(world, _person("alice")).status_code == 404
    assert _boot(world, _person("alice"), install.id).status_code == 404


def test_tampered_stored_bytes_are_404_and_logged(world, caplog):
    row, install = _private(world)
    world.stored[(row.storage_prefix, ENTRY)] = world.package.files[ENTRY].data + b"\n// changed"
    world.stored[(row.storage_prefix, PREPAINT_PATH)] = b'{"knobs": {}}'
    with caplog.at_level(logging.ERROR, logger="services.themes.serve"):
        assert _file(world, _person("alice")).status_code == 404
        boot = _boot(world, _person("alice"), install.id).json()
    assert any("tampered" in record.getMessage() and ENTRY in record.getMessage() for record in caplog.records)
    assert boot["addClasses"] == ["studio"] and boot["integrity"] is None


def test_a_prepaint_that_fails_the_css_value_check_on_read_gives_the_base_boot(world, caplog):
    bad = json.dumps({"knobs": {"ground": {"ink": {"--hello-bg": "url(https://x.test/a.png)"}}}}).encode("utf-8")
    rows = {**world.package.file_manifest, PREPAINT_PATH: {"type": "application/json", "size": len(bad),
                                                            "sha256": hashlib.sha256(bad).hexdigest()}}
    row = _row(world, "workspace", TEAM, file_manifest=rows)
    world.stored[(row.storage_prefix, PREPAINT_PATH)] = bad
    install = _install(world, TEAM, row, opted_in=(ALICE,))
    with caplog.at_level(logging.ERROR, logger="services.themes.serve"):
        boot = _boot(world, _person("alice"), install.id).json()
    assert boot["addClasses"] == ["studio"] and boot["prepaint"] is None
    assert any(PREPAINT_PATH in record.getMessage() for record in caplog.records)


def test_a_service_worker_script_request_is_refused(world):
    _private(world)
    resp = _file(world, _person("alice"), headers={"Service-Worker": "script"})
    assert resp.status_code == 403
    _assert_serving_headers(resp)


def test_a_path_outside_the_file_manifest_an_svg_or_a_bad_digest_is_404(world):
    _private(world)
    for path in ("dist/missing.js", "assets/brand/logo.svg", "../theme.json", "dist"):
        assert _file(world, _person("alice"), path).status_code == 404, path
    assert _file(world, _person("alice"), digest="A" * 64).status_code == 404
    assert _file(world, _person("alice"), digest="0" * 64).status_code == 404


def test_the_integrity_value_is_the_base64_of_the_stored_hex(world):
    _, install = _private(world)
    integrity = _boot(world, _person("alice"), install.id).json()["integrity"]
    entry_hex = world.package.file_manifest[ENTRY]["sha256"]
    raw = hashlib.sha256(world.package.files[ENTRY].data).digest()
    expected = "sha256-" + base64.b64encode(raw).decode("ascii")
    assert serve.integrity_value(entry_hex) == expected
    js_modules = [path for path in world.package.file_manifest if path.endswith(".js")]
    assert integrity == {f"/api/themes/files/{world.package.digest}/{path}": serve.integrity_value(
        world.package.file_manifest[path]["sha256"]) for path in js_modules}
    assert integrity[f"/api/themes/files/{world.package.digest}/{ENTRY}"] == expected


def test_a_disable_revocation_is_404_and_the_base_boot_and_a_warn_still_serves(world):
    _, install = _private(world)
    world.db.add(ThemeRevocation(digest=world.package.digest, level="warn", reason="a notice"))
    world.db.commit()
    assert _file(world, _person("alice")).status_code == 200
    world.db.add(ThemeRevocation(theme_id=world.package.manifest["id"], version_range="^1.0.0", level="disable", reason="bad"))
    world.db.commit()
    assert _file(world, _person("alice")).status_code == 404
    assert _boot(world, _person("alice"), install.id).json()["addClasses"] == ["studio"]


def test_an_approved_listing_serves_anyone_signed_in_and_a_pending_one_only_nobody(world):
    pending = _row(world, "marketplace", status="pending", reviewed=True)
    world.db.commit()
    assert _file(world, _person("bob")).status_code == 404
    _install(world, HOME, pending)
    assert _file(world, _person("alice")).status_code == 404  # never installable: never served
    pending.status = "approved"
    world.db.commit()
    assert _file(world, _person("bob")).status_code == 200
    pending.status = "deactivated"
    world.db.commit()
    assert _file(world, _person("bob")).status_code == 404
    assert _file(world, _person("alice")).status_code == 200  # its installs keep running


def test_an_unreviewed_marketplace_package_runs_only_after_the_members_opt_in(world):
    unreviewed = _row(world, "marketplace", status="approved")
    install = _install(world, HOME, unreviewed)
    assert _file(world, _person("bob")).status_code == 404  # no approved listing: never reviewed
    assert _file(world, _person("alice")).status_code == 404
    assert _boot(world, _person("alice"), install.id).json()["addClasses"] == ["studio"]
    world.db.add(ThemeOptIn(workspace_id=HOME, user_id=ALICE, package_id=unreviewed.id))
    world.db.commit()
    assert _file(world, _person("alice")).status_code == 200
    assert _boot(world, _person("alice"), install.id).json()["addClasses"] == ["studio", "hello"]


def test_the_local_operator_counts_as_owner_of_every_workspace(world, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    _, install = _private(world, ELSEWHERE, opted_in=(OPERATOR,))
    user = UserContext(id="operator@example.com", email="operator@example.com", system_role="super_admin",
                       raw_claims={"source": "local_operator", "user_id": OPERATOR})
    operator = RequestContext(workspace_id=HOME, user=user, auth_type="anonymous")
    assert _file(world, operator).status_code == 200
    assert _boot(world, operator, install.id).json()["addClasses"] == ["studio", "hello"]
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    assert _file(world, operator).status_code == 404


def test_an_api_key_caller_belongs_to_no_workspace(world):
    _private(world, HOME)
    key = RequestContext(workspace_id=HOME, user=UserContext(system_role="admin"), auth_type="api_key")
    assert _file(world, key).status_code == 404


def test_a_storage_failure_is_a_503(world):
    _private(world)

    def broken(_prefix, _path):
        raise serve.ThemeStorageUnavailable("down")

    world.reader = broken
    resp = _file(world, _person("alice"))
    assert resp.status_code == 503
    assert "Try again" in resp.json()["detail"]


# --- the parts on their own --------------------------------------------------------------------


def test_requested_range_reads_one_range_against_the_size():
    from services.themes.ranges import ByteRange, RangeNotSatisfiable, requested_range

    assert requested_range(None, 10) is None
    assert requested_range("bytes=2-4", 10) == ByteRange(2, 4)
    assert requested_range("bytes=8-", 10) == ByteRange(8, 9)
    assert requested_range("bytes=-3", 10) == ByteRange(7, 9)
    assert requested_range("bytes=-30", 10) == ByteRange(0, 9)
    assert requested_range("bytes=0-1,3-4", 10) is None
    assert requested_range("bytes=" + "9" * 5000 + "-", 10) is None
    assert requested_range("bytes=-" + "9" * 5000, 10) is None
    for header, size in (("bytes=10-", 10), ("bytes=-0", 10), ("bytes=0-", 0)):
        with pytest.raises(RangeNotSatisfiable):
            requested_range(header, size)


def test_a_revocation_covers_its_digest_or_its_theme_versions_until_lifted(world):
    from services.themes.revocations import revocation_level

    db, theme = world.db, "acme/ink"
    db.add_all([
        ThemeRevocation(theme_id=theme, version_range="2.0.0", level="disable", reason="exact"),
        ThemeRevocation(theme_id=theme, version_range="^1.2.0", level="warn", reason="caret"),
        ThemeRevocation(digest="d" * 64, level="disable", reason="lifted", lifted_at=datetime.now(timezone.utc)),
    ])
    db.commit()
    assert revocation_level(db, "c" * 64, theme, "2.0.0") == "disable"
    assert revocation_level(db, "c" * 64, theme, "1.4.0") == "warn"
    assert revocation_level(db, "c" * 64, theme, "1.1.0") is None
    assert revocation_level(db, "d" * 64, "acme/other", "1.0.0") is None
    db.add(ThemeRevocation(theme_id="acme/other", version_range=">=1", level="disable", reason="unreadable"))
    db.commit()
    assert revocation_level(db, "e" * 64, "acme/other", "9.9.9") == "disable"  # an unreadable range fails closed


class _Objects:
    """An S3 client that serves one key, refuses one as missing, and fails for the rest."""

    def __init__(self):
        self.asked = []

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3's argument names
        self.asked.append((Bucket, Key))
        if Key.endswith("/dist/entry.js"):
            return {"Body": SimpleNamespace(read=lambda: b"export {}")}
        code = "NoSuchKey" if Key.endswith("/gone.js") else "InternalError"
        raise _ClientError(code)


class _ClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


def test_read_stored_file_reads_the_marketplace_bucket_and_tells_missing_from_down(monkeypatch):
    objects = _Objects()
    monkeypatch.setattr(serve, "get_s3_client", lambda: objects)
    prefix = "themes/_workspace/ws/acme/ink/1.0.0/"
    assert serve.read_stored_file(prefix, "dist/entry.js") == b"export {}"
    assert objects.asked[0] == (config.MARKETPLACE_S3_BUCKET, f"{prefix}dist/entry.js")
    assert serve.read_stored_file(prefix, "dist/gone.js") is None
    with pytest.raises(serve.ThemeStorageUnavailable):
        serve.read_stored_file(prefix, "dist/other.js")
    with pytest.raises(serve.ThemeStorageUnavailable):
        serve.read_stored_file("skins/acme/", "dist/entry.js")
