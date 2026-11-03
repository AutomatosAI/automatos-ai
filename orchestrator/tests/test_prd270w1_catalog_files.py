"""PRD-270 US-270-02 (§5.2, K9) — the catalogue's public files, ``GET /api/marketplace/catalog/files/{digest}/{path}``.

The real router (``api/marketplace_catalog.py``) runs on a mini FastAPI app over in-memory SQLite holding
the theme tables, with no request-context override installed, so a 200 is an unauthenticated one.
Packages are real signed ``.autotheme`` files (``tests/helpers_theme_publish.py``: the RFC 8032 vector's
seed signs, its public key is the one active key), stored in a fake S3 that answers a missing key as S3
does (``NoSuchKey``). The per-IP limit runs the real ``RateLimitStore`` over a fake Redis. Pins:

* a catalogued version's ``package.autotheme`` (application/zip) and the web and phone screenshots its
  theme.json lists are served with their listed types, ``public, max-age=<THEME_FILE_CACHE_SECONDS>,
  immutable``, the strong ETag of their sha256 (the package's digest) and ``SERVING_HEADERS``; every URL
  the catalogue names is served;
* README.md, theme.json, the entry, a look preset, a screenshot theme.json doesn't list, an unlisted or
  malformed digest, a private upload's, a pending, rejected, deactivated, unsigned, disabled or Sync-written
  version's are 404 with ``SERVING_HEADERS`` and ``no-store``; so is everything while the switch is off;
* the rebuilt package reads (``package.read_theme_package``) to the row's digest and file hashes, and the
  row's stored signature verifies over them as a local install checks it; two requests give the same bytes;
  304 on the ETag, answered without a single storage read; a stored manifest that names another file as a
  screenshot serves only real screenshot paths;
* a stored file gone or changed serves no package and no screenshot (404, logged); storage down and a keys
  file that doesn't load are 503; ``Service-Worker: script`` is 403; the local edition is 404 before the
  limit counts; 429 with ``Retry-After`` in the route's own bucket;
* storage is read with the route's session holding no transaction, for a package and a screenshot; with
  every rebuild slot taken a package is 503 with ``Retry-After`` (logged as a warning) and storage is never
  read, while a screenshot and a 304 still answer; a rebuild gives its slot back, served or refused;
* the route is in the route manifest, a plain ``def`` with no request-context or permission dependency.
"""
from __future__ import annotations

import inspect
import json
import logging
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.marketplace_catalog as catalog_api
import api.public_rate_limit as rate_limit
from api.themes_files import SERVING_HEADERS, get_theme_file_reader
from api.widgets.rate_limit import RateLimitStore
from config import config
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.themes import ThemePackage, ThemePublisher, ThemeRevocation, ThemeRevocationSync
from services.marketplace_bundles import catalog_items
from services.marketplace_cloud import catalog, catalog_files
from services.themes import serve, signature_verify, signed_upload, storage
from services.themes.digest import sha256_hex
from services.themes.package import JPEG_SIGNATURE, PNG_SIGNATURE, read_theme_package
from tests import helpers_theme_publish as hp
from tests.fixtures.themes import build
from tests.test_prd264_content_provenance import _FakeRedis

FILES = "/api/marketplace/catalog/files"
ROUTE = FILES + "/{digest}/{path:path}"
ORCH = Path(__file__).resolve().parents[1]
TABLES = [m.__table__ for m in (ThemePackage, ThemePublisher, ThemeRevocation, ThemeRevocationSync)]
ARCADE = "automatos/arcade"
PNG, JPEG = PNG_SIGNATURE + bytes(8), JPEG_SIGNATURE + bytes(8)
WEB_CHAT, TEAM = "assets/previews/default-chat.png", "assets/previews/team.png"
PHONE, PHONE_JPEG = "assets/previews/phone/night.png", "assets/previews/phone/day.jpg"
UNLISTED = "assets/previews/unlisted.png"  # in the package, but theme.json doesn't list it
PACKAGE = "package.autotheme"
NOT_SERVED_PATHS = (
    "README.md", "theme.json", build.ENTRY, "dist/theme.css", "looks/presets/night.look.json", UNLISTED,
    "package.autotheme/theme.json", "PACKAGE.AUTOTHEME", "",
)


class _NoSuchKey(Exception):
    """What boto3 raises for a missing object (``serve.read_stored_file`` reads its code)."""

    response = {"Error": {"Code": "NoSuchKey"}}


class _StoredS3(hp.FakeS3):
    """The publish tests' fake S3: a missing key is ``NoSuchKey``, ``down`` fails every read, reads are counted."""

    def __init__(self) -> None:
        super().__init__()
        self.down, self.reads = False, 0

    def get_object(self, Bucket, Key):  # noqa: N803 (boto3's names)
        self.reads += 1
        if self.down:
            raise ConnectionError("storage is down")
        if Key not in self.objects:
            raise _NoSuchKey(Key)
        return super().get_object(Bucket, Key)


@pytest.fixture
def cat(monkeypatch, tmp_path):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=TABLES)
    session = sessionmaker(bind=engine)()
    s3, redis = _StoredS3(), _FakeRedis()
    monkeypatch.setattr(storage, "get_s3_client", lambda: s3)
    monkeypatch.setattr(storage, "ensure_bucket", lambda bucket: None)
    monkeypatch.setattr(serve, "get_s3_client", lambda: s3)
    monkeypatch.setattr(signature_verify, "KEYS_PATH", hp.keys_file(tmp_path))
    monkeypatch.setattr(rate_limit, "_rate_store", RateLimitStore(redis_factory=lambda: redis))
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", "on", raising=False)
    monkeypatch.setattr(config, "MARKETPLACE_PUBLIC_RATE_PER_MINUTE", 1000)
    monkeypatch.setattr(config, "THEME_FILE_CACHE_SECONDS", 4321)
    monkeypatch.setattr(catalog, "CATALOG_KEPT_SECONDS", 0)
    # PRD-270 W2: this database holds the theme tables only, so the catalogue lists no item entry here (as in
    # test_prd270w1_catalog_feed; test_prd270w2_catalog_items covers the entries, and their bundles, on Postgres).
    monkeypatch.setattr(catalog_items, "item_set", lambda db: [])
    catalog.forget_cached()
    app = FastAPI()
    app.include_router(catalog_api.router)
    app.dependency_overrides[get_db] = lambda: session
    try:
        yield SimpleNamespace(db=session, s3=s3, redis=redis, client=TestClient(app), tmp=tmp_path, app=app)
    finally:
        catalog.forget_cached()
        session.close()
        engine.dispose()


def _bytes(theme_id: str, version: str) -> bytes:
    """The hello package with two listed web screenshots, two listed phone ones and one theme.json doesn't list."""
    manifest = build.changed_manifest({
        "id": theme_id, "version": version, "previews": [WEB_CHAT, TEAM], "phonePreviews": [PHONE, PHONE_JPEG],
    })
    files = {build.MANIFEST: manifest, WEB_CHAT: PNG, TEAM: PNG + b"team", PHONE: PNG + b"phone",
             PHONE_JPEG: JPEG, UNLISTED: PNG + b"unlisted"}
    return build.build_package(replace=files)


def _row(db, theme_id=ARCADE, version="1.0.0", signer=hp.SIGNER, owner="marketplace", status="approved",
         source=None):
    """A stored version: its files in the fake S3, its row signed by ``signer`` (None: unsigned)."""
    data = _bytes(theme_id, version)
    checked = read_theme_package(data)
    workspace_id = uuid.uuid4() if owner == "workspace" else None
    prefix = storage.theme_prefix(theme_id, version, workspace_id)
    storage.write_theme_files(prefix, checked.files)
    row = ThemePackage(
        theme_id=theme_id, version=version, owner_type=owner, workspace_id=workspace_id, digest=checked.digest,
        manifest=checked.manifest, file_manifest=checked.file_manifest, storage_prefix=prefix, status=status,
        reviewed=owner == "marketplace" and status != "pending",
        signature=None if signer is None else json.loads(hp.envelope(data, signer))["signature"],
        scan_verdict="pass", source=source,
    )
    db.add(row)
    db.commit()
    return row


def _url(row, path: str) -> str:
    return f"{FILES}/{row.digest}/{path}"


def _assert_refused(resp, status: int) -> None:
    assert resp.status_code == status, resp.text
    for name, value in SERVING_HEADERS.items():
        assert resp.headers[name] == value, name
    assert resp.headers["cache-control"] == "no-store" and "etag" not in resp.headers


def _recording_reader(cat) -> list:
    """Install a reader over the fake S3 that records, on every call, whether the route's session (the test's
    overridden ``get_db``) holds a transaction; the list it fills is returned."""
    seen = []

    def read(storage_prefix: str, path: str):
        seen.append(cat.db.in_transaction())
        return serve.read_stored_file(storage_prefix, path)

    cat.app.dependency_overrides[get_theme_file_reader] = lambda: read
    return seen


def _slots_free() -> bool:
    """True when every rebuild slot can be taken now (each taken one is given straight back)."""
    taken = [catalog_files.REBUILD_SLOTS.acquire(blocking=False) for _ in range(catalog_files.CATALOG_PACKAGE_REBUILDS)]
    for held in taken:
        if held:
            catalog_files.REBUILD_SLOTS.release()
    return all(taken)


def _calls(dependant):
    found = []
    for sub in dependant.dependencies:
        found += [sub.call] + _calls(sub)
    return found


# ---------------------------------------------------------------------------
# What is served
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [WEB_CHAT, TEAM, PHONE, PHONE_JPEG])
def test_a_listed_web_or_phone_screenshot_is_served_with_its_type_and_headers(cat, path):
    row = _row(cat.db)
    resp = cat.client.get(_url(row, path))
    assert resp.status_code == 200, resp.text
    listed = row.file_manifest[path]
    assert resp.content == cat.s3.objects[storage.theme_key(row.storage_prefix, path)][0]
    assert sha256_hex(resp.content) == listed["sha256"]
    assert resp.headers["content-type"] == listed["type"] == ("image/jpeg" if path == PHONE_JPEG else "image/png")
    assert resp.headers["cache-control"] == "public, max-age=4321, immutable"
    assert resp.headers["etag"] == f'"{listed["sha256"]}"'
    for name, value in SERVING_HEADERS.items():
        assert resp.headers[name] == value, name


def test_the_package_is_served_as_a_zip_with_its_digest_as_the_etag(cat):
    row = _row(cat.db)
    resp = cat.client.get(_url(row, PACKAGE))
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/zip"
    assert resp.headers["cache-control"] == "public, max-age=4321, immutable"
    assert resp.headers["etag"] == f'"{row.digest}"'
    for name, value in SERVING_HEADERS.items():
        assert resp.headers[name] == value, name


def test_the_rebuilt_package_reads_to_the_row_s_digest_and_its_stored_signature_verifies(cat):
    row = _row(cat.db)
    checked = read_theme_package(cat.client.get(_url(row, PACKAGE)).content)
    assert checked.digest == row.digest
    assert checked.manifest["id"] == row.theme_id and checked.manifest["version"] == row.version
    assert {path: entry["sha256"] for path, entry in checked.file_manifest.items()} == {
        path: entry["sha256"] for path, entry in row.file_manifest.items()
    }
    hashes = {path: entry["sha256"] for path, entry in checked.file_manifest.items()}
    payload = signature_verify.payload_for(row.theme_id, row.version, hashes)
    assert signature_verify.verify(payload, row.signature).verified
    envelope = {
        "form": signed_upload.ENVELOPE_FORM, "algorithm": signed_upload.ENVELOPE_ALGORITHM,
        "keyId": hp.SIGNER["keyId"], "id": row.theme_id, "version": row.version, "digest": row.digest,
        "signature": row.signature,
    }
    verified = signed_upload.check_upload_signature(cat.db, checked, json.dumps(envelope).encode("utf-8"))
    assert verified.key_id == hp.SIGNER["keyId"]


def test_two_requests_give_the_same_bytes_and_the_etag_gives_a_304(cat):
    row = _row(cat.db)
    first, second = (cat.client.get(_url(row, PACKAGE)) for _ in range(2))
    assert first.status_code == second.status_code == 200 and first.content == second.content
    for path in (PACKAGE, PHONE):
        etag = cat.client.get(_url(row, path)).headers["etag"]
        again = cat.client.get(_url(row, path), headers={"If-None-Match": etag})
        assert again.status_code == 304 and again.content == b""
        assert again.headers["etag"] == etag and again.headers["cache-control"] == "public, max-age=4321, immutable"
        assert again.headers["x-content-type-options"] == "nosniff"
        weak = cat.client.get(_url(row, path), headers={"If-None-Match": f"W/{etag}"})
        assert weak.status_code == 304


def test_a_304_reads_nothing_from_storage_and_never_rebuilds_the_package(cat):
    row = _row(cat.db)
    etags = {path: cat.client.get(_url(row, path)).headers["etag"] for path in (PACKAGE, PHONE)}
    reads = cat.s3.reads
    assert reads > len(row.file_manifest)  # the 200s read every file of the package, then the screenshot
    for path, etag in etags.items():
        assert cat.client.get(_url(row, path), headers={"If-None-Match": etag}).status_code == 304
    assert cat.s3.reads == reads


def test_a_stored_manifest_that_names_another_file_as_a_screenshot_serves_nothing_but_screenshots(cat):
    row = _row(cat.db)
    row.manifest = {**row.manifest, "previews": [build.ENTRY, "README.md", WEB_CHAT], "phonePreviews": ["theme.json"]}
    cat.db.commit()
    for path in (build.ENTRY, "README.md", "theme.json"):
        _assert_refused(cat.client.get(_url(row, path)), 404)
    assert cat.client.get(_url(row, WEB_CHAT)).status_code == 200


def test_every_url_the_catalogue_names_is_served(cat):
    _row(cat.db)
    _row(cat.db, version="1.1.0")
    items = cat.client.get(catalog_api.CATALOG_ROUTE).json()["items"]
    assert len(items) == 2
    for item in items:
        shots = [shot["url"] for preset in item["presets"] for kind in ("previews", "phonePreviews")
                 for shot in preset[kind]]
        urls = [item["package"], item["cover"], item["phoneCover"], *shots, *item["otherPreviews"],
                *item["otherPhonePreviews"]]
        assert len(set(urls) - {item["package"]}) == 4
        for url in urls:
            assert cat.client.get(url).status_code == 200, url


# ---------------------------------------------------------------------------
# What is never served
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", NOT_SERVED_PATHS)
def test_no_other_file_of_a_catalogued_version_is_served(cat, path):
    row = _row(cat.db)
    _assert_refused(cat.client.get(_url(row, path)), 404)


@pytest.mark.parametrize("case", [
    {"owner": "workspace"}, {"status": "pending"}, {"status": "rejected"}, {"status": "deactivated"},
    {"signer": None}, {"signer": hp.OTHER_SIGNER}, {"source": "cloud"},
])
def test_a_version_the_catalogue_does_not_list_serves_nothing(cat, case):
    row = _row(cat.db, **case)
    for path in (PACKAGE, WEB_CHAT, PHONE):
        _assert_refused(cat.client.get(_url(row, path)), 404)


def test_a_disabled_version_an_unlisted_digest_and_a_malformed_one_serve_nothing(cat):
    revoked = _row(cat.db)
    cat.db.add(ThemeRevocation(digest=revoked.digest, level="disable", reason="leaks data"))
    cat.db.commit()
    for url in (_url(revoked, PACKAGE), f"{FILES}/{'ab' * 32}/{PACKAGE}", f"{FILES}/{revoked.digest.upper()}/{PACKAGE}",
                f"{FILES}/not-a-digest/{PACKAGE}"):
        _assert_refused(cat.client.get(url), 404)


def test_the_switch_off_serves_nothing(cat, monkeypatch):
    row = _row(cat.db)
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", "off")
    for path in (PACKAGE, WEB_CHAT):
        _assert_refused(cat.client.get(_url(row, path)), 404)


def test_a_stored_file_gone_or_changed_serves_no_package_and_no_screenshot(cat, caplog):
    row = _row(cat.db)
    del cat.s3.objects[storage.theme_key(row.storage_prefix, build.ENTRY)]
    with caplog.at_level(logging.ERROR, logger=catalog_files.__name__):
        _assert_refused(cat.client.get(_url(row, PACKAGE)), 404)
    assert any("can't be rebuilt" in record.getMessage() for record in caplog.records)
    assert cat.client.get(_url(row, PHONE)).status_code == 200
    cat.s3.objects[storage.theme_key(row.storage_prefix, PHONE)] = (PNG + b"swapped", "image/png")
    _assert_refused(cat.client.get(_url(row, PHONE)), 404)


def test_storage_down_is_a_503(cat, caplog):
    row = _row(cat.db)
    cat.s3.down = True
    with caplog.at_level(logging.ERROR, logger=catalog_api.__name__):
        for path in (PACKAGE, WEB_CHAT):
            resp = cat.client.get(_url(row, path))
            _assert_refused(resp, 503)
            assert resp.json()["detail"] == catalog_api.FILES_DOWN
    assert any("storage failed" in record.getMessage() for record in caplog.records)


def test_a_keys_file_that_does_not_load_is_a_503(cat, monkeypatch):
    row = _row(cat.db)
    monkeypatch.setattr(signature_verify, "KEYS_PATH", cat.tmp / "missing.json")
    resp = cat.client.get(_url(row, PACKAGE))
    _assert_refused(resp, 503)
    assert resp.json()["detail"] == catalog_api.FILES_DOWN


# ---------------------------------------------------------------------------
# What a request may hold: no connection over storage, a bounded number of rebuilds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [PACKAGE, PHONE])
def test_storage_is_read_with_no_transaction_open(cat, path):
    row = _row(cat.db)
    url = _url(row, path)
    seen = _recording_reader(cat)
    assert cat.client.get(url).status_code == 200
    assert seen and not any(seen), seen  # every read, every file of the package included


def test_with_every_rebuild_slot_taken_a_package_is_a_503_and_storage_is_never_read(cat, monkeypatch, caplog):
    row = _row(cat.db)
    package_url, phone_url = _url(row, PACKAGE), _url(row, PHONE)
    seen = _recording_reader(cat)
    monkeypatch.setattr(catalog_files, "REBUILD_SLOT_WAIT_SECONDS", 0.01)
    taken = [catalog_files.REBUILD_SLOTS.acquire(blocking=False) for _ in range(catalog_files.CATALOG_PACKAGE_REBUILDS)]
    try:
        assert all(taken)
        with caplog.at_level(logging.WARNING, logger=catalog_api.__name__):
            refused = cat.client.get(package_url)
        assert seen == []
        assert cat.client.get(package_url, headers={"If-None-Match": f'"{row.digest}"'}).status_code == 304
        assert cat.client.get(phone_url).status_code == 200
    finally:
        for held in taken:
            if held:
                catalog_files.REBUILD_SLOTS.release()
    _assert_refused(refused, 503)
    assert refused.json()["detail"] == catalog_api.FILES_BUSY
    assert refused.headers["retry-after"] == str(catalog_files.REBUILD_RETRY_SECONDS)
    assert any(record.levelno == logging.WARNING and "rebuild slot" in record.getMessage() for record in caplog.records)
    assert seen == [False]  # the screenshot's one read; the package's request read nothing
    assert cat.client.get(package_url).status_code == 200


def test_a_rebuild_gives_its_slot_back_served_or_refused(cat):
    row = _row(cat.db)
    package_url = _url(row, PACKAGE)
    for _ in range(catalog_files.CATALOG_PACKAGE_REBUILDS + 1):
        assert cat.client.get(package_url).status_code == 200
    assert _slots_free()
    cat.s3.down = True
    _assert_refused(cat.client.get(package_url), 503)
    cat.s3.down = False
    del cat.s3.objects[storage.theme_key(row.storage_prefix, build.ENTRY)]
    _assert_refused(cat.client.get(package_url), 404)
    assert _slots_free()


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def test_a_service_worker_registration_is_refused_403(cat):
    row = _row(cat.db)
    for path in (PACKAGE, WEB_CHAT):
        _assert_refused(cat.client.get(_url(row, path), headers={"Service-Worker": "script"}), 403)


def test_the_local_edition_answers_404_before_the_limit_counts(cat, monkeypatch):
    row = _row(cat.db)
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    resp = cat.client.get(_url(row, PACKAGE))
    _assert_refused(resp, 404)
    assert "hosted edition" in resp.json()["detail"]
    assert cat.redis.z == {} and cat.s3.reads == 0


def test_over_the_per_ip_limit_is_a_429_with_retry_after_in_the_route_s_own_bucket(cat, monkeypatch):
    row = _row(cat.db)
    monkeypatch.setattr(config, "MARKETPLACE_PUBLIC_RATE_PER_MINUTE", 2)
    headers = {"X-Forwarded-For": "198.51.100.7"}
    assert [cat.client.get(_url(row, WEB_CHAT), headers=headers).status_code for _ in range(2)] == [200, 200]
    refused = cat.client.get(_url(row, PACKAGE), headers=headers)
    _assert_refused(refused, 429)
    assert int(refused.headers["retry-after"]) > 0
    assert len(cat.redis.z) == 1 and next(iter(cat.redis.z)).endswith(":marketplace-catalog-files:ip:198.51.100.7")
    assert cat.client.get(catalog_api.CATALOG_ROUTE, headers=headers).status_code == 200  # the catalogue's own bucket
    assert cat.client.get(_url(row, WEB_CHAT), headers={"X-Forwarded-For": "198.51.100.8"}).status_code == 200


def test_the_route_is_public_a_plain_def_in_the_manifest():
    routes = [r for r in catalog_api.router.routes if isinstance(r, APIRoute) and r.path == ROUTE]
    assert len(routes) == 1 and routes[0].methods == {"GET"}
    route = routes[0]
    assert not inspect.iscoroutinefunction(route.endpoint)
    calls = _calls(route.dependant)
    assert get_request_context_hybrid not in calls
    assert calls[:2] == [catalog_api.require_hosted_files, catalog_api.limit_file_reads]
    assert get_db in calls and get_theme_file_reader in calls
    assert not any("permission" in getattr(call, "__qualname__", "") for call in calls)
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    assert {"method": "GET", "path": ROUTE} in manifest["routes"]
