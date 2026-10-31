"""PRD-270 W2 US-270-12d (§5.6, K12) — the public plugin package route, a deterministic zip of an exported plugin.

``GET /api/marketplace/catalog/plugins/{slug}/{version}/package.zip`` runs on a mini FastAPI app. The S3
client is a fake bucket monkeypatched into ``plugin_files`` (the ``tests/test_prd261_theme_files.py:632``
pattern) that lists two keys a page, so every plugin here is read over several pages; the per-IP limit runs
the real ``RateLimitStore`` over a fake Redis. The rows (W2's export test's ``_plugin``) are written on one
Postgres connection whose transaction the teardown rolls back; the session joins it by SAVEPOINT and each
seed commits (releasing its savepoint), so the route's own rollback never takes the seed (the PRD-266
``test_prd266_mobile_devices_routes.py`` pattern). Proven here:

* an exported plugin's zip holds exactly the files stored under its prefix (another plugin's and another
  version's are not in it), each reading back byte for byte, a binary file and a UTF-8 one included;
  ``application/zip``, the immutable cache, ETag the zip's sha256 and 304, SERVING_HEADERS;
* two requests give the same bytes;
* a pending, 'blocked' or inactive plugin, another version and an unknown slug are 404 with SERVING_HEADERS
  and no-store, and storage is never read for them;
* files over ``PLUGIN_MAX_UPLOAD_SIZE_MB``, or whose manifest.json names another version, are a logged 404;
  storage that fails is a logged 503;
* the local edition is 404 before the limit counts; 429 with ``Retry-After`` in the route's own bucket;
* a slug or version off the bundle format's patterns is 404 before the export is read; a database failure
  is a logged 503; the route ends its read transaction before storage is read;
* the route is a plain ``def`` with no request-context or permission dependency, in the route manifest.

Without a database: ``plugin_zip`` (sorted, dated 1980-01-01, deflated, fixed attributes: the same files give
the same bytes whatever their order), ``stored_plugin_files`` (raw reads page by page, folder markers
skipped, the cap checked on the listed sizes before any read and on the bytes read, a prefix not the
plugin's or an unsafe key (absolute, climbing, not normal, a backslash or a control character) refused,
storage failures logged) and ``checked_files`` (the manifest's slug and version).
"""
from __future__ import annotations

import hashlib
import inspect
import io
import json
import logging
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

import api.marketplace_catalog_items as items_api
import api.public_rate_limit as rate_limit
from api.themes_files import SERVING_HEADERS
from api.widgets.rate_limit import RateLimitStore
from config import config
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.storage import StorageNotConfigured
from router_manifest import MANIFEST_ROUTERS, RouterSpec
from services.marketplace_bundles import plugin_files
from services.marketplace_bundles.catalog_paths import plugin_package_path
from services.marketplace_bundles.export_plugins import exported_plugins
from services.marketplace_bundles.plugin_files import (
    PluginFilesRefused, PluginFilesUnavailable, StoredPlugin, checked_files, plugin_prefix, plugin_zip,
    stored_plugin_files,
)
from tests.test_prd264_content_provenance import _FakeRedis
from tests.test_prd270w2_export_more import _plugin

ORCH = Path(__file__).resolve().parents[1]
ROUTE = "/api/marketplace/catalog/plugins/{slug}/{version}/package.zip"
TABLES = ("users", "workspaces", "marketplace_plugins", "plugin_security_scans", "plugin_categories")
PAGE = 2  # keys per listed page in the fake bucket
BINARY = bytes(range(256)) * 4  # not UTF-8: MarketplaceS3Service.get_file would refuse it
MB = 1024 * 1024


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for table in TABLES:
                c.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the plugin package tests need Postgres with the marketplace plugin tables: {exc}")
    yield eng
    eng.dispose()


# --- the fake bucket -----------------------------------------------------------------------------


class _ClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _Body:
    def __init__(self, data):
        self.data, self.closed, self.asked = data, False, []

    def read(self, amt=None):
        self.asked.append(amt)
        return self.data if amt is None else self.data[:amt]

    def close(self):
        self.closed = True


class _Bucket:
    """The two calls plugin_files makes, on a dict of (bucket, key) -> bytes, listing PAGE keys a page."""

    def __init__(self):
        self.objects, self.sizes = {}, {}
        self.listed, self.read, self.bodies = [], [], []
        self.down = None  # "list" or "read": that call fails
        self.on_list = None  # called on every listing (the transaction check)

    def put(self, prefix, files):
        for path, data in files.items():
            self.objects[(config.MARKETPLACE_S3_BUCKET, prefix + path)] = data

    def list_objects_v2(self, **asked):
        self.listed.append(asked)
        if self.on_list is not None:
            self.on_list()
        if self.down == "list":
            raise _ClientError("InternalError")
        keys = sorted(
            key for bucket, key in self.objects if bucket == asked["Bucket"] and key.startswith(asked["Prefix"])
        )
        start = int(asked.get("ContinuationToken") or 0)
        page, more = keys[start:start + PAGE], start + PAGE < len(keys)
        answer = {"IsTruncated": more, "KeyCount": len(page)}
        if page:
            answer["Contents"] = [
                {"Key": key, "Size": self.sizes.get(key, len(self.objects[(asked["Bucket"], key)]))} for key in page
            ]
        if more:
            answer["NextContinuationToken"] = str(start + PAGE)
        return answer

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3's argument names
        self.read.append((Bucket, Key))
        if self.down == "read":
            raise _ClientError("InternalError")
        body = _Body(self.objects[(Bucket, Key)])
        self.bodies.append(body)
        return {"Body": body}


def _files(slug, version, manifest=None):
    """A plugin's files: its manifest (``manifest`` overriding its keys), markdown, a UTF-8 file with
    non-ASCII text and a binary file."""
    named = {"slug": slug, "name": "Test Plugin", "version": version, **(manifest or {})}
    return {
        "manifest.json": json.dumps(named).encode(),
        "README.md": b"A plugin of the package test.\n",
        "skills/test-skill/SKILL.md": b"# Test skill\n\nDo the test thing.\n",
        "commands/run.md": "Run it — café ✓".encode("utf-8"),
        "assets/logo.png": BINARY,
    }


def _row(slug="test-kit", version="1.0.0", s3_path=None):
    return StoredPlugin(slug, version, plugin_prefix(slug, version) if s3_path is None else s3_path)


def _unzipped(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}, archive.infolist()


@pytest.fixture
def bucket(monkeypatch):
    fake = _Bucket()
    monkeypatch.setattr(plugin_files, "get_s3_client", lambda: fake)
    return fake


# --- the zip and the stored files, without a database ------------------------------------------------


def test_plugin_zip_is_deterministic_sorted_dated_deflated_and_reads_back_byte_for_byte():
    files = _files("test-kit", "1.0.0")
    shuffled = dict(reversed(list(files.items())))
    data = plugin_zip(files)
    assert data == plugin_zip(files) == plugin_zip(shuffled)
    read, infos = _unzipped(data)
    assert read == files and read["assets/logo.png"] == BINARY
    assert [info.filename for info in infos] == sorted(files)
    for info in infos:
        assert info.date_time == (1980, 1, 1, 0, 0, 0) and info.compress_type == zipfile.ZIP_DEFLATED
        assert info.create_system == 3 and info.external_attr == 0o100644 << 16
    assert plugin_zip({**files, "README.md": b"Another README.\n"}) != data


def test_stored_plugin_files_reads_every_object_under_the_prefix_raw_page_by_page(bucket):
    row, files = _row(), _files("test-kit", "1.0.0")
    bucket.put(row.s3_path, {**files, "skills/": b""})  # a folder marker, skipped
    bucket.put(plugin_prefix("test-kit", "2.0.0"), _files("test-kit", "2.0.0"))
    bucket.put(plugin_prefix("test-kit-other", "1.0.0"), _files("test-kit-other", "1.0.0"))
    stored = stored_plugin_files(row)
    assert stored == files and stored["assets/logo.png"] == BINARY
    assert len(bucket.listed) == 3 and [asked.get("ContinuationToken") for asked in bucket.listed] == [None, "2", "4"]
    assert all(asked["Bucket"] == config.MARKETPLACE_S3_BUCKET and asked["Prefix"] == row.s3_path
               for asked in bucket.listed)
    assert sorted(key for _bucket, key in bucket.read) == sorted(row.s3_path + path for path in files)
    assert all(body.closed for body in bucket.bodies)


def test_files_over_the_cap_are_refused_on_their_listed_sizes_before_any_read(bucket, monkeypatch):
    monkeypatch.setattr(config, "PLUGIN_MAX_UPLOAD_SIZE_MB", 1)
    row = _row()
    bucket.put(row.s3_path, {**_files("test-kit", "1.0.0"), "assets/big.bin": b"\0" * MB})
    with pytest.raises(PluginFilesRefused, match="cap"):
        stored_plugin_files(row)
    assert bucket.read == []


def test_files_over_the_cap_are_refused_on_the_bytes_read_whatever_the_listing_says(bucket, monkeypatch):
    monkeypatch.setattr(config, "PLUGIN_MAX_UPLOAD_SIZE_MB", 1)
    row = _row()
    big = row.s3_path + "assets/big.bin"
    bucket.put(row.s3_path, {"assets/big.bin": b"\0" * (MB + 1)})
    bucket.sizes[big] = 1  # a listing that understates the object
    with pytest.raises(PluginFilesRefused, match="cap"):
        stored_plugin_files(row)
    assert bucket.bodies[0].asked == [MB + 1]  # never more than one byte past the room left


def test_exactly_at_the_cap_is_served(bucket, monkeypatch):
    monkeypatch.setattr(config, "PLUGIN_MAX_UPLOAD_SIZE_MB", 1)
    row = _row()
    manifest = _files("test-kit", "1.0.0")["manifest.json"]
    bucket.put(row.s3_path, {"manifest.json": manifest, "assets/fill.bin": b"\0" * (MB - len(manifest))})
    assert sum(len(data) for data in stored_plugin_files(row).values()) == MB


@pytest.mark.parametrize("s3_path", [None, "", "plugins/", "plugins/test-kit/", "plugins/other/1.0.0/",
                                     "plugins/test-kit/1.0.0", "themes/test-kit/1.0.0/"])
def test_a_row_whose_prefix_is_not_the_plugin_s_is_refused_and_nothing_is_listed(bucket, s3_path):
    row = StoredPlugin("test-kit", "1.0.0", s3_path)
    with pytest.raises(PluginFilesRefused, match="prefix"):
        stored_plugin_files(row)
    assert bucket.listed == [] and bucket.read == []


@pytest.mark.parametrize("path", [
    "../escape.md", "skills/../../escape.md", "/absolute.md", "./manifest.json", "skills//SKILL.md",
    "skills/./SKILL.md", "skills\\..\\..\\escape.md", "nul\x00.md", "line\nbreak.md", "del\x7f.md",
])
def test_a_stored_key_with_an_unsafe_path_refuses_the_plugin(bucket, path):
    row = _row()
    bucket.put(row.s3_path, {**_files("test-kit", "1.0.0"), path: b"x"})
    with pytest.raises(PluginFilesRefused, match="unsafe"):
        stored_plugin_files(row)


@pytest.mark.parametrize("down", ["list", "read"])
def test_storage_that_fails_is_unavailable_and_logged(bucket, down, caplog):
    row = _row()
    bucket.put(row.s3_path, _files("test-kit", "1.0.0"))
    bucket.down = down
    with caplog.at_level(logging.ERROR, logger=plugin_files.__name__), pytest.raises(PluginFilesUnavailable):
        stored_plugin_files(row)
    assert any(record.exc_info for record in caplog.records)


def test_no_storage_client_is_unavailable_and_logged(monkeypatch, caplog):
    def unconfigured():
        raise StorageNotConfigured()

    monkeypatch.setattr(plugin_files, "get_s3_client", unconfigured)
    with caplog.at_level(logging.ERROR, logger=plugin_files.__name__), pytest.raises(PluginFilesUnavailable):
        stored_plugin_files(_row())
    assert any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("manifest, reason", [
    (None, "no manifest"), (b"{not json", "not JSON"), (b"\xff\xfe", "not JSON"), (b"[]", "not an object"),
    (json.dumps({"slug": "test-other", "version": "1.0.0"}).encode(), "another"),
    (json.dumps({"slug": "test-kit", "version": "2.0.0"}).encode(), "another"),
    (json.dumps({"name": "Test Plugin"}).encode(), "another"),
])
def test_the_manifest_must_be_there_and_name_the_plugin_s_slug_and_version(manifest, reason):
    files = {path: data for path, data in _files("test-kit", "1.0.0").items() if path != "manifest.json"}
    if manifest is not None:
        files = {**files, "manifest.json": manifest}
    with pytest.raises(PluginFilesRefused, match=reason):
        checked_files(_row(), files)
    right = _files("test-kit", "1.0.0")
    assert checked_files(_row(), right) is right


# --- the route, without a database -------------------------------------------------------------------


class _NoDatabase:
    """The session the route gets when the export is monkeypatched: it only counts its rollbacks."""

    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


@pytest.fixture
def served(bucket, monkeypatch):
    """The plugin route on a mini app in the hosted edition, over the fake bucket and a fake Redis."""
    redis = _FakeRedis()
    monkeypatch.setattr(rate_limit, "_rate_store", RateLimitStore(redis_factory=lambda: redis))
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "MARKETPLACE_PUBLIC_RATE_PER_MINUTE", 1000)
    state = NS(bucket=bucket, redis=redis, db=_NoDatabase())
    app = FastAPI()
    app.include_router(items_api.router)
    app.dependency_overrides[get_db] = lambda: state.db
    state.client = TestClient(app)
    return state


def _assert_serving_headers(resp):
    for name, value in SERVING_HEADERS.items():
        assert resp.headers[name.lower()] == value, name


def _assert_refused(resp, status, detail):
    assert resp.status_code == status and resp.json()["detail"] == detail
    assert resp.headers["cache-control"] == "no-store" and "etag" not in resp.headers
    _assert_serving_headers(resp)


def _never(*_args, **_kwargs):
    raise AssertionError("the export must not be read")


@pytest.mark.parametrize("slug, version", [
    ("Test-Kit", "1.0.0"), ("test_kit", "1.0.0"), ("-test-kit", "1.0.0"), ("t" * 101, "1.0.0"),
    ("test-kit", "-1.0.0"), ("test-kit", ".hidden"), ("test-kit", "1" * 33), ("test-kit", "1.0.0+local"),
])
def test_a_slug_or_version_off_the_format_is_404_before_the_export_is_read(served, monkeypatch, slug, version):
    monkeypatch.setattr(items_api, "exported_plugin", _never)
    _assert_refused(served.client.get(plugin_package_path(slug, version)), 404, items_api.PLUGIN_NOT_LISTED)


def test_a_database_failure_is_a_logged_503_with_the_serving_headers(served, monkeypatch, caplog):
    def down(db, slug, version):
        raise OperationalError("SELECT 1", {}, ConnectionError("the database is down"))

    monkeypatch.setattr(items_api, "exported_plugin", down)
    with caplog.at_level(logging.ERROR, logger=items_api.__name__):
        resp = served.client.get(plugin_package_path("test-kit", "1.0.0"))
    _assert_refused(resp, 503, items_api.PLUGINS_DOWN)
    assert any(record.exc_info for record in caplog.records)


def test_the_route_ends_its_read_transaction_before_storage_is_read(served, monkeypatch):
    row = _row()
    served.bucket.put(row.s3_path, _files(row.slug, row.version))
    monkeypatch.setattr(items_api, "exported_plugin", lambda db, slug, version: row)
    rollbacks_at_listing = []
    served.bucket.on_list = lambda: rollbacks_at_listing.append(served.db.rollbacks)
    resp = served.client.get(plugin_package_path(row.slug, row.version))
    assert resp.status_code == 200, resp.text
    assert rollbacks_at_listing and all(count >= 1 for count in rollbacks_at_listing)
    assert _unzipped(resp.content)[0] == _files(row.slug, row.version)


def test_the_local_edition_answers_404_before_the_limit_counts(served, monkeypatch):
    monkeypatch.setattr(items_api, "exported_plugin", _never)
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    resp = served.client.get(plugin_package_path("test-kit", "1.0.0"))
    assert resp.status_code == 404 and "hosted edition" in resp.json()["detail"]
    assert resp.headers["cache-control"] == "no-store"
    _assert_serving_headers(resp)
    assert served.redis.z == {}


def test_over_the_per_ip_limit_is_a_429_with_retry_after_in_the_route_s_own_bucket(served, monkeypatch):
    monkeypatch.setattr(items_api, "exported_plugin", lambda db, slug, version: None)
    monkeypatch.setattr(config, "MARKETPLACE_PUBLIC_RATE_PER_MINUTE", 2)
    path, headers = plugin_package_path("test-kit", "1.0.0"), {"X-Forwarded-For": "198.51.100.7"}
    assert [served.client.get(path, headers=headers).status_code for _ in range(2)] == [404, 404]
    refused = served.client.get(path, headers=headers)
    assert refused.status_code == 429
    assert int(refused.headers["retry-after"]) > 0 and refused.headers["cache-control"] == "no-store"
    _assert_serving_headers(refused)
    assert len(served.redis.z) == 1
    assert next(iter(served.redis.z)).endswith(":marketplace-catalog-plugins:ip:198.51.100.7")
    assert served.client.get(path, headers={"X-Forwarded-For": "198.51.100.8"}).status_code == 404


def _calls(dependant):
    found = []
    for sub in dependant.dependencies:
        found += [sub.call] + _calls(sub)
    return found


def test_the_route_is_public_a_plain_def_in_the_manifest_and_mounted_by_router_spec():
    routes = [r for r in items_api.router.routes if isinstance(r, APIRoute) and r.path == ROUTE]
    assert len(routes) == 1 and routes[0].methods == {"GET"}
    route = routes[0]
    assert not inspect.iscoroutinefunction(route.endpoint)
    calls = _calls(route.dependant)
    assert get_request_context_hybrid not in calls
    assert calls[:2] == [items_api.require_hosted_bundles, items_api.limit_plugin_reads] and get_db in calls
    assert not any("permission" in getattr(call, "__qualname__", "") for call in calls)
    assert items_api.limit_plugin_reads is not items_api.limit_bundle_reads
    assert items_api.PLUGINS_RATE_BUCKET != items_api.BUNDLES_RATE_BUCKET
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    assert {"method": "GET", "path": ROUTE} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])
    assert RouterSpec("api.marketplace_catalog_items") in MANIFEST_ROUTERS
    assert plugin_package_path("test-kit", "1.0.0") == ROUTE.format(slug="test-kit", version="1.0.0")


# --- the route over Postgres ---------------------------------------------------------------------------


@pytest.fixture
def hosted(engine, served):
    """``served`` over Postgres: one connection whose transaction the teardown rolls back, the session joined
    by SAVEPOINT, so the route's rollback ends only the savepoint after the last seed's commit."""
    from sqlalchemy.orm import Session

    connection = engine.connect()
    transaction = connection.begin()
    served.db = Session(bind=connection, join_transaction_mode="create_savepoint")
    served.tag = uuid.uuid4().hex[:10]
    try:
        yield served
    finally:
        served.db.close()
        transaction.rollback()
        connection.close()


def _seeded(hosted, scan=None, **columns):
    """W2's export test's plugin, committed (in this session's mode, its savepoint released)."""
    plugin = _plugin(hosted, scan=scan, **columns)
    hosted.db.commit()
    return plugin


def _stored(hosted, plugin, manifest=None):
    files = _files(plugin.slug, plugin.version, manifest)
    hosted.bucket.put(plugin.s3_path, files)
    return files


def _get(hosted, plugin, version=None, **headers):
    return hosted.client.get(plugin_package_path(plugin.slug, version or plugin.version), headers=headers)


def test_an_exported_plugin_s_zip_holds_exactly_its_stored_files(hosted):
    plugin, other = _seeded(hosted), _seeded(hosted)
    files, _other_files = _stored(hosted, plugin), _stored(hosted, other)
    hosted.bucket.put(plugin_prefix(plugin.slug, "2.0.0"), _files(plugin.slug, "2.0.0"))
    assert [item.bundle["plugin"]["slug"] for item in exported_plugins(hosted.db, slug=plugin.slug)] == [plugin.slug]
    resp = _get(hosted, plugin)
    assert resp.status_code == 200, resp.text
    read, infos = _unzipped(resp.content)
    assert read == files and read["assets/logo.png"] == BINARY
    assert [info.filename for info in infos] == sorted(files)
    assert resp.headers["content-type"] == "application/zip"
    assert resp.headers["cache-control"] == f"public, max-age={config.THEME_FILE_CACHE_SECONDS}, immutable"
    assert resp.headers["etag"] == f'"{hashlib.sha256(resp.content).hexdigest()}"'
    _assert_serving_headers(resp)


def test_two_requests_give_the_same_bytes_and_a_matching_etag_is_a_304(hosted):
    plugin = _seeded(hosted)
    _stored(hosted, plugin)
    first, second = _get(hosted, plugin), _get(hosted, plugin)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content and first.headers["etag"] == second.headers["etag"]
    again = _get(hosted, plugin, **{"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304 and again.content == b"" and again.headers["etag"] == first.headers["etag"]
    _assert_serving_headers(again)


@pytest.mark.parametrize("case", ["pending", "blocked", "inactive", "another version", "unknown slug"])
def test_a_plugin_the_export_does_not_list_is_404_and_storage_is_never_read(hosted, case):
    columns = {
        "pending": {"approval_status": "pending"}, "inactive": {"is_active": False},
        "blocked": {"security_status": "blocked", "scan": {"overall_verdict": "blocked", "llm_risk_score": 90}},
    }.get(case, {})
    plugin = _seeded(hosted, scan=columns.pop("scan", None), **columns)
    _stored(hosted, plugin)
    hosted.bucket.put(plugin_prefix(plugin.slug, "2.0.0"), _files(plugin.slug, "2.0.0"))
    if case == "another version":
        resp = _get(hosted, plugin, version="2.0.0")
    elif case == "unknown slug":
        resp = hosted.client.get(plugin_package_path(f"test-unknown-{hosted.tag}", "1.0.0"))
    else:
        resp = _get(hosted, plugin)
    _assert_refused(resp, 404, items_api.PLUGIN_NOT_LISTED)
    assert hosted.bucket.listed == [] and hosted.bucket.read == []


def test_files_over_the_cap_are_a_logged_404(hosted, monkeypatch, caplog):
    monkeypatch.setattr(config, "PLUGIN_MAX_UPLOAD_SIZE_MB", 1)
    plugin = _seeded(hosted)
    _stored(hosted, plugin)
    hosted.bucket.put(plugin.s3_path, {"assets/big.bin": b"\0" * MB})
    with caplog.at_level(logging.ERROR, logger=plugin_files.__name__):
        resp = _get(hosted, plugin)
    _assert_refused(resp, 404, items_api.PLUGIN_NOT_LISTED)
    assert any(plugin.slug in record.getMessage() and "cap" in record.getMessage() for record in caplog.records)


def test_a_manifest_naming_another_version_is_a_logged_404(hosted, caplog):
    plugin = _seeded(hosted)
    _stored(hosted, plugin, {"version": "9.9.9"})
    with caplog.at_level(logging.ERROR, logger=plugin_files.__name__):
        resp = _get(hosted, plugin)
    _assert_refused(resp, 404, items_api.PLUGIN_NOT_LISTED)
    assert any(plugin.slug in record.getMessage() for record in caplog.records)


def test_storage_that_fails_is_a_logged_503_with_the_serving_headers(hosted, caplog):
    plugin = _seeded(hosted)
    _stored(hosted, plugin)
    hosted.bucket.down = "read"
    with caplog.at_level(logging.ERROR):
        resp = _get(hosted, plugin)
    _assert_refused(resp, 503, items_api.PLUGINS_DOWN)
    assert any(record.exc_info and record.name == plugin_files.__name__ for record in caplog.records)
    assert any(record.name == items_api.__name__ for record in caplog.records)
