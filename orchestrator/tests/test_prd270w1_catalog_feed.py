"""PRD-270 US-270-01 (§5.2, K8) — the hosted catalogue, ``GET /api/marketplace/catalog`` (schema 1).

The real router (``api/marketplace_catalog.py``) runs on a mini FastAPI app over in-memory SQLite
holding the theme tables, with no request-context override installed, so a 200 is an
unauthenticated one. Packages are real signed ``.autotheme`` files (``tests/helpers_theme_publish.py``:
the RFC 8032 vector's seed signs, its public key is the one active key), stored in a fake S3. The
per-IP limit runs the real ``RateLimitStore`` over a fake Redis. Pins:

* only approved, reviewed, signature-verified, local-origin Marketplace versions appear, each version
  its own item, by theme id and newest first; a private upload, a pending, rejected, deactivated or
  unreviewed version, one a ``disable`` covers, an unsigned one, one only a retired key verifies and a
  row a Sync wrote never do;
* every item verifies as a local install checks it (``signed_upload.check_upload_signature`` over its
  envelope, ``files`` and ``themeJson``), and base64-decoded ``themeJson`` hashes to
  ``files['theme.json'].sha256``;
* every key of every item is on the allowlist, every URL is a catalogue path, and nothing of the row
  (its id, workspace, submitter or scan report) leaves;
* 404 on a local edition (before the limit counts); ``Cache-Control: public, max-age=300``, nosniff,
  a weak ETag that ignores ``generatedAt`` and 304; 503 over the size cap (never a 304), when the keys
  file doesn't load and when storage can't give back a theme.json, each refusal ``no-store``;
  theme.json is read once per version; one build serves every caller for ``CATALOG_KEPT_SECONDS``
  (the other tests keep nothing, so each request builds); 429 with ``Retry-After`` over the per-IP
  limit, in the route's own bucket; the switch off lists no item;
* a build reads every theme.json with no transaction open; while another caller builds, a caller gets
  the last snapshot held, however old, at once, or with none held ``CatalogBusy`` within
  ``CATALOG_BUILD_WAIT_SECONDS``, reading neither the database nor storage; the route answers that 503
  with ``Retry-After``;
* the route is in the route manifest and mounted through ``RouterSpec``; a plain ``def`` with no
  request-context or workspace-permission dependency.
"""
from __future__ import annotations

import base64
import inspect
import json
import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
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
from api.widgets.rate_limit import RateLimitStore
from config import config
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.themes import ThemePackage, ThemePublisher, ThemeRevocation, ThemeRevocationSync
from router_manifest import MANIFEST_ROUTERS, RouterSpec
from services.marketplace_bundles import catalog_items
from services.marketplace_cloud import catalog
from services.themes import serve, signature_verify, signed_upload, storage
from services.themes.digest import sha256_hex
from services.themes.package import PNG_SIGNATURE, ThemePackage as CheckedThemePackage, read_theme_package
from services.themes.revocation_feed import feed_etag
from tests import helpers_theme_publish as hp
from tests.fixtures.themes import build
from tests.test_prd264_content_provenance import _FakeRedis

CATALOG = "/api/marketplace/catalog"
FILES = "/api/marketplace/catalog/files/"
ORCH = Path(__file__).resolve().parents[1]
TABLES = [m.__table__ for m in (ThemePackage, ThemePublisher, ThemeRevocation, ThemeRevocationSync)]
ARCADE, INK = "automatos/arcade", "acme/ink"
PNG = PNG_SIGNATURE + bytes(8)
WEB_CHAT, WEB, TEAM = "assets/previews/default-chat.png", "assets/previews/default.png", "assets/previews/team.png"
PHONE = "assets/previews/phone/night.png"
SCAN_SECRET = "scan-note-never-public"
ITEM_KEYS = {
    "type", "id", "listingId", "version", "digest", "name", "description", "publisher", "engines", "uses",
    "pricing", "signature", "files", "themeJson", "package", "cover", "phoneCover", "presets", "otherPreviews",
    "otherPhonePreviews",
}
PRESET_KEYS = {"id", "default", "previews", "phonePreviews"}
# The build wait the bounded-wait tests use, the margin a caller may take past it, and how long a held read waits.
BUILD_WAIT_SECONDS, WAIT_MARGIN_SECONDS, HOLD_LIMIT_SECONDS = 0.2, 1.0, 10.0


class _CountingS3(hp.FakeS3):
    """The publish tests' fake S3, counting reads."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    def get_object(self, Bucket, Key):  # noqa: N803 (boto3's names)
        self.reads += 1
        return super().get_object(Bucket, Key)


def _keys(tmp_path: Path, retired=None) -> Path:
    """The RFC key as the one active key; ``retired`` adds another vector key as retired."""
    entries = [{"keyId": hp.SIGNER["keyId"], "spki": hp.SIGNER["spki"], "status": "active", "added": "2026-10-20"}]
    if retired is not None:
        entries.append({"keyId": retired["keyId"], "spki": retired["spki"], "status": "retired", "added": "2026-10-01"})
    path = tmp_path / "signing-keys.json"
    path.write_text(json.dumps({"schema": 1, "keys": entries}), encoding="utf-8")
    return path


@pytest.fixture
def cat(monkeypatch, tmp_path):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=TABLES)
    session = sessionmaker(bind=engine)()
    s3, redis = _CountingS3(), _FakeRedis()
    monkeypatch.setattr(storage, "get_s3_client", lambda: s3)
    monkeypatch.setattr(storage, "ensure_bucket", lambda bucket: None)
    monkeypatch.setattr(serve, "get_s3_client", lambda: s3)
    monkeypatch.setattr(signature_verify, "KEYS_PATH", _keys(tmp_path, retired=hp.OTHER_SIGNER))
    monkeypatch.setattr(rate_limit, "_rate_store", RateLimitStore(redis_factory=lambda: redis))
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", "on", raising=False)
    monkeypatch.setattr(config, "MARKETPLACE_PUBLIC_RATE_PER_MINUTE", 1000)
    monkeypatch.setattr(catalog, "CATALOG_KEPT_SECONDS", 0)
    # PRD-270 W2: this database holds the theme tables only, so the catalogue lists no item entry here
    # (test_prd270w2_catalog_items covers them on Postgres).
    monkeypatch.setattr(catalog_items, "item_set", lambda db: [])
    catalog.forget_cached()
    app = FastAPI()
    app.include_router(catalog_api.router)
    app.dependency_overrides[get_db] = lambda: session
    try:
        yield SimpleNamespace(db=session, s3=s3, redis=redis, client=TestClient(app), tmp=tmp_path)
    finally:
        catalog.forget_cached()
        session.close()
        engine.dispose()


def _bytes(theme_id: str, version: str) -> bytes:
    """The hello package as ``theme_id`` at ``version``, with three web screenshots and one phone screenshot."""
    manifest = build.changed_manifest({
        "id": theme_id, "version": version, "name": f"{theme_id} {version}",
        "previews": [WEB_CHAT, WEB, TEAM], "phonePreviews": [PHONE],
    })
    return build.build_package(replace={build.MANIFEST: manifest, WEB_CHAT: PNG, WEB: PNG, TEAM: PNG, PHONE: PNG})


def _row(db, theme_id=ARCADE, version="1.0.0", signer=hp.SIGNER, owner="marketplace", status="approved",
         reviewed=None, source=None):
    """A stored version: its files in the fake S3, its row signed by ``signer`` (None: unsigned)."""
    data = _bytes(theme_id, version)
    checked = read_theme_package(data)
    workspace_id = uuid.uuid4() if owner == "workspace" else None
    prefix = storage.theme_prefix(theme_id, version, workspace_id)
    storage.write_theme_files(prefix, checked.files)
    row = ThemePackage(
        theme_id=theme_id, version=version, owner_type=owner, workspace_id=workspace_id, digest=checked.digest,
        manifest=checked.manifest, file_manifest=checked.file_manifest, storage_prefix=prefix, status=status,
        reviewed=(owner == "marketplace" and status != "pending") if reviewed is None else reviewed,
        signature=None if signer is None else json.loads(hp.envelope(data, signer))["signature"],
        scan_report={"note": SCAN_SECRET}, scan_verdict="pass", source=source,
    )
    db.add(row)
    db.commit()
    return row


def _catalog(cat, **headers):
    resp = cat.client.get(CATALOG, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _calls(dependant):
    found = []
    for sub in dependant.dependencies:
        found += [sub.call] + _calls(sub)
    return found


# ---------------------------------------------------------------------------
# What is listed
# ---------------------------------------------------------------------------


def test_only_approved_reviewed_signed_local_marketplace_versions_appear_each_version_its_own_item(cat):
    newest, older = _row(cat.db, version="1.1.0"), _row(cat.db, version="1.0.0")
    ink = _row(cat.db, theme_id=INK)
    _row(cat.db, theme_id="acme/private", owner="workspace")
    _row(cat.db, theme_id="acme/pending", status="pending")
    _row(cat.db, theme_id="acme/rejected", status="rejected")
    _row(cat.db, theme_id="acme/gone", status="deactivated")
    _row(cat.db, theme_id="acme/unreviewed", reviewed=False)
    _row(cat.db, theme_id="acme/unsigned", signer=None)
    _row(cat.db, theme_id="acme/retired", signer=hp.OTHER_SIGNER)  # listed by the Themes tab (QB-28), never here
    _row(cat.db, theme_id="acme/synced", source="cloud")
    revoked = _row(cat.db, theme_id="acme/revoked")
    cat.db.add(ThemeRevocation(digest=revoked.digest, level="disable", reason="leaks data"))
    cat.db.commit()
    body = _catalog(cat)
    assert set(body) == {"schema", "generatedAt", "items"} and body["schema"] == 1
    assert datetime.fromisoformat(body["generatedAt"]).tzinfo is not None
    assert [(item["id"], item["version"]) for item in body["items"]] == [
        (INK, "1.0.0"), (ARCADE, "1.1.0"), (ARCADE, "1.0.0"),
    ]
    assert [item["digest"] for item in body["items"]] == [ink.digest, newest.digest, older.digest]
    assert {item["type"] for item in body["items"]} == {"theme"}
    arcade = body["items"][1]
    assert arcade["listingId"] == "automatos.arcade" and arcade["name"] == f"{ARCADE} 1.1.0"
    assert arcade["publisher"] == {"name": "automatos", "official": True}
    assert body["items"][0]["publisher"] == {"name": "acme", "official": False}
    assert arcade["engines"] == build.hello_manifest()["engines"] and arcade["uses"] == []
    assert arcade["pricing"] == "Free"


def test_every_item_verifies_as_a_local_install_checks_it(cat):
    _row(cat.db, version="1.1.0")
    _row(cat.db, theme_id=INK)
    items = _catalog(cat)["items"]
    assert len(items) == 2
    for item in items:
        theme_json = base64.b64decode(item["themeJson"], validate=True)
        assert sha256_hex(theme_json) == item["files"]["theme.json"]["sha256"]
        manifest = json.loads(theme_json)
        assert (manifest["id"], manifest["version"], manifest["name"]) == (item["id"], item["version"], item["name"])
        envelope = item["signature"]
        assert (envelope["id"], envelope["version"], envelope["digest"]) == (item["id"], item["version"], item["digest"])
        hashes = {path: row["sha256"] for path, row in item["files"].items()}
        payload = signature_verify.payload_for(item["id"], item["version"], hashes)
        assert payload["digest"] == item["digest"]
        verdict = signature_verify.verify(payload, envelope["signature"])
        assert verdict.verified and verdict.key_id == envelope["keyId"] == hp.SIGNER["keyId"]
        checked = CheckedThemePackage(manifest, {}, item["files"], item["digest"])
        verified = signed_upload.check_upload_signature(cat.db, checked, json.dumps(envelope).encode("utf-8"))
        assert verified.key_id == hp.SIGNER["keyId"]


def test_every_key_of_every_item_is_on_the_allowlist_and_nothing_of_the_row_leaves(cat):
    row = _row(cat.db)
    resp = cat.client.get(CATALOG)
    item = resp.json()["items"][0]
    assert set(item) == ITEM_KEYS
    assert set(item["publisher"]) == {"name", "official"}
    assert tuple(sorted(item["signature"])) == signed_upload.ENVELOPE_KEYS
    assert item["signature"]["form"] == signed_upload.ENVELOPE_FORM and item["signature"]["algorithm"] == "ed25519"
    assert set(item["files"]) == set(row.file_manifest)
    assert all(set(entry) == {"type", "size", "sha256"} for entry in item["files"].values())
    assert all(set(preset) == PRESET_KEYS for preset in item["presets"])
    shots = [shot for preset in item["presets"] for kind in ("previews", "phonePreviews") for shot in preset[kind]]
    assert shots and all(set(shot) == {"screen", "url"} for shot in shots)
    text = resp.text  # a UUID, a dotted prefix and the hyphenated note can't occur inside base64
    for private in (str(row.id), row.storage_prefix, SCAN_SECRET):
        assert private not in text, private
    readable = json.dumps({key: value for key, value in item.items() if key not in ("themeJson", "signature")})
    for word in ("packageId", "workspace", "submitted", "scan", "installed", "revoked"):
        assert word not in readable, word


def test_every_url_is_a_catalogue_path_and_previews_keep_the_listing_rules(cat):
    row = _row(cat.db)
    item = _catalog(cat)["items"][0]
    base = f"{FILES}{row.digest}/"
    assert item["package"] == f"{base}package.autotheme"
    assert item["cover"] == base + WEB_CHAT  # the first one theme.json lists, not the first by name
    assert item["phoneCover"] == base + PHONE
    presets = {preset["id"]: preset for preset in item["presets"]}
    assert set(presets) == {"default", "night"} and presets["default"]["default"] is True
    assert presets["default"]["previews"] == [{"screen": "chat", "url": base + WEB_CHAT},
                                              {"screen": None, "url": base + WEB}]
    assert presets["night"]["phonePreviews"] == [{"screen": None, "url": base + PHONE}]
    assert presets["night"]["previews"] == [] and presets["default"]["phonePreviews"] == []
    assert item["otherPreviews"] == [base + TEAM] and item["otherPhonePreviews"] == []
    assert "/api/themes/files" not in json.dumps(item)


def test_the_switch_off_lists_no_item(cat, monkeypatch):
    _row(cat.db)
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", "off")
    assert _catalog(cat)["items"] == []


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def test_the_local_edition_answers_404_before_the_limit_counts(cat, monkeypatch):
    _row(cat.db)
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    resp = cat.client.get(CATALOG)
    assert resp.status_code == 404 and "hosted edition" in resp.json()["detail"]
    assert resp.headers["cache-control"] == "no-store"
    assert cat.redis.z == {}


def test_the_cache_header_and_a_weak_etag_that_ignores_generated_at_give_a_304(cat):
    _row(cat.db)
    first = cat.client.get(CATALOG)
    assert first.status_code == 200
    assert first.headers["cache-control"] == "public, max-age=300"
    assert first.headers["x-content-type-options"] == "nosniff"
    assert first.headers["content-type"].startswith("application/json")
    etag = first.headers["etag"]
    assert etag.startswith('W/"')
    later = catalog.build_catalog(cat.db, now=datetime.now(timezone.utc) + timedelta(hours=1))
    assert later["generatedAt"] != first.json()["generatedAt"] and feed_etag(later) == etag
    again = cat.client.get(CATALOG, headers={"If-None-Match": etag})
    assert again.status_code == 304 and again.content == b""
    assert again.headers["etag"] == etag and again.headers["cache-control"] == "public, max-age=300"
    _row(cat.db, version="1.1.0")
    changed = cat.client.get(CATALOG, headers={"If-None-Match": etag})
    assert changed.status_code == 200 and changed.headers["etag"] != etag


def test_a_catalogue_over_the_cap_is_a_logged_503_never_a_304(cat, monkeypatch, caplog):
    _row(cat.db)
    etag = cat.client.get(CATALOG).headers["etag"]
    monkeypatch.setattr(config, "MARKETPLACE_CATALOG_FEED_MAX_KB", 0)
    with caplog.at_level(logging.ERROR, logger=catalog_api.__name__):
        resp = cat.client.get(CATALOG, headers={"If-None-Match": etag})
    assert resp.status_code == 503 and resp.json()["detail"] == catalog_api.CATALOG_TOO_LARGE
    assert resp.headers["cache-control"] == "no-store" and "etag" not in resp.headers
    assert any("over the" in record.getMessage() for record in caplog.records)


def test_a_keys_file_that_does_not_load_is_a_503(cat, monkeypatch, caplog):
    _row(cat.db)
    monkeypatch.setattr(signature_verify, "KEYS_PATH", cat.tmp / "missing.json")
    with caplog.at_level(logging.ERROR, logger=catalog_api.__name__):
        resp = cat.client.get(CATALOG)
    assert resp.status_code == 503 and resp.json()["detail"] == catalog_api.CATALOG_DOWN
    assert resp.headers["cache-control"] == "no-store"
    assert any("signing keys" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_a_theme_json_storage_cannot_give_back_is_a_503(cat, damage):
    row = _row(cat.db)
    key = storage.theme_key(row.storage_prefix, "theme.json")
    if damage == "missing":
        del cat.s3.objects[key]
    else:
        cat.s3.objects[key] = (b'{"id": "automatos/other"}', "application/json")
    resp = cat.client.get(CATALOG)
    assert resp.status_code == 503 and resp.json()["detail"] == catalog_api.CATALOG_DOWN


def test_theme_json_is_read_once_per_version(cat):
    _row(cat.db)
    _row(cat.db, version="1.1.0")
    _catalog(cat)
    assert cat.s3.reads == 2
    _catalog(cat)
    assert cat.s3.reads == 2


def test_one_build_serves_every_caller_for_the_kept_seconds_and_a_failed_build_keeps_nothing(cat, monkeypatch):
    monkeypatch.setattr(catalog, "CATALOG_KEPT_SECONDS", 30)
    builds = []
    real_build = catalog.build_catalog

    def counted(db, now=None):
        builds.append(1)
        return real_build(db, now)

    monkeypatch.setattr(catalog, "build_catalog", counted)
    monkeypatch.setattr(signature_verify, "KEYS_PATH", cat.tmp / "missing.json")
    assert cat.client.get(CATALOG).status_code == 503  # a failed build keeps nothing ...
    monkeypatch.setattr(signature_verify, "KEYS_PATH", _keys(cat.tmp))
    _row(cat.db)
    first = _catalog(cat)  # ... so the next request builds
    _row(cat.db, version="1.1.0")
    second = cat.client.get(CATALOG)
    assert len(builds) == 2 and second.json() == first and len(first["items"]) == 1
    monkeypatch.setattr(catalog, "CATALOG_KEPT_SECONDS", 0)
    assert len(_catalog(cat)["items"]) == 2 and len(builds) == 3


def test_over_the_per_ip_limit_is_a_429_with_retry_after_in_the_route_s_own_bucket(cat, monkeypatch):
    monkeypatch.setattr(config, "MARKETPLACE_PUBLIC_RATE_PER_MINUTE", 2)
    headers = {"X-Forwarded-For": "198.51.100.7"}
    assert [cat.client.get(CATALOG, headers=headers).status_code for _ in range(2)] == [200, 200]
    refused = cat.client.get(CATALOG, headers=headers)
    assert refused.status_code == 429
    assert int(refused.headers["retry-after"]) > 0 and refused.headers["cache-control"] == "no-store"
    assert len(cat.redis.z) == 1 and next(iter(cat.redis.z)).endswith(":marketplace-catalog:ip:198.51.100.7")
    assert cat.client.get(CATALOG, headers={"X-Forwarded-For": "198.51.100.8"}).status_code == 200


def test_the_route_is_public_a_plain_def_in_the_manifest_and_mounted_by_router_spec():
    routes = [r for r in catalog_api.router.routes if isinstance(r, APIRoute) and r.path == CATALOG]
    assert len(routes) == 1 and routes[0].methods == {"GET"}
    route = routes[0]
    assert not inspect.iscoroutinefunction(route.endpoint)
    calls = _calls(route.dependant)
    assert get_request_context_hybrid not in calls
    assert calls[:2] == [catalog_api.require_hosted_edition, catalog_api.limit_catalog_reads] and get_db in calls
    assert not any("permission" in getattr(call, "__qualname__", "") for call in calls)
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    assert {"method": "GET", "path": CATALOG} in manifest["routes"]
    assert RouterSpec("api.marketplace_catalog") in MANIFEST_ROUTERS


# ---------------------------------------------------------------------------
# No caller waits on storage or another's build without a bound (P270W1-RVW-6)
# ---------------------------------------------------------------------------


class _Reader:
    """``catalog.read_stored_file`` wrapped: records ``session.in_transaction()`` on every read and, when
    ``hold`` is set, waits inside the read (``entered`` set) until the test sets ``released``."""

    def __init__(self, session, hold=False) -> None:
        self.session, self.hold, self.seen = session, hold, []
        self.entered, self.released = threading.Event(), threading.Event()

    def __call__(self, prefix, path):
        self.seen.append(self.session.in_transaction())
        if self.hold:
            self.entered.set()
            self.released.wait(HOLD_LIMIT_SECONDS)
        return serve.read_stored_file(prefix, path)


class _NoDatabase:
    """A session that fails any use: a caller answered without a build never reads the database."""

    def __getattr__(self, name):
        raise AssertionError(f"the database was read ({name})")


def _on_thread(call) -> SimpleNamespace:
    """``call()`` on a started daemon thread: its ``result``, or what it ``raised``, once ``thread`` is done."""
    outcome = SimpleNamespace(result=None, raised=None)

    def run():
        try:
            outcome.result = call()
        except Exception as error:  # noqa: BLE001 — the test reads what it raised
            outcome.raised = error

    outcome.thread = threading.Thread(target=run, daemon=True)
    outcome.thread.start()
    return outcome


def _second_caller_while_a_build_is_held(cat, reader) -> SimpleNamespace:
    """With one build held inside its theme.json read, a second ``catalog_snapshot`` on another thread, given
    ``BUILD_WAIT_SECONDS`` plus the margin to answer; the held build is let go afterwards and finishes."""
    builder = _on_thread(lambda: catalog.catalog_snapshot(cat.db))
    try:
        assert reader.entered.wait(HOLD_LIMIT_SECONDS)
        second = _on_thread(lambda: catalog.catalog_snapshot(_NoDatabase()))
        second.thread.join(BUILD_WAIT_SECONDS + WAIT_MARGIN_SECONDS)
        second.answered_in_time, second.reads = not second.thread.is_alive(), len(reader.seen)
    finally:
        reader.released.set()
        builder.thread.join(HOLD_LIMIT_SECONDS)
    assert builder.raised is None and not builder.thread.is_alive()
    second.builder = builder
    return second


def test_a_cold_build_reads_every_listed_theme_json_with_no_transaction_open(cat, monkeypatch):
    _row(cat.db)
    _row(cat.db, version="1.1.0")
    _row(cat.db, theme_id=INK)
    reader = _Reader(cat.db)
    monkeypatch.setattr(catalog, "read_stored_file", reader)
    items = _catalog(cat)["items"]
    assert [(item["id"], item["version"]) for item in items] == [(INK, "1.0.0"), (ARCADE, "1.1.0"), (ARCADE, "1.0.0")]
    assert reader.seen == [False, False, False]
    assert [item["publisher"] for item in items] == [  # read before the transaction ended
        {"name": "acme", "official": False}, {"name": "automatos", "official": True},
        {"name": "automatos", "official": True},
    ]


def test_a_caller_past_the_build_wait_with_no_snapshot_held_is_busy_and_reads_nothing(cat, monkeypatch):
    monkeypatch.setattr(catalog, "CATALOG_BUILD_WAIT_SECONDS", BUILD_WAIT_SECONDS)
    _row(cat.db)
    reader = _Reader(cat.db, hold=True)
    monkeypatch.setattr(catalog, "read_stored_file", reader)
    second = _second_caller_while_a_build_is_held(cat, reader)
    assert second.answered_in_time and isinstance(second.raised, catalog.CatalogBusy)
    assert second.reads == 1  # the held build's own read: the busy caller read nothing from storage
    assert len(second.builder.result.catalog["items"]) == 1


def test_a_caller_while_another_builds_gets_the_last_snapshot_held_however_old_without_waiting(cat, monkeypatch):
    monkeypatch.setattr(catalog, "CATALOG_BUILD_WAIT_SECONDS", HOLD_LIMIT_SECONDS)  # answered well inside it
    _row(cat.db)
    old = catalog.catalog_snapshot(cat.db)  # kept 0 s (the fixture): expired at once
    _row(cat.db, version="1.1.0")  # the next build reads the new version's theme.json
    reader = _Reader(cat.db, hold=True)
    monkeypatch.setattr(catalog, "read_stored_file", reader)
    second = _second_caller_while_a_build_is_held(cat, reader)
    assert second.answered_in_time and second.raised is None and second.reads == 1
    assert (second.result.body, second.result.etag) == (old.body, old.etag) and len(old.catalog["items"]) == 1
    built = second.builder.result
    assert len(built.catalog["items"]) == 2 and built.etag != old.etag


def test_a_busy_build_is_a_logged_503_with_retry_after_and_no_store(cat, monkeypatch, caplog):
    monkeypatch.setattr(catalog, "CATALOG_BUILD_WAIT_SECONDS", BUILD_WAIT_SECONDS)
    _row(cat.db)
    reader = _Reader(cat.db)
    monkeypatch.setattr(catalog, "read_stored_file", reader)
    assert catalog._building.acquire(timeout=HOLD_LIMIT_SECONDS)  # another caller's build, never finishing
    try:
        with caplog.at_level(logging.WARNING, logger=catalog_api.__name__):
            resp = cat.client.get(CATALOG)
    finally:
        catalog._building.release()
    assert resp.status_code == 503 and resp.json()["detail"] == catalog_api.CATALOG_BUSY
    assert resp.headers["retry-after"] == str(catalog.CATALOG_BUSY_RETRY_SECONDS)
    assert resp.headers["cache-control"] == "no-store" and "etag" not in resp.headers
    assert any("outlasted the wait" in record.getMessage() for record in caplog.records)
    assert reader.seen == [] and not cat.db.in_transaction()  # neither storage nor the database read for it
    assert len(_catalog(cat)["items"]) == 1  # the build free again: the next caller builds
