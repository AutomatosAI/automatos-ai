"""PRD-270 W2 US-270-12c (§5.2, §5.6, K11, K12) — item entries in the hosted catalogue, and the bundle route.

W1's catalogue route and this wave's bundle route run on a mini FastAPI app over Postgres: the rows are
flushed on one session and never committed (the teardown rolls them back), and the catalogue's themes are
kept to this test's (``catalog.all_listings`` filtered), so rows the suite seeded never matter. A theme
version is a real signed ``.autotheme`` in a fake S3 (W1's ``tests/helpers_theme_publish.py``); the per-IP
limit runs the real ``RateLimitStore`` over a fake Redis. Proven here:

* the catalogue lists W1's themes unchanged and first (what ``build_catalog`` alone lists), then one entry
  per exported item, agents, Playbooks, packages and plugins in that order, each entry's values its
  bundle's;
* every entry has exactly K11's keys (a walk of every key), ``trust: 'reviewed'`` and its bundle's path;
* the catalogue's ETag moves when a bundle changes, and the old revision is no longer served;
* a bundle over ``MARKETPLACE_CATALOG_BUNDLE_MAX_KB`` is left out, logged as a warning, and not served;
* a database failure on the bundle route is a 503, logged, with SERVING_HEADERS and no-store;
* the bundle route answers an entry's revision with bytes whose sha256 is it (the canonical bytes of the
  bundle), ``application/json``, the immutable cache, ETag ``"<revision>"`` and 304, SERVING_HEADERS; any
  other revision (unlisted, upper-case, short, long, not hex) is 404 with SERVING_HEADERS and no-store;
  the local edition is 404 before the limit counts; 429 with ``Retry-After`` in the route's own bucket;
* the bundle route answers from the catalogue's build: a fresh build is reused, an unlisted revision
  builds nothing of its own, and a process without a build builds once;
* W1's own reader (``feed.parse_catalog`` as on this branch) given this catalogue lists the same themes,
  refuses nothing and skips the entries (K11's older reader);
* the route is a plain ``def`` with no request-context or permission dependency, in the route manifest,
  mounted by ``RouterSpec``.

Without a database: ``with_items`` answers a new catalogue and leaves its argument alone, ``entry_of`` is
K11's entry, and ``bundle_bytes_for`` answers only what the build lists.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import re
import uuid
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

import api.marketplace_catalog as catalog_api
import api.marketplace_catalog_items as items_api
import api.public_rate_limit as rate_limit
from api.themes_files import SERVING_HEADERS
from api.widgets.rate_limit import RateLimitStore
from config import config
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.themes import ThemePackage
from router_manifest import MANIFEST_ROUTERS, RouterSpec
from services.marketplace_bundles import catalog_items
from services.marketplace_bundles.canonical import canonical_bytes, revision_of
from services.marketplace_bundles.export_agents import exported_agents
from services.marketplace_bundles.exported import ExportedItem, ItemEntry
from services.marketplace_bundles.lineage import external_id
from services.marketplace_cloud import catalog, feed
from services.themes import serve, signature_verify, storage
from services.themes.contract import KB
from services.themes.package import read_theme_package
from tests import helpers_theme_publish as hp
from tests.test_prd264_content_provenance import _FakeRedis
from tests.test_prd270w1_catalog_feed import _bytes
from tests.test_prd270w2_export_more import _agent, _llm, _package, _playbook, _plugin

CATALOG = "/api/marketplace/catalog"
BUNDLES = "/api/marketplace/catalog/bundles/"
ORCH = Path(__file__).resolve().parents[1]
FIXTURES = ORCH / "tests" / "fixtures" / "marketplace_bundles"
TABLES = (
    "users", "workspaces", "agents", "workflow_recipes", "skills", "agent_skills", "agent_tool_assignments",
    "marketplace_packages", "marketplace_plugins", "agent_assigned_plugins", "personas", "plugin_security_scans",
    "plugin_categories", "theme_packages", "theme_publishers", "theme_revocations",
)
TYPE_ORDER = ("agent", "playbook", "package", "plugin")
# Every key path of an entry (K11): a nested key as parent.child, a list's items as parent[].
ENTRY_PATHS = {
    "type", "externalId", "version", "revision", "name", "description", "category", "icon", "tags", "tags[]",
    "publisher", "publisher.name", "trust", "bundle",
}
HEX32, HEX64 = r"[0-9a-f]{32}", r"[0-9a-f]{64}"


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for table in TABLES:
                c.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the catalogue item tests need Postgres with the marketplace and theme tables: {exc}")
    yield eng
    eng.dispose()


def _forget():
    catalog.forget_cached()
    catalog_items.forget_cached()


@pytest.fixture
def hosted(engine, new_session, monkeypatch, tmp_path):
    """The hosted edition over one uncommitted session, its catalogue themes kept to this test's."""
    db, s3, redis, themes = new_session(), hp.FakeS3(), _FakeRedis(), set()
    every_listing = catalog.all_listings
    monkeypatch.setattr(catalog, "all_listings",
                        lambda session: {key: rows for key, rows in every_listing(session).items() if key in themes})
    monkeypatch.setattr(storage, "get_s3_client", lambda: s3)
    monkeypatch.setattr(storage, "ensure_bucket", lambda bucket: None)
    monkeypatch.setattr(serve, "get_s3_client", lambda: s3)
    monkeypatch.setattr(signature_verify, "KEYS_PATH", hp.keys_file(tmp_path))
    monkeypatch.setattr(rate_limit, "_rate_store", RateLimitStore(redis_factory=lambda: redis))
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", "on", raising=False)
    monkeypatch.setattr(config, "MARKETPLACE_PUBLIC_RATE_PER_MINUTE", 1000)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_FEED_MAX_KB", 1024)
    monkeypatch.setattr(catalog, "CATALOG_KEPT_SECONDS", 0)
    monkeypatch.setattr(catalog_items, "ITEMS_KEPT_SECONDS", 30)
    _forget()
    app = FastAPI()
    app.include_router(catalog_api.router)
    app.include_router(items_api.router)
    app.dependency_overrides[get_db] = lambda: db
    try:
        yield NS(db=db, tag=uuid.uuid4().hex[:10], themes=themes, redis=redis, client=TestClient(app))
    finally:
        _forget()
        new_session.sweep().close()


# --- rows ----------------------------------------------------------------------------------------


def _theme(hosted, version="1.0.0"):
    """A signed, approved, reviewed Marketplace theme version: its files in the fake S3, its row flushed."""
    theme_id = f"prd270w2/t{hosted.tag}"
    data = _bytes(theme_id, version)
    checked = read_theme_package(data)
    prefix = storage.theme_prefix(theme_id, version)
    storage.write_theme_files(prefix, checked.files)
    row = ThemePackage(
        theme_id=theme_id, version=version, owner_type="marketplace", workspace_id=None, digest=checked.digest,
        manifest=checked.manifest, file_manifest=checked.file_manifest, storage_prefix=prefix, status="approved",
        reviewed=True, signature=json.loads(hp.envelope(data, hp.SIGNER))["signature"], scan_verdict="pass",
    )
    hosted.db.add(row)
    hosted.db.flush()
    hosted.themes.add(theme_id)
    return row


def _items(hosted):
    """One exported item of each type; their externalIds by type."""
    agent, playbook, plugin = _agent(hosted), _playbook(hosted), _plugin(hosted)
    package = _package(hosted, [_llm()])
    return NS(agent=agent, ids={
        "agent": external_id("agent", f"row:{agent.public_id}"),
        "playbook": external_id("playbook", f"row:{playbook.template_id}"),
        "package": external_id("package", f"row:{package.slug}"),
        "plugin": external_id("plugin", f"row:{plugin.slug}"),
    })


# --- reads ---------------------------------------------------------------------------------------


def _get_catalog(hosted, **headers):
    resp = hosted.client.get(CATALOG, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp


def _entries(body):
    """The catalogue's item entries by (type, externalId)."""
    return {(item["type"], item["externalId"]): item for item in body["items"] if item["type"] != "theme"}


def _exported(hosted):
    """Every item the export answers whose bundle fits the catalogue's cap, by (type, externalId)."""
    cap = catalog_items.bundle_max_bytes()
    return {
        (item.entry.type, item.entry.external_id): item for item in catalog_items.item_set(hosted.db)
        if len(canonical_bytes(item.bundle)) <= cap
    }


def _paths(value, prefix=""):
    """Every key path in ``value``: a nested key as parent.child, a list's items as parent[]."""
    if isinstance(value, dict):
        return {path for key, child in value.items() for path in _with_children(_join(prefix, key), child)}
    if isinstance(value, list):
        return {path for child in value for path in _with_children(f"{prefix}[]", child)}
    return set()


def _with_children(path, value):
    return {path} | _paths(value, path)


def _join(prefix, key):
    return f"{prefix}.{key}" if prefix else key


def _assert_serving_headers(resp):
    for name, value in SERVING_HEADERS.items():
        assert resp.headers[name.lower()] == value, name


def _calls(dependant):
    found = []
    for sub in dependant.dependencies:
        found += [sub.call] + _calls(sub)
    return found


def _fixture_item(name="agent"):
    bundle = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    entry = ItemEntry(bundle["type"], bundle["externalId"], bundle["version"], frozenset())
    return ExportedItem(entry, bundle, revision_of(bundle))


# --- without a database ------------------------------------------------------------------------


def test_with_items_answers_a_new_catalogue_and_leaves_its_argument_alone(monkeypatch):
    item = _fixture_item()
    monkeypatch.setattr(catalog_items, "item_set", lambda db: [item])
    theme = {"type": "theme", "id": "acme/ink", "version": "1.0.0"}
    built = {"schema": 1, "generatedAt": "2026-10-31T09:00:00+00:00", "items": [theme]}
    before = copy.deepcopy(built)
    try:
        answered = catalog_items.with_items(built, None)
        assert built == before and answered is not built and answered["items"] is not built["items"]
        assert answered["schema"] == 1 and answered["generatedAt"] == built["generatedAt"]
        assert answered["items"] == [theme, catalog_items.entry_of(item)]
        data = catalog_items.bundle_bytes_for(None, item.revision)
        assert data == canonical_bytes(item.bundle) and hashlib.sha256(data).hexdigest() == item.revision
        assert catalog_items.bundle_bytes_for(None, "0" * 64) is None
    finally:
        catalog_items.forget_cached()


def test_an_entry_is_k11_s_metadata_never_the_bundle_body():
    item = _fixture_item()
    entry = catalog_items.entry_of(item)
    bundle = item.bundle
    assert _paths(entry) == ENTRY_PATHS
    assert entry == {
        "type": "agent", "externalId": bundle["externalId"], "version": bundle["version"], "revision": item.revision,
        "name": bundle["name"], "description": bundle["description"], "category": bundle["category"],
        "icon": bundle["icon"], "tags": bundle["tags"], "publisher": {"name": bundle["publisher"]["name"]},
        "trust": "reviewed", "bundle": f"{BUNDLES}{item.revision}",
    }
    assert entry["tags"] is not bundle["tags"]
    assert "agent" not in entry and "official" not in json.dumps(entry).lower()


# --- the catalogue -------------------------------------------------------------------------------


def test_the_catalogue_lists_w1_s_themes_unchanged_then_one_entry_per_exported_item(hosted):
    theme, mine = _theme(hosted), _items(hosted)
    body = _get_catalog(hosted).json()
    themes = [item for item in body["items"] if item["type"] == "theme"]
    assert themes == catalog.build_catalog(hosted.db)["items"] and [item["digest"] for item in themes] == [theme.digest]
    entries, rest = _entries(body), body["items"][len(themes):]
    assert body["items"][:len(themes)] == themes and len(rest) == len(entries)
    exported = _exported(hosted)
    assert set(entries) == set(exported)
    assert [item["type"] for item in rest] == sorted((item["type"] for item in rest), key=TYPE_ORDER.index)
    for item_type, item_id in mine.ids.items():
        entry, bundle = entries[(item_type, item_id)], exported[(item_type, item_id)].bundle
        assert entry["revision"] == exported[(item_type, item_id)].revision
        assert (entry["version"], entry["name"], entry["description"]) == (bundle["version"], bundle["name"],
                                                                            bundle["description"])
        assert (entry["category"], entry["icon"], entry["tags"], entry["publisher"]) == (
            bundle["category"], bundle["icon"], bundle["tags"], bundle["publisher"])


def test_every_entry_has_exactly_k11_s_keys_trust_reviewed_and_its_bundle_path(hosted):
    _items(hosted)
    entries = _entries(_get_catalog(hosted).json())
    assert len(entries) >= len(TYPE_ORDER)
    for entry in entries.values():
        assert _paths(entry) - {"tags[]"} == ENTRY_PATHS - {"tags[]"}, entry["externalId"]
        assert entry["type"] in TYPE_ORDER and entry["trust"] == "reviewed"
        assert entry["bundle"] == f"{BUNDLES}{entry['revision']}"
        assert re.fullmatch(HEX32, entry["externalId"]) and re.fullmatch(HEX64, entry["revision"])
        assert all(isinstance(tag, str) for tag in entry["tags"])


def test_the_catalogue_etag_moves_when_a_bundle_changes_and_the_old_revision_is_not_served(hosted):
    mine = _items(hosted)
    key = ("agent", mine.ids["agent"])
    first = _get_catalog(hosted)
    etag, old = first.headers["etag"], _entries(first.json())[key]["revision"]
    assert hosted.client.get(CATALOG, headers={"If-None-Match": etag}).status_code == 304
    mine.agent.description = "A changed description."
    hosted.db.flush()
    second = hosted.client.get(CATALOG, headers={"If-None-Match": etag})
    assert second.status_code == 200 and second.headers["etag"] != etag
    new = _entries(second.json())[key]["revision"]
    assert new != old
    assert hosted.client.get(BUNDLES + old).status_code == 404
    assert hosted.client.get(BUNDLES + new).status_code == 200


def test_a_bundle_over_the_cap_is_left_out_logged_and_not_served(hosted, monkeypatch, caplog):
    small = _agent(hosted)
    big = _agent(hosted, use_custom_persona=True, custom_persona_prompt="x" * 4000)
    small_id, big_id = (external_id("agent", f"row:{agent.public_id}") for agent in (small, big))
    agents = {item.entry.external_id: item for item in exported_agents(hosted.db)}
    small_size, big_size = (len(canonical_bytes(agents[item_id].bundle)) for item_id in (small_id, big_id))
    cap_kb = -(-small_size // KB)  # the smallest cap the small bundle fits under
    assert big_size > cap_kb * KB
    monkeypatch.setattr(config, "MARKETPLACE_CATALOG_BUNDLE_MAX_KB", cap_kb)
    with caplog.at_level(logging.WARNING, logger=catalog_items.__name__):
        entries = _entries(_get_catalog(hosted).json())
    assert ("agent", small_id) in entries and ("agent", big_id) not in entries
    assert any(big_id in record.getMessage() and record.levelno == logging.WARNING for record in caplog.records)
    assert hosted.client.get(BUNDLES + agents[big_id].revision).status_code == 404


# --- the bundle route ----------------------------------------------------------------------------


def test_the_bundle_route_answers_an_entry_s_revision_with_its_canonical_bytes(hosted):
    mine = _items(hosted)
    entries, exported = _entries(_get_catalog(hosted).json()), _exported(hosted)
    for key in mine.ids.items():
        revision = entries[key]["revision"]
        resp = hosted.client.get(entries[key]["bundle"])
        assert resp.status_code == 200, resp.text
        assert hashlib.sha256(resp.content).hexdigest() == revision
        assert resp.content == canonical_bytes(exported[key].bundle)
        assert json.loads(resp.content) == exported[key].bundle
        assert resp.headers["content-type"] == "application/json"
        assert resp.headers["cache-control"] == f"public, max-age={config.THEME_FILE_CACHE_SECONDS}, immutable"
        assert resp.headers["etag"] == f'"{revision}"'
        _assert_serving_headers(resp)
        again = hosted.client.get(entries[key]["bundle"], headers={"If-None-Match": f'"{revision}"'})
        assert again.status_code == 304 and again.content == b"" and again.headers["etag"] == f'"{revision}"'
        _assert_serving_headers(again)


@pytest.mark.parametrize("change", ["unlisted", "upper", "short", "long", "not-hex"])
def test_any_other_revision_is_a_404_with_the_serving_headers(hosted, change):
    mine = _items(hosted)
    listed = _entries(_get_catalog(hosted).json())[("agent", mine.ids["agent"])]["revision"]
    revision = {
        "unlisted": hashlib.sha256(b"never listed").hexdigest(), "upper": listed.upper(), "short": listed[:-1],
        "long": listed + "0", "not-hex": "z" + listed[1:],
    }[change]
    resp = hosted.client.get(BUNDLES + revision)
    assert resp.status_code == 404 and resp.json()["detail"] == items_api.BUNDLE_NOT_LISTED
    assert resp.headers["cache-control"] == "no-store" and "etag" not in resp.headers
    _assert_serving_headers(resp)


def test_a_database_failure_is_a_logged_503_with_the_serving_headers(hosted, monkeypatch, caplog):
    def down(db, revision):
        raise OperationalError("SELECT 1", {}, ConnectionError("the database is down"))

    monkeypatch.setattr(items_api, "bundle_bytes_for", down)
    with caplog.at_level(logging.ERROR, logger=items_api.__name__):
        resp = hosted.client.get(BUNDLES + "0" * 64)
    assert resp.status_code == 503 and resp.json()["detail"] == items_api.BUNDLES_DOWN
    assert resp.headers["cache-control"] == "no-store" and "etag" not in resp.headers
    _assert_serving_headers(resp)
    assert any(record.exc_info for record in caplog.records)


def test_the_local_edition_answers_404_before_the_limit_counts(hosted, monkeypatch):
    mine = _items(hosted)
    revision = _entries(_get_catalog(hosted).json())[("agent", mine.ids["agent"])]["revision"]
    hosted.redis.z.clear()
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    resp = hosted.client.get(BUNDLES + revision)
    assert resp.status_code == 404 and "hosted edition" in resp.json()["detail"]
    assert resp.headers["cache-control"] == "no-store"
    _assert_serving_headers(resp)
    assert hosted.redis.z == {}


def test_over_the_per_ip_limit_is_a_429_with_retry_after_in_the_route_s_own_bucket(hosted, monkeypatch):
    monkeypatch.setattr(config, "MARKETPLACE_PUBLIC_RATE_PER_MINUTE", 2)
    headers, revision = {"X-Forwarded-For": "198.51.100.7"}, "0" * 64
    assert [hosted.client.get(BUNDLES + revision, headers=headers).status_code for _ in range(2)] == [404, 404]
    refused = hosted.client.get(BUNDLES + revision, headers=headers)
    assert refused.status_code == 429
    assert int(refused.headers["retry-after"]) > 0 and refused.headers["cache-control"] == "no-store"
    _assert_serving_headers(refused)
    assert len(hosted.redis.z) == 1
    assert next(iter(hosted.redis.z)).endswith(":marketplace-catalog-bundles:ip:198.51.100.7")
    assert hosted.client.get(BUNDLES + revision, headers={"X-Forwarded-For": "198.51.100.8"}).status_code == 404


def test_the_bundle_route_answers_from_the_catalogue_s_build(hosted, monkeypatch):
    mine = _items(hosted)
    builds, every_item = [], catalog_items.item_set

    def counted(db):
        builds.append(1)
        return every_item(db)

    monkeypatch.setattr(catalog_items, "item_set", counted)
    revision = _entries(_get_catalog(hosted).json())[("agent", mine.ids["agent"])]["revision"]
    assert len(builds) == 1
    assert hosted.client.get(BUNDLES + revision).status_code == 200
    assert hosted.client.get(BUNDLES + "0" * 64).status_code == 404
    assert len(builds) == 1  # the catalogue's build answers both
    catalog_items.forget_cached()  # a worker that has not built yet ...
    assert hosted.client.get(BUNDLES + "0" * 64).status_code == 404
    assert hosted.client.get(BUNDLES + revision).status_code == 200
    assert len(builds) == 2  # ... builds once, for every request after it


def test_w1_s_own_reader_lists_the_same_themes_and_fails_nothing(hosted):
    theme, _mine = _theme(hosted), _items(hosted)
    resp = _get_catalog(hosted)
    entries = _entries(resp.json())
    parsed = feed.parse_catalog(resp.content, hosted.db)
    assert [(item.theme_id, item.version, item.digest) for item in parsed.items] == [
        (theme.theme_id, theme.version, theme.digest)]
    assert parsed.refused == () and parsed.skipped == len(entries) >= len(TYPE_ORDER)


def test_the_route_is_public_a_plain_def_in_the_manifest_and_mounted_by_router_spec():
    path = BUNDLES + "{revision}"
    routes = [r for r in items_api.router.routes if isinstance(r, APIRoute) and r.path == path]
    assert len(routes) == 1 and routes[0].methods == {"GET"}
    route = routes[0]
    assert not inspect.iscoroutinefunction(route.endpoint)
    calls = _calls(route.dependant)
    assert get_request_context_hybrid not in calls
    assert calls[:2] == [items_api.require_hosted_bundles, items_api.limit_bundle_reads] and get_db in calls
    assert not any("permission" in getattr(call, "__qualname__", "") for call in calls)
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    assert {"method": "GET", "path": path} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])
    assert RouterSpec("api.marketplace_catalog_items") in MANIFEST_ROUTERS
