"""PRD-270 W2 P270W2-RVW-3 (§5.2, §5.6, K11) — the hosted catalogue stays within what every reader accepts.

W1's route refuses the whole catalogue, themes included, over ``MARKETPLACE_CATALOG_FEED_MAX_KB``, and every
local reader (``feed.parse_catalog``) refuses it over ``feed.CATALOG_MAX_ITEMS`` items or
``AUTOMATOS_CLOUD_FEED_MAX_KB``. Before this story the item entries were listed with no bound, so a large
community Marketplace failed every Sync outright. W1's catalogue route and the bundle route run on a mini
FastAPI app over in-memory SQLite holding the theme tables (W1's ``tests/test_prd270w1_catalog_feed.py``
fixture and rows: one real signed theme in a fake S3), with ``catalog_items.item_set`` answering more items
than fit. Proven here, once over the item count and once over the byte cap (both caps at their defaults):

* the catalogue is a 200, its theme first, then the longest run of the items in ``item_set``'s order that
  fits (the next one would not), within both caps;
* each item left out is logged as a warning with its type and externalId;
* ``feed.parse_catalog`` of that body refuses nothing and reads every listed entry;
* a left-out item's bundle is 404 and the last listed one's is served.

Without a route: ``room_beside`` measures the bytes the route sends, a catalogue whose themes leave no room
lists no entry, and a bundle-route rebuild fits the room of the build it replaces (a left-out bundle stays
404 after the listing ages), or with none held the widest room.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.marketplace_catalog as catalog_api
import api.marketplace_catalog_items as items_api
import api.public_rate_limit as rate_limit
from api.widgets.rate_limit import RateLimitStore
from config import config
from core.database.database import get_db
from core.models.themes import ThemePackage
from services.marketplace_bundles import catalog_items
from services.marketplace_bundles.canonical import revision_of
from services.marketplace_bundles.exported import ExportedItem, ItemEntry
from services.marketplace_bundles.schema import DESCRIPTION_MAX, TAG_MAX, TAGS_MAX
from services.marketplace_cloud import catalog, feed
from services.themes import serve, signature_verify, storage
from services.themes.contract import KB
from services.themes.revocation_feed import feed_bytes
from tests import helpers_theme_publish as hp
from tests.test_prd264_content_provenance import _FakeRedis
from tests.test_prd270w1_catalog_feed import TABLES, _keys, _row

CATALOG = "/api/marketplace/catalog"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "marketplace_bundles"
DEFAULT_FEED_KB = 1024
# More long entries than 1024 KB holds (each is about 5.5 KB), and fewer than the item count would stop.
LONG_ITEMS = 220
GENERATED_AT = "2026-10-31T09:00:00+00:00"
LEFT_OUT = "is left out"


@pytest.fixture
def bound(monkeypatch, tmp_path):
    """The hosted edition over SQLite with the theme tables, both caps at their defaults, nothing kept."""
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=TABLES)
    session, s3, redis = sessionmaker(bind=engine)(), hp.FakeS3(), _FakeRedis()
    monkeypatch.setattr(storage, "get_s3_client", lambda: s3)
    monkeypatch.setattr(storage, "ensure_bucket", lambda bucket: None)
    monkeypatch.setattr(serve, "get_s3_client", lambda: s3)
    monkeypatch.setattr(signature_verify, "KEYS_PATH", _keys(tmp_path))
    monkeypatch.setattr(rate_limit, "_rate_store", RateLimitStore(redis_factory=lambda: redis))
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", "on", raising=False)
    monkeypatch.setattr(config, "MARKETPLACE_PUBLIC_RATE_PER_MINUTE", 100000)
    monkeypatch.setattr(config, "MARKETPLACE_CATALOG_FEED_MAX_KB", DEFAULT_FEED_KB)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_FEED_MAX_KB", DEFAULT_FEED_KB)
    monkeypatch.setattr(catalog, "CATALOG_KEPT_SECONDS", 0)
    monkeypatch.setattr(catalog_items, "ITEMS_KEPT_SECONDS", 30)
    _forget()
    app = FastAPI()
    app.include_router(catalog_api.router)
    app.include_router(items_api.router)
    app.dependency_overrides[get_db] = lambda: session
    try:
        yield SimpleNamespace(db=session, client=TestClient(app), monkeypatch=monkeypatch)
    finally:
        _forget()
        session.close()
        engine.dispose()


def _forget():
    catalog.forget_cached()
    catalog_items.forget_cached()


def _items(count, long=False):
    """``count`` agents made from the fixture bundle, each its own externalId; ``long`` gives each the longest
    description and the most, longest tags the format allows."""
    base = json.loads((FIXTURES / "agent.json").read_text(encoding="utf-8"))
    return [_item(base, number, long) for number in range(count)]


def _item(base, number, long):
    external_id = f"{number:032x}"
    bundle = {**base, "externalId": external_id, "name": f"Test agent {number}"}
    if long:
        tags = [f"{tag:02d}" + "t" * (TAG_MAX - 2) for tag in range(TAGS_MAX)]
        bundle = {**bundle, "description": "d" * DESCRIPTION_MAX, "tags": tags}
    return ExportedItem(ItemEntry("agent", external_id, bundle["version"], frozenset()), bundle, revision_of(bundle))


def _serve(bound, items):
    bound.monkeypatch.setattr(catalog_items, "item_set", lambda db: items)


def _left_out_ids(records):
    """The externalIds the catalogue's warnings name as left out."""
    return {
        record.args[1] for record in records
        if record.name == catalog_items.__name__ and record.levelno == logging.WARNING
        and LEFT_OUT in record.getMessage()
    }


def _following_bytes(item):
    """What listing ``item`` next would add to the catalogue's body."""
    return len(feed_bytes(catalog_items.entry_of(item))) + catalog_items.ENTRY_SEPARATOR_BYTES


def _assert_bounded(bound, items, caplog):
    """The catalogue is a 200 within both caps: its theme, then the longest run of ``items`` that fits, the
    rest logged; the reader refuses nothing; a left-out bundle is 404 and the last listed one served."""
    digest = _row(bound.db).digest
    with caplog.at_level(logging.WARNING, logger=catalog_items.__name__):
        resp = bound.client.get(CATALOG)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["items"][0]["type"] == "theme" and body["items"][0]["digest"] == digest
    entries = body["items"][1:]
    fits = len(entries)
    assert 0 < fits < len(items)
    assert entries == [catalog_items.entry_of(item) for item in items[:fits]]
    assert len(body["items"]) <= feed.CATALOG_MAX_ITEMS and len(resp.content) <= DEFAULT_FEED_KB * KB
    assert _left_out_ids(caplog.records) == {item.entry.external_id for item in items[fits:]}
    parsed = feed.parse_catalog(resp.content, bound.db)
    assert [item.digest for item in parsed.items] == [digest]
    assert parsed.refused == () and parsed.skipped == 0 and parsed.entries.refused == ()
    assert [entry.external_id for entry in parsed.entries.listed] == [item.entry.external_id for item in items[:fits]]
    assert bound.client.get(entries[-1]["bundle"]).status_code == 200
    assert bound.client.get(catalog_items.entry_of(items[fits])["bundle"]).status_code == 404
    return resp, fits


# --- through the route ----------------------------------------------------------------------------


def test_over_the_item_count_the_catalogue_lists_its_theme_then_what_fits_and_logs_the_rest(bound, caplog):
    items = _items(feed.CATALOG_MAX_ITEMS + 5)
    _serve(bound, items)
    resp, fits = _assert_bounded(bound, items, caplog)
    assert fits == feed.CATALOG_MAX_ITEMS - 1  # the count stopped it, not the bytes:
    assert len(resp.content) + _following_bytes(items[fits]) <= DEFAULT_FEED_KB * KB


def test_over_the_byte_cap_the_catalogue_lists_its_theme_then_what_fits_and_logs_the_rest(bound, caplog):
    items = _items(LONG_ITEMS, long=True)
    _serve(bound, items)
    resp, fits = _assert_bounded(bound, items, caplog)
    assert fits < LONG_ITEMS < feed.CATALOG_MAX_ITEMS  # the bytes stopped it, not the count
    assert len(resp.content) + _following_bytes(items[fits]) > DEFAULT_FEED_KB * KB  # the next would not fit


# --- without a route ------------------------------------------------------------------------------


def _themes(count):
    items = [{"type": "theme", "id": f"acme/t{number}", "version": "1.0.0"} for number in range(count)]
    return {"schema": 1, "generatedAt": GENERATED_AT, "items": items}


def test_room_beside_measures_the_bytes_the_route_sends(monkeypatch):
    monkeypatch.setattr(config, "MARKETPLACE_CATALOG_FEED_MAX_KB", 4)
    themes = _themes(3)
    room = catalog_items.room_beside(themes)
    assert room == catalog_items.Room(feed.CATALOG_MAX_ITEMS - 3, 4 * KB - len(feed_bytes(themes)))
    items = _items(40)
    monkeypatch.setattr(catalog_items, "item_set", lambda db: items)
    try:
        answered = catalog_items.with_items(themes, None)
    finally:
        catalog_items.forget_cached()
    assert answered["items"][:3] == themes["items"] and 0 < len(answered["items"]) - 3 < len(items)
    assert len(feed_bytes(answered)) <= 4 * KB


def test_themes_that_leave_no_room_list_no_entry(monkeypatch, caplog):
    monkeypatch.setattr(config, "MARKETPLACE_CATALOG_FEED_MAX_KB", DEFAULT_FEED_KB)
    items = _items(3)
    monkeypatch.setattr(catalog_items, "item_set", lambda db: items)
    themes = _themes(feed.CATALOG_MAX_ITEMS)
    try:
        with caplog.at_level(logging.WARNING, logger=catalog_items.__name__):
            answered = catalog_items.with_items(themes, None)
        assert answered["items"] == themes["items"]
        assert _left_out_ids(caplog.records) == {item.entry.external_id for item in items}
        assert all(catalog_items.bundle_bytes_for(None, item.revision) is None for item in items)
    finally:
        catalog_items.forget_cached()


def test_a_bundle_route_rebuild_fits_the_held_room_or_with_none_the_widest(monkeypatch):
    monkeypatch.setattr(config, "MARKETPLACE_CATALOG_FEED_MAX_KB", DEFAULT_FEED_KB)
    items = _items(5)
    monkeypatch.setattr(catalog_items, "item_set", lambda db: items)
    themes = _themes(feed.CATALOG_MAX_ITEMS - 2)  # room for two entries
    try:
        assert len(catalog_items.with_items(themes, None)["items"]) == feed.CATALOG_MAX_ITEMS
        monkeypatch.setattr(catalog_items, "ITEMS_KEPT_SECONDS", 0)  # the held build has aged: the route rebuilds
        assert catalog_items.bundle_bytes_for(None, items[1].revision) is not None
        assert catalog_items.bundle_bytes_for(None, items[2].revision) is None
        catalog_items.forget_cached()  # a worker that holds no build fits the widest room
        assert all(catalog_items.bundle_bytes_for(None, item.revision) is not None for item in items)
    finally:
        catalog_items.forget_cached()
