"""PRD-270 W2 P270W2-RVW-5 — on Postgres, the hosted catalogue holds no transaction over a theme.json read, and
reads its item entries from the database after the release.

W1's ``build_catalog`` does every database read first, detaches the listed versions and ends its read
transaction (``catalog._end_reads``, ``release_if_read_only``), then reads each theme.json from storage; W2's
``with_items`` then reads the exports in a new transaction. ``test_prd270w2_catalog_items`` seeds on a session it
never commits, so the release finds a transaction that wrote and keeps it, and W1's own release test
(``test_prd270w1_catalog_feed``, SQLite) stubs ``item_set`` out: neither runs the path production runs.

Here the seed (a signed theme version and an exported agent, Playbook, package and plugin) is COMMITTED on its
own session and deleted on teardown (``new_session`` and its ``sweep``), and the route reads on a fresh session,
as ``get_db`` gives every request one: its transaction begins with the request and only reads.

Not the SAVEPOINT-join pattern (``test_prd266_mobile_devices_routes``): a session joined to an outer transaction
begins each of its transactions with a ``SAVEPOINT`` statement, which ``core/database/read_release.py`` counts as
a write on the connection, so the release never runs there and a held transaction can't be told from a released
one.
"""
from __future__ import annotations

import logging
import uuid
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

import api.marketplace_catalog as catalog_api
import api.public_rate_limit as rate_limit
from api.widgets.rate_limit import RateLimitStore
from config import config
from core.database.database import get_db
from core.models.core import Agent, WorkflowTemplate as WorkflowRecipe
from core.models.marketplace_packages import MarketplacePackage
from core.models.marketplace_plugins import MarketplacePlugin, PluginSecurityScan
from core.models.themes import ThemePackage
from services.marketplace_bundles import catalog_items
from services.marketplace_bundles.lineage import external_id
from services.marketplace_cloud import catalog
from services.themes import serve, signature_verify, storage
from tests import helpers_theme_publish as hp
from tests.test_prd264_content_provenance import _FakeRedis
from tests.test_prd270w2_catalog_items import CATALOG, TABLES, _theme
from tests.test_prd270w2_export_more import _agent, _llm, _package, _playbook, _plugin

ITEMS, ITEMS_READ = "item_set called", "item_set answered"


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for table in TABLES:
                c.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the catalogue release test needs Postgres with the marketplace and theme tables: {exc}")
    yield eng
    eng.dispose()


def _forget():
    """Drop the kept catalogue, theme.json texts and item build, so the request builds from the database."""
    catalog.forget_cached()
    catalog_items.forget_cached()


def _seed(state):
    """A signed theme version and one exported item of each type, committed; the rows by type."""
    rows = NS(theme=_theme(state), agent=_agent(state), playbook=_playbook(state), plugin=_plugin(state))
    rows.package = _package(state, [_llm()])
    state.db.commit()
    return rows


def _delete(sweep, rows):
    """Delete the committed seed in one commit (``rows`` None: nothing was committed); the plugin before its scan."""
    try:
        if rows is not None:
            for model, row_id in (
                (MarketplacePackage, rows.package.id), (MarketplacePlugin, rows.plugin.id),
                (PluginSecurityScan, rows.plugin.security_scan_id), (WorkflowRecipe, rows.playbook.id),
                (Agent, rows.agent.id), (ThemePackage, rows.theme.id),
            ):
                sweep.query(model).filter(model.id == row_id).delete(synchronize_session=False)
        sweep.commit()
    finally:
        sweep.close()


@pytest.fixture
def hosted(engine, new_session, monkeypatch, tmp_path):
    """The hosted edition over a committed seed, its catalogue themes kept to this test's; the route's session
    is a fresh one (``route``), never the seed's."""
    s3, redis, themes = hp.FakeS3(), _FakeRedis(), set()
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
    state = NS(db=new_session(), tag=uuid.uuid4().hex[:10], themes=themes, rows=None)
    _forget()
    try:
        state.rows = _seed(state)
        state.route = new_session()
        app = FastAPI()
        app.include_router(catalog_api.router)
        app.dependency_overrides[get_db] = lambda: state.route
        state.client = TestClient(app)
        yield state
    finally:
        _forget()
        _delete(new_session.sweep(), state.rows)


def _recorded(hosted, monkeypatch):
    """Every theme.json read and the item export, each with whether the route's session held a transaction."""
    events, every_item = [], catalog_items.item_set

    def reader(prefix, path):
        events.append((path, hosted.route.in_transaction()))
        return serve.read_stored_file(prefix, path)

    def items(db):
        events.append((ITEMS, db.in_transaction()))
        found = every_item(db)
        events.append((ITEMS_READ, db.in_transaction()))
        return found

    monkeypatch.setattr(catalog, "read_stored_file", reader)
    monkeypatch.setattr(catalog_items, "item_set", items)
    return events


def test_no_transaction_is_held_over_a_theme_json_read_and_the_items_are_read_after_the_release(
        hosted, monkeypatch, caplog):
    events, rows = _recorded(hosted, monkeypatch), hosted.rows
    with caplog.at_level(logging.WARNING, logger=catalog.__name__):
        resp = hosted.client.get(CATALOG)
    assert resp.status_code == 200, resp.text
    # The build's reads released before storage; the exports then read in a transaction of their own.
    assert events == [(catalog.THEME_JSON, False), (ITEMS, False), (ITEMS_READ, True)]
    assert not [record for record in caplog.records if record.name == catalog.__name__]  # nothing "kept"
    body = resp.json()
    themes = [item for item in body["items"] if item["type"] == "theme"]
    assert [(item["id"], item["version"], item["digest"]) for item in themes] == [
        (rows.theme.theme_id, rows.theme.version, rows.theme.digest)]
    entries = {(item["type"], item["externalId"]): item for item in body["items"] if item["type"] != "theme"}
    for key in (
        ("agent", external_id("agent", f"row:{rows.agent.public_id}")),
        ("playbook", external_id("playbook", f"row:{rows.playbook.template_id}")),
        ("package", external_id("package", f"row:{rows.package.slug}")),
        ("plugin", external_id("plugin", f"row:{rows.plugin.slug}")),
    ):
        assert entries[key]["trust"] == "reviewed", key
