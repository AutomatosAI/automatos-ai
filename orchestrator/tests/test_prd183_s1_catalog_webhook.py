"""PRD-183 S1 (F032) — catalog webhooks actually update the commerce graph.

The old ``/events`` handler scheduled a Shopify-shaped pending dict
(``{"source": "shopify", "event": ..., "shop": ...}``) via
``GraphifyService.schedule_incremental_update``. That path funnels every
pending through ``partition_pending_sources``, which only recognises
``document`` / ``mission_synthesis`` / ``generated_document`` / ``report``
types — a dict with no ``type`` key falls through untouched and
``_incremental_build`` returns early ("no extractable sources"). So every
catalog webhook was a silent no-op: the commerce graph never changed.

The commerce graph is built from the Shopify *catalog* (products, variants,
collections, vendors) by ``map_shopify_catalog`` → ``import_graph`` inside
``_product_sync_impl`` — not by re-extracting documents. These tests pin the
fixed contract: a catalog-shaped ``/events`` payload triggers a real catalog
re-sync for the resolved workspace, and a non-catalog event does not.

Pure: the Composio/httpx/graph internals of ``_product_sync_impl`` are mocked
at the boundary; no DB, no network. The webhook body shape the Shopify Remix
app (Part B) must POST is asserted directly against ``EventRequest``.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio  # noqa: E402

import tests.conftest as _conftest  # noqa: E402
_conftest._restore_real_app_modules()

from api import shopify  # noqa: E402


class _FakeQuery:
    """Minimal query stub — returns a single fake workspace for the shop."""

    def __init__(self, workspace):
        self._workspace = workspace

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._workspace


class _FakeWorkspace:
    def __init__(self, wid="ws-123"):
        self.id = wid
        self.is_active = True
        self.settings = {"shopify_domain": "inbuilduk.myshopify.com"}


class _FakeDb:
    def __init__(self, workspace):
        self._workspace = workspace

    def query(self, *args, **kwargs):
        return _FakeQuery(self._workspace)


def _run_and_drain(coro):
    """Run *coro* to completion, then drain any background tasks it spawned.

    ``forward_event`` dispatches the catalog re-sync via ``create_task`` so the
    webhook returns immediately; a deterministic test must then let those
    scheduled tasks run before asserting. We run everything on one fresh loop.
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro)
        # Drain remaining scheduled tasks (the detached re-sync).
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        return result
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_events_body_shape_is_the_remix_contract():
    """The /events contract Part B posts to: shop + event + optional data."""
    req = shopify.EventRequest(
        shop="inbuilduk.myshopify.com",
        event="products/update",
        data={"id": 123},
    )
    assert req.shop == "inbuilduk.myshopify.com"
    assert req.event == "products/update"
    # `data` is optional — a bare {shop, event} must still validate.
    bare = shopify.EventRequest(shop="x.myshopify.com", event="products/create")
    assert bare.data is None


def test_catalog_webhook_updates_graph(monkeypatch):
    """A products/update event fires a real catalog re-sync for the workspace.

    Asserts the fixed behaviour: the handler resolves the workspace by shop
    domain and invokes ``_sync_catalog_for_workspace`` (which runs the
    ``map_shopify_catalog`` → ``import_graph`` pipeline) — NOT the document-only
    ``schedule_incremental_update`` that dropped the pending.
    """
    ws = _FakeWorkspace()
    db = _FakeDb(ws)

    calls = {}

    async def _fake_sync(workspace_id, event):
        calls["workspace_id"] = workspace_id
        calls["event"] = event

    # The graph re-sync is dispatched through this seam; capture it instead of
    # touching Composio/httpx/Qdrant.
    monkeypatch.setattr(shopify, "_sync_catalog_for_workspace", _fake_sync)

    req = shopify.EventRequest(shop="inbuilduk.myshopify.com", event="products/update")
    result = _run_and_drain(shopify.forward_event(request=req, db=db, _auth=None))

    assert result["status"] == "received"
    # The catalog re-sync was scheduled for the resolved workspace + event.
    assert calls.get("workspace_id") == "ws-123"
    assert calls.get("event") == "products/update"


def test_non_catalog_event_does_not_resync(monkeypatch):
    """A non-catalog event (e.g. app/uninstalled) must NOT trigger a catalog sync."""
    ws = _FakeWorkspace()
    db = _FakeDb(ws)

    fired = {"count": 0}

    async def _fake_sync(workspace_id, event):
        fired["count"] += 1

    monkeypatch.setattr(shopify, "_sync_catalog_for_workspace", _fake_sync)

    req = shopify.EventRequest(shop="inbuilduk.myshopify.com", event="app/uninstalled")
    result = _run_and_drain(shopify.forward_event(request=req, db=db, _auth=None))

    assert result["status"] == "received"
    assert fired["count"] == 0


def test_catalog_event_unknown_shop_is_safe(monkeypatch):
    """A catalog event for an unknown shop resolves to no workspace and no-ops safely."""
    db = _FakeDb(None)  # no workspace for this shop

    fired = {"count": 0}

    async def _fake_sync(workspace_id, event):
        fired["count"] += 1

    monkeypatch.setattr(shopify, "_sync_catalog_for_workspace", _fake_sync)

    req = shopify.EventRequest(shop="ghost.myshopify.com", event="products/update")
    result = _run_and_drain(shopify.forward_event(request=req, db=db, _auth=None))

    assert result["status"] == "received"
    assert fired["count"] == 0
