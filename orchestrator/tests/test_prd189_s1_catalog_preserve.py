"""PRD-189 S1 (F1) — a catalog re-sync preserves cross-sell + non-catalog graph.

The wipe this pins: ``_product_sync_impl`` ended in ``import_graph(merge=False)``
— a full graph replacement with only the catalog nodes/edges — so every catalog
sync (fired by every catalog webhook since F032) erased the
``frequently_bought_with`` edges the orders sync had computed and every
non-catalog node merged in since. The pilot's ``orders_sync`` block reported
``fbt_edges_added: 16`` while the persisted graph held 0.

These tests drive the REAL ``_product_sync_impl`` end-to-end with recorded
bulk-op JSONL fixtures — Composio/httpx mocked at the boundary (the
``test_prd183_s1_catalog_webhook.py`` posture) and the REAL ``GraphifyService``
merge pipeline with only its DB-backed writers stubbed (the
``test_prd164_flywheel.py`` posture). No DB, no network, no live store.

Pre-state per test: catalog import → orders merge (real mappers, real import
path) → a flywheel/document-sourced node merged in. Then a catalog re-sync runs
over an updated catalog fixture and the graph must keep what the catalog
bulk-op does not itself carry.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from types import SimpleNamespace  # noqa: E402
from uuid import uuid4  # noqa: E402

import pytest  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import integrations  # noqa: E402,F401 — registers the shopify graph-source mappers

from tests.helpers_shopify_sync import (  # noqa: E402
    CATALOG_JSONL_PATH,
    CATALOG_RESYNC_JSONL_PATH,
    ORDERS_JSONL_PATH,
    FakeDb,
    FakeWorkspace,
    fbt_edges_of,
    make_graph_service,
    mock_sync_boundaries,
)

# A flywheel/document-sourced node + edge, exactly what the old merge=False
# replacement used to erase alongside the FBT edges.
_FLYWHEEL_NODE_ID = "concept_ambient_lighting_strategy"
_FLYWHEEL_DELTA = {
    "nodes": [
        {
            "id": _FLYWHEEL_NODE_ID,
            "label": "Ambient Lighting Strategy",
            "file_type": "concept",
            "source_file": "documents/lighting-strategy.md",
            "source_location": None,
            "confidence": "EXTRACTED",
            "weight": 1.0,
            "team_access": [],
        }
    ],
    "edges": [
        {
            "source": _FLYWHEEL_NODE_ID,
            "target": "shopify_product_9001",
            "relation": "mentions",
            "confidence": "EXTRACTED",
            "confidence_score": 0.9,
            "source_file": "documents/lighting-strategy.md",
            "source_location": None,
            "weight": 1.0,
            "_src": _FLYWHEEL_NODE_ID,
            "_tgt": "shopify_product_9001",
        }
    ],
    "hyperedges": [],
}


async def _seed_synced_workspace(monkeypatch, workspace_id: str):
    """Catalog + orders + flywheel content through the REAL import path."""
    from modules.knowledge.graph_extraction import (
        map_shopify_catalog,
        map_shopify_orders,
    )
    from tests.helpers_shopify_sync import silence_graph_primitive

    silence_graph_primitive(monkeypatch)
    svc = make_graph_service()

    catalog = map_shopify_catalog(
        CATALOG_JSONL_PATH.read_text().splitlines(),
        bulk_op_id="gid://shopify/BulkOperation/seed-catalog",
    )
    await svc.import_graph(workspace_id, catalog, merge=False)

    orders = map_shopify_orders(
        ORDERS_JSONL_PATH.read_text().splitlines(),
        bulk_op_id="gid://shopify/BulkOperation/seed-orders",
        min_support=2,
    )
    await svc.import_graph(workspace_id, orders, merge=True)

    await svc.import_graph(workspace_id, dict(_FLYWHEEL_DELTA), merge=True)
    return svc


async def _run_catalog_resync(monkeypatch, svc, workspace_id: str):
    """The re-sync under test: REAL ``_product_sync_impl`` over the updated
    catalog fixture, externals mocked at the boundary."""
    from api import shopify

    workspace = FakeWorkspace(workspace_id)
    db = FakeDb(workspace)
    mock_sync_boundaries(
        monkeypatch,
        graph_service=svc,
        jsonl_texts=[CATALOG_RESYNC_JSONL_PATH.read_text()],
    )
    response = await shopify._product_sync_impl(workspace_id, db)
    return response, workspace


@pytest.mark.asyncio
async def test_catalog_resync_preserves_fbt_edges(monkeypatch):
    """The marquee restore: frequently_bought_with edges survive a catalog
    re-sync with their co_count/total_orders provenance intact (before the fix
    this graph came back with 0 FBT edges)."""
    workspace_id = str(uuid4())
    svc = await _seed_synced_workspace(monkeypatch, workspace_id)

    before = fbt_edges_of(await svc.load_graph(workspace_id))
    assert len(before) == 2, "seed must hold the two fixture FBT edges"

    response, workspace = await _run_catalog_resync(monkeypatch, svc, workspace_id)
    assert response.status == "complete"

    graph = await svc.load_graph(workspace_id)
    after = fbt_edges_of(graph)
    assert len(after) == 2, "catalog re-sync must not wipe FBT edges"

    pairs = {frozenset((u, v)): attrs for u, v, attrs in after}
    lamp_bulbs = pairs[frozenset(("shopify_product_9001", "shopify_product_9002"))]
    assert lamp_bulbs.get("attrs", {}).get("co_count") == 3
    assert lamp_bulbs.get("attrs", {}).get("total_orders") == 5

    # The status block records what the merge preserved — the number S2 gates on.
    assert workspace.settings["product_sync"]["fbt_edges_preserved"] == 2


@pytest.mark.asyncio
async def test_catalog_resync_preserves_non_catalog_nodes(monkeypatch):
    """A flywheel/document-sourced node (and its edge) present before the sync
    is still present after — the old replacement dropped every non-catalog
    node merged in since the last catalog sync."""
    workspace_id = str(uuid4())
    svc = await _seed_synced_workspace(monkeypatch, workspace_id)

    response, _ = await _run_catalog_resync(monkeypatch, svc, workspace_id)
    assert response.status == "complete"

    graph = await svc.load_graph(workspace_id)
    assert _FLYWHEEL_NODE_ID in graph, "document-sourced node must survive"
    assert graph.nodes[_FLYWHEEL_NODE_ID].get("file_type") == "concept"
    assert graph.has_edge(_FLYWHEEL_NODE_ID, "shopify_product_9001")


@pytest.mark.asyncio
async def test_catalog_resync_refreshes_catalog_and_drops_deleted(monkeypatch):
    """Preservation must not freeze the catalog: the fresh bulk-op wins for
    catalog content (attrs refresh, store-deleted products drop out) while
    the co-purchase history it never carried stays."""
    workspace_id = str(uuid4())
    svc = await _seed_synced_workspace(monkeypatch, workspace_id)

    response, _ = await _run_catalog_resync(monkeypatch, svc, workspace_id)
    assert response.status == "complete"

    graph = await svc.load_graph(workspace_id)

    # Fresh catalog attrs win — the Mk II rename and price change land.
    lamp = graph.nodes["shopify_product_9001"]
    assert lamp.get("label") == "Aurora Desk Lamp Mk II"
    assert lamp.get("attrs", {}).get("price_min") == "54.0"
    metafield = graph.nodes["shopify_metafield_8001"]
    assert metafield.get("attrs", {}).get("value") == "9W"

    # The store-deleted product's CATALOG presence is gone: its variant and
    # single-product vendor disappear, and the node itself keeps no catalog
    # attrs (it remains only as a bare endpoint of preserved FBT history,
    # which the widget resolvers skip).
    assert "shopify_variant_9103" not in graph
    assert "shopify_vendor_voltbay" not in graph
    assert graph.nodes["shopify_product_9003"].get("file_type") is None

    # ...while the FBT edge that references it is retained until the next
    # orders sync refreshes the co-purchase set.
    pairs = {frozenset((u, v)) for u, v, _ in fbt_edges_of(graph)}
    assert frozenset(("shopify_product_9002", "shopify_product_9003")) in pairs


@pytest.mark.asyncio
async def test_first_sync_without_existing_graph_imports_catalog_unchanged():
    """No existing graph → nothing to preserve; the fresh catalog is imported
    as-is (the merge helper is a pass-through, not an error)."""
    from api.shopify import _merge_catalog_over_existing

    async def _no_graph(_ws):
        return None

    gs = SimpleNamespace(load_graph=_no_graph)
    catalog = {"nodes": [{"id": "n1"}], "edges": [], "hyperedges": []}

    combined, stats = await _merge_catalog_over_existing(gs, "ws-1", catalog)

    assert combined is catalog
    assert stats == {
        "nodes_preserved": 0,
        "edges_preserved": 0,
        "fbt_edges_preserved": 0,
    }
