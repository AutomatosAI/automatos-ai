"""PRD-189 S2 (F1-guard) — FBT-persistence integrity check + Commerce tile.

The wipe stayed invisible because the sync status blocks reported a sync *ran*
while the graph no longer held what it computed: the pilot's ``orders_sync``
said ``fbt_edges_added: 16`` with 0 ``frequently_bought_with`` edges present,
and nothing read the drift. These tests pin the guardrail both ways:

* ``test_fbt_integrity_detects_drift`` — the exact 16-reported/0-present state
  reads as a drift through the Commerce tile endpoint (own-workspace strip).
* ``test_fbt_integrity_clean_after_resync`` — with S1 in place, a full
  orders-sync → catalog-re-sync loop keeps reported == present and both sync
  status blocks record ``fbt_integrity.ok == True``.

Pure: recorded JSONL fixtures via the shared harness, Composio/httpx and graph
persistence mocked at the boundary, no DB, no network.
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

import networkx as nx  # noqa: E402
import pytest  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import integrations  # noqa: E402,F401 — registers the shopify graph-source mappers

from integrations.shopify.integrity import count_fbt_edges, fbt_integrity  # noqa: E402
from tests.helpers_shopify_sync import (  # noqa: E402
    CATALOG_JSONL_PATH,
    CATALOG_RESYNC_JSONL_PATH,
    ORDERS_JSONL_PATH,
    FakeDb,
    FakeWorkspace,
    make_graph_service,
    mock_sync_boundaries,
    silence_graph_primitive,
)


# ---------------------------------------------------------------------------
# Pure helper behaviour
# ---------------------------------------------------------------------------


def test_count_fbt_edges_counts_only_fbt():
    g = nx.Graph()
    g.add_edge("a", "b", relation="frequently_bought_with")
    g.add_edge("b", "c", relation="in_collection")
    g.add_edge("c", "d", relation="FREQUENTLY_BOUGHT_WITH")  # case-insensitive
    g.add_edge("d", "e")  # no relation at all
    assert count_fbt_edges(g) == 2


def test_fbt_integrity_reports_the_wipe_shape():
    """16 reported / 0 present — the pilot's exact lying state."""
    report = fbt_integrity(16, 0)
    assert report == {"reported": 16, "present": 0, "drift": -16, "ok": False}


def test_fbt_integrity_clean_and_unknown_states():
    assert fbt_integrity(2, 2) == {
        "reported": 2,
        "present": 2,
        "drift": 0,
        "ok": True,
    }
    # No orders sync yet → honest unknown, never a fabricated green.
    assert fbt_integrity(None, 0) == {
        "reported": None,
        "present": 0,
        "drift": None,
        "ok": None,
    }


# ---------------------------------------------------------------------------
# The Commerce tile endpoint (own-workspace strip, PRD-185 S12 posture)
# ---------------------------------------------------------------------------


def _patch_tile_graph(monkeypatch, graph):
    """Point the tile's lazy GraphifyService at a fixed graph."""
    import modules.knowledge.graph_service as graph_service_mod

    class _StubService:
        async def load_graph(self, _workspace_id):
            return graph

    monkeypatch.setattr(graph_service_mod, "GraphifyService", _StubService)


@pytest.mark.asyncio
async def test_fbt_integrity_detects_drift(monkeypatch):
    """A workspace whose last orders_sync reported 16 FBT edges but whose graph
    holds 0 must read as a drift — the single query that would have caught the
    wipe on day one."""
    from api.analytics_real import get_commerce_integrity

    workspace = FakeWorkspace(
        "ws-drift",
        settings={
            "shopify_domain": "fixture-lighting.myshopify.com",
            "orders_sync": {
                "status": "complete",
                "fbt_edges_added": 16,
                "completed_at": 1720500000.0,
            },
        },
    )
    wiped_graph = nx.Graph()
    wiped_graph.add_edge(
        "shopify_product_9001",
        "shopify_vendor_lumenworks",
        relation="by_vendor",
    )
    _patch_tile_graph(monkeypatch, wiped_graph)

    result = await get_commerce_integrity(
        ctx=SimpleNamespace(workspace_id="ws-drift"), db=FakeDb(workspace)
    )

    assert result["synced"] is True
    assert result["reported_fbt_edges"] == 16
    assert result["present_fbt_edges"] == 0
    assert result["drift"] == -16
    assert result["ok"] is False
    assert result["last_orders_sync_at"] == 1720500000.0


@pytest.mark.asyncio
async def test_tile_never_synced_workspace_is_honest_null(monkeypatch):
    """No commerce sync history → synced=false and nulls, not a fake green
    (and no graph load is attempted at all)."""
    from api.analytics_real import get_commerce_integrity

    workspace = FakeWorkspace("ws-plain", settings={})

    result = await get_commerce_integrity(
        ctx=SimpleNamespace(workspace_id="ws-plain"), db=FakeDb(workspace)
    )

    assert result["synced"] is False
    assert result["reported_fbt_edges"] is None
    assert result["present_fbt_edges"] is None
    assert result["ok"] is None


# ---------------------------------------------------------------------------
# Reported == present through the REAL sync path (S1 + S2 composed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fbt_integrity_clean_after_resync(monkeypatch):
    """After S1, the full loop — orders sync, then a catalog re-sync — keeps
    reported == present; both status blocks and the Commerce tile agree."""
    from api import shopify
    from api.analytics_real import get_commerce_integrity
    from modules.knowledge.graph_extraction import map_shopify_catalog

    workspace_id = str(uuid4())
    silence_graph_primitive(monkeypatch)
    svc = make_graph_service()

    # Initial catalog import (warm graph, no FBT yet) via the real import path.
    catalog = map_shopify_catalog(
        CATALOG_JSONL_PATH.read_text().splitlines(),
        bulk_op_id="gid://shopify/BulkOperation/seed-catalog",
    )
    await svc.import_graph(workspace_id, catalog, merge=False)

    workspace = FakeWorkspace(workspace_id)
    db = FakeDb(workspace)
    mock_sync_boundaries(
        monkeypatch,
        graph_service=svc,
        jsonl_texts=[
            ORDERS_JSONL_PATH.read_text(),
            CATALOG_RESYNC_JSONL_PATH.read_text(),
        ],
    )

    # Orders sync: reports 2 FBT edges and the merged graph holds exactly 2.
    orders_resp = await shopify._orders_sync_impl(workspace_id, 90, 2, db)
    assert orders_resp.status == "complete"
    assert orders_resp.fbt_edges_added == 2
    orders_block = workspace.settings["orders_sync"]
    assert orders_block["fbt_integrity"] == {
        "reported": 2,
        "present": 2,
        "drift": 0,
        "ok": True,
    }

    # Catalog re-sync: preservation keeps reported == present.
    catalog_resp = await shopify._product_sync_impl(workspace_id, db)
    assert catalog_resp.status == "complete"
    product_block = workspace.settings["product_sync"]
    assert product_block["fbt_edges_preserved"] == 2
    assert product_block["fbt_integrity"] == {
        "reported": 2,
        "present": 2,
        "drift": 0,
        "ok": True,
    }

    # And the Commerce tile reads the same clean state.
    tile = await get_commerce_integrity(
        ctx=SimpleNamespace(workspace_id=workspace_id), db=db
    )
    assert tile["synced"] is True
    assert tile["reported_fbt_edges"] == 2
    assert tile["present_fbt_edges"] == 2
    assert tile["ok"] is True
