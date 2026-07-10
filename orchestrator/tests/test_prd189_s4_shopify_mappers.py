"""PRD-189 S4 (F3) — behavioral tests for the Shopify graph mappers.

``map_shopify_catalog`` / ``map_shopify_orders`` produce every fact the
storefront widget cites to real shoppers ("bought together in X of Y orders"),
and since F032 the catalog mapper runs automatically on every catalog webhook
— yet until this file their only coverage was a registry identity check. A
silent mapper regression would corrupt every opener with a confident,
provenance-styled sentence and nothing would catch it.

Both mappers are pure, deterministic and IO-free: a recorded Bulk-Op JSONL
fixture in (``integrations/shopify/tests/fixtures/``), assert the graph out.
No mocks, no DB, no network.

Fixture co-purchase ground truth (min_support=2):
    valid orders (>=2 products, not cancelled): 5001, 5002, 5003, 5006, 5007 → 5
    pair (9001, 9002): orders 5001+5002+5003          → co_count 3 → edge
    pair (9002, 9003): orders 5002+5006+5007          → co_count 3 → edge
    pair (9001, 9003): order  5002 only               → co_count 1 → gated
        (order 5004 also paired them but is CANCELLED — counting it would
        make co_count 2 and wrongly emit the edge)
    order 5005 is single-item (no co-occurrence), line item 6015 has a null
    variant (deleted product) — both must be ignored without error.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

from modules.knowledge.graph_extraction import (  # noqa: E402
    SHOPIFY_CATALOG_SOURCE,
    map_shopify_catalog,
    map_shopify_orders,
)
from tests.helpers_shopify_sync import (  # noqa: E402
    CATALOG_JSONL_PATH,
    ORDERS_JSONL_PATH,
)


def _catalog_lines() -> list[str]:
    return CATALOG_JSONL_PATH.read_text().splitlines()


def _orders_lines() -> list[str]:
    return ORDERS_JSONL_PATH.read_text().splitlines()


def _nodes_by_type(graph: dict) -> dict[str, dict[str, dict]]:
    by_type: dict[str, dict[str, dict]] = {}
    for node in graph["nodes"]:
        by_type.setdefault(node["file_type"], {})[node["id"]] = node
    return by_type


def _edges_by_relation(graph: dict) -> dict[str, list[dict]]:
    by_rel: dict[str, list[dict]] = {}
    for edge in graph["edges"]:
        by_rel.setdefault(edge["relation"], []).append(edge)
    return by_rel


# ---------------------------------------------------------------------------
# map_shopify_catalog — node types, edge relations, attrs, provenance
# ---------------------------------------------------------------------------


def test_map_shopify_catalog_produces_typed_nodes_and_edges():
    graph = map_shopify_catalog(_catalog_lines())
    nodes = _nodes_by_type(graph)
    edges = _edges_by_relation(graph)

    # Products with grounding attrs the opener leans on.
    products = nodes["shopify_product"]
    assert set(products) == {
        "shopify_product_9001",
        "shopify_product_9002",
        "shopify_product_9003",
    }
    lamp = products["shopify_product_9001"]
    assert lamp["label"] == "Aurora Desk Lamp"
    assert lamp["attrs"]["handle"] == "aurora-desk-lamp"
    assert lamp["attrs"]["vendor"] == "Lumenworks"
    assert lamp["attrs"]["price_min"] == "49.0"
    assert lamp["attrs"]["currency"] == "GBP"
    assert lamp["attrs"]["description"] == "Warm LED desk lamp with a weighted base."

    # Variants → variant_of their parent product.
    assert set(nodes["shopify_variant"]) == {
        "shopify_variant_9101",
        "shopify_variant_9102",
        "shopify_variant_9103",
    }
    variant_of = {
        (e["source"], e["target"]) for e in edges["variant_of"]
    }
    assert ("shopify_variant_9101", "shopify_product_9001") in variant_of
    assert len(variant_of) == 3

    # One collection (deduped across its two member lines) + in_collection.
    assert set(nodes["shopify_collection"]) == {"shopify_collection_7001"}
    in_collection = {
        (e["source"], e["target"]) for e in edges["in_collection"]
    }
    assert in_collection == {
        ("shopify_product_9001", "shopify_collection_7001"),
        ("shopify_product_9002", "shopify_collection_7001"),
    }

    # Vendors derived + deduped (two Lumenworks products, one vendor node).
    assert set(nodes["shopify_vendor"]) == {
        "shopify_vendor_lumenworks",
        "shopify_vendor_voltbay",
    }
    by_vendor = {(e["source"], e["target"]) for e in edges["by_vendor"]}
    assert by_vendor == {
        ("shopify_product_9001", "shopify_vendor_lumenworks"),
        ("shopify_product_9002", "shopify_vendor_lumenworks"),
        ("shopify_product_9003", "shopify_vendor_voltbay"),
    }

    # Metafield node + has_metafield edge.
    metafield = nodes["shopify_metafield"]["shopify_metafield_8001"]
    assert metafield["label"] == "specs.wattage"
    assert metafield["attrs"]["value"] == "8W"
    assert [(e["source"], e["target"]) for e in edges["has_metafield"]] == [
        ("shopify_product_9001", "shopify_metafield_8001")
    ]

    # No other relations sneak in — this is the full catalog edge vocabulary.
    assert set(edges) == {"variant_of", "in_collection", "by_vendor", "has_metafield"}


def test_map_shopify_catalog_tags_catalog_provenance():
    """Every catalog node/edge carries the shopify://catalog source tag — the
    provenance S1's preserve-merge uses to tell catalog content apart from the
    intelligence a rebuild must keep."""
    bulk_op = "gid://shopify/BulkOperation/prov-1"
    graph = map_shopify_catalog(_catalog_lines(), bulk_op_id=bulk_op)

    expected = f"{SHOPIFY_CATALOG_SOURCE}#{bulk_op}"
    assert graph["nodes"], "fixture catalog must produce nodes"
    assert all(n["source_file"] == expected for n in graph["nodes"])
    assert all(e["source_file"] == expected for e in graph["edges"])

    # Without a bulk-op id the bare catalog source still prefixes correctly.
    untagged = map_shopify_catalog(_catalog_lines())
    assert all(
        n["source_file"] == SHOPIFY_CATALOG_SOURCE for n in untagged["nodes"]
    )


def test_map_shopify_catalog_skips_malformed_lines():
    lines = ["not-json{{{", "", *_catalog_lines()]
    graph = map_shopify_catalog(lines)
    assert len(_nodes_by_type(graph)["shopify_product"]) == 3


# ---------------------------------------------------------------------------
# map_shopify_orders — FBT math, min_support, cancellation, privacy
# ---------------------------------------------------------------------------


def test_map_shopify_orders_fbt_math():
    """co_count / total_orders / confidence must be the exact co-purchase
    arithmetic — these numbers are cited verbatim to shoppers."""
    graph = map_shopify_orders(_orders_lines(), min_support=2)

    edges = {
        (e["source"], e["target"]): e for e in graph["edges"]
    }
    assert set(edges) == {
        ("shopify_product_9001", "shopify_product_9002"),
        ("shopify_product_9002", "shopify_product_9003"),
    }

    lamp_bulbs = edges[("shopify_product_9001", "shopify_product_9002")]
    assert lamp_bulbs["relation"] == "frequently_bought_with"
    assert lamp_bulbs["attrs"] == {"co_count": 3, "total_orders": 5}
    assert lamp_bulbs["weight"] == 3.0
    assert lamp_bulbs["confidence_score"] == 3 / 5

    bulbs_cables = edges[("shopify_product_9002", "shopify_product_9003")]
    assert bulbs_cables["attrs"] == {"co_count": 3, "total_orders": 5}


def test_map_shopify_orders_min_support_gate():
    """A pair below min_support is gated out; lowering the gate admits it with
    its true (cancellation-excluded) count."""
    gated = map_shopify_orders(_orders_lines(), min_support=2)
    gated_pairs = {(e["source"], e["target"]) for e in gated["edges"]}
    assert ("shopify_product_9001", "shopify_product_9003") not in gated_pairs

    open_gate = map_shopify_orders(_orders_lines(), min_support=1)
    edges = {(e["source"], e["target"]): e for e in open_gate["edges"]}
    weak = edges[("shopify_product_9001", "shopify_product_9003")]
    assert weak["attrs"]["co_count"] == 1


def test_map_shopify_orders_excludes_cancelled_orders():
    """Order 5004 (cancelled) pairs 9001+9003 — counting it would lift that
    pair to co_count 2 and wrongly emit an edge at min_support=2, and would
    inflate total_orders to 6. Cancelled orders are not revealed preference."""
    graph = map_shopify_orders(_orders_lines(), min_support=2)

    pairs = {(e["source"], e["target"]) for e in graph["edges"]}
    assert ("shopify_product_9001", "shopify_product_9003") not in pairs
    assert all(
        e["attrs"]["total_orders"] == 5 for e in graph["edges"]
    ), "cancelled and single-item orders must not inflate the denominator"


def test_map_shopify_orders_emits_no_customer_nodes():
    """Privacy by design: only aggregated product↔product edges — never a
    customer, order, or line-item node, whatever the JSONL carries."""
    graph = map_shopify_orders(_orders_lines(), min_support=1)

    assert graph["nodes"] == []
    assert graph["hyperedges"] == []
    for edge in graph["edges"]:
        assert edge["source"].startswith("shopify_product_")
        assert edge["target"].startswith("shopify_product_")
        assert edge["relation"] == "frequently_bought_with"


def test_map_shopify_orders_empty_input_is_empty_graph():
    graph = map_shopify_orders([], min_support=2)
    assert graph == {"nodes": [], "edges": [], "hyperedges": []}
