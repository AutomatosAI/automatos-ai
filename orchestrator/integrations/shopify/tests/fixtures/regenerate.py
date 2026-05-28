"""Regenerate PRD-141 US-004 fixtures.

This script builds the synthetic INBUILD-flavored knowledge graph, hand-crafted
proactive-opener and cart-idle page contexts, then runs the CURRENT (pre-refactor)
``api.widgets.chat`` proactive helpers against them to capture verbatim expected
output strings.

Why a script + AST extraction rather than a normal import? ``api.widgets.chat``
pulls in the full FastAPI / SQLAlchemy / RAG / multimodal dependency tree, which
requires a configured database and optional native libs (camelot, etc) to import.
For a fixture generator that just needs four pure-ish functions, we parse chat.py
with ``ast`` and exec only the four function definitions plus their two helpers
into an isolated namespace. The function source is byte-identical to chat.py's,
so the captured outputs ARE the chat.py outputs.

If chat.py changes the proactive helpers, re-run this script and re-review the
generated fixtures before the change merges:

    cd orchestrator/integrations/shopify/tests/fixtures
    python3 regenerate.py

The graph is intentionally synthetic but representative — see README.md.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import sys
import types
from pathlib import Path
from typing import Optional  # noqa: F401 — injected into exec namespace

import networkx as nx
from networkx.readwrite import json_graph


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent
# orchestrator root: parent of fixtures (tests) → parent (shopify) → parent
# (integrations) → parent (orchestrator).
ORCHESTRATOR_ROOT = FIXTURES_DIR.parents[3]
CHAT_PY = ORCHESTRATOR_ROOT / "api" / "widgets" / "chat.py"


# ---------------------------------------------------------------------------
# Synthetic graph build — INBUILD UK fire/safety vertical flavour
# ---------------------------------------------------------------------------

def build_graph() -> nx.Graph:
    """Build the synthetic INBUILD-style knowledge graph.

    Nodes encode Shopify products and collections; edges encode the three
    relations the proactive opener cares about: ``frequently_bought_with``
    (real customer co-purchase signal), ``in_collection`` (catalog framing),
    and ``by_vendor`` (vendor sibling fallback).

    Co-counts and total_orders values are illustrative but plausible for a UK
    fire-safety trade shop. NOT real INBUILD data.
    """
    g = nx.Graph()

    # ---- Products (file_type="shopify_product") ----
    # NodeID == Shopify handle to keep things readable; attrs.handle is what
    # _resolve_graph_related_products looks for.
    products = [
        ("hochiki-aln", "Hochiki ALN Optical Smoke Detector"),
        ("hochiki-acb", "Hochiki ACB Multi-Sensor Detector"),
        ("hochiki-atg", "Hochiki ATG Heat Detector"),
        ("hochiki-ybo", "Hochiki YBO Manual Call Point"),
        ("hochiki-banshee", "Hochiki Banshee Wall Sounder"),
        ("hochiki-base-ybn", "Hochiki YBN Detector Base"),
        ("apollo-xp95-optical", "Apollo XP95 Optical Detector"),
        ("apollo-xp95-heat", "Apollo XP95 Heat Detector"),
        ("advanced-mxpro5-4loop", "Advanced MxPro5 4-Loop Panel"),
        ("kentec-syncro-2loop", "Kentec Syncro 2-Loop Panel"),
    ]
    for handle, label in products:
        g.add_node(
            handle,
            label=label,
            file_type="shopify_product",
            attrs={"handle": handle},
        )

    # ---- Collections (file_type="shopify_collection") ----
    collections = [
        ("col-addressable-fire-systems", "Addressable Fire Systems"),
        ("col-hochiki-range", "Hochiki Range"),
    ]
    for cid, label in collections:
        g.add_node(
            cid,
            label=label,
            file_type="shopify_collection",
            attrs={"handle": cid.replace("col-", "")},
        )

    # ---- FBT edges (frequently_bought_with) ----
    # (a, b, co_count, total_orders) — co_count is "bought together in N
    # orders out of total_orders". Insertion order matters for ties; we order
    # so the seed=hochiki-aln walk yields the expected top-1 first.
    fbt_edges = [
        ("hochiki-aln",  "hochiki-base-ybn",      42, 57),  # very strong pair
        ("hochiki-aln",  "hochiki-atg",           18, 57),
        ("hochiki-aln",  "advanced-mxpro5-4loop", 12, 57),
        ("hochiki-aln",  "hochiki-ybo",            9, 57),
        ("hochiki-aln",  "hochiki-banshee",        7, 57),
        ("hochiki-acb",  "hochiki-base-ybn",      24, 31),
        ("hochiki-acb",  "hochiki-banshee",       15, 31),
        ("hochiki-acb",  "advanced-mxpro5-4loop",  8, 31),
        ("hochiki-atg",  "hochiki-base-ybn",      20, 31),
        ("hochiki-atg",  "advanced-mxpro5-4loop", 11, 31),
        ("apollo-xp95-optical", "apollo-xp95-heat", 16, 22),
        ("apollo-xp95-optical", "kentec-syncro-2loop", 7, 22),
    ]
    for a, b, co_count, total_orders in fbt_edges:
        g.add_edge(
            a, b,
            relation="frequently_bought_with",
            confidence_score=0.85,
            weight=co_count,
            attrs={"co_count": co_count, "total_orders": total_orders},
        )

    # ---- in_collection edges (product -> collection) ----
    # Insertion order matters: first edge of this relation type wins
    # the "take 1 in_collection" pick for the proactive opener.
    in_collection_edges = [
        ("hochiki-aln", "col-addressable-fire-systems"),
        ("hochiki-aln", "col-hochiki-range"),
        ("hochiki-acb", "col-addressable-fire-systems"),
        ("hochiki-acb", "col-hochiki-range"),
        ("hochiki-atg", "col-hochiki-range"),
        ("hochiki-ybo", "col-hochiki-range"),
        ("hochiki-banshee", "col-hochiki-range"),
        ("hochiki-base-ybn", "col-hochiki-range"),
        ("apollo-xp95-optical", "col-addressable-fire-systems"),
        ("apollo-xp95-heat", "col-addressable-fire-systems"),
        ("advanced-mxpro5-4loop", "col-addressable-fire-systems"),
        ("kentec-syncro-2loop", "col-addressable-fire-systems"),
    ]
    for a, b in in_collection_edges:
        g.add_edge(
            a, b,
            relation="in_collection",
            confidence_score=0.9,
            weight=1.0,
            attrs={},
        )

    # ---- by_vendor edges (same vendor sibling — symmetric) ----
    # Keep sparse but realistic. hochiki-aln needs at least one by_vendor
    # edge so the proactive opener has a vendor sibling to mention.
    by_vendor_edges = [
        ("hochiki-aln", "hochiki-acb"),
        ("hochiki-aln", "hochiki-banshee"),
        ("hochiki-acb", "hochiki-atg"),
        ("apollo-xp95-optical", "apollo-xp95-heat"),
    ]
    for a, b in by_vendor_edges:
        g.add_edge(
            a, b,
            relation="by_vendor",
            confidence_score=0.6,
            weight=1.0,
            attrs={},
        )

    return g


# ---------------------------------------------------------------------------
# Page contexts — hand-crafted, plausible for INBUILD UK
# ---------------------------------------------------------------------------

PRODUCT_PAGE_CONTEXT: dict = {
    "pageType": "product",
    "productHandle": "hochiki-aln",
    "productTitle": "Hochiki ALN Optical Smoke Detector",
    "productType": "Smoke Detector",
    "productPrice": "32.50",
    "productVendor": "Hochiki",
    "productImageUrl": (
        "https://example-cdn.shopify.com/products/hochiki-aln.jpg"
    ),
    "productCollection": "Addressable Fire Systems",
    "productAvailable": True,
    "cartItemCount": 0,
    "cartCurrency": "GBP",
    "shopCurrency": "GBP",
    "shopDomain": "inbuild-uk.myshopify.com",
    "shopLocale": "en-GB",
}

CART_IDLE_CONTEXT: dict = {
    "pageType": "cart",
    "cartItemCount": 5,
    "cartItems": [
        {
            "handle": "hochiki-aln",
            "title": "Hochiki ALN Optical Smoke Detector",
            "qty": 2,
        },
        {
            "handle": "hochiki-acb",
            "title": "Hochiki ACB Multi-Sensor Detector",
            "qty": 1,
        },
        {
            "handle": "hochiki-atg",
            "title": "Hochiki ATG Heat Detector",
            "qty": 2,
        },
    ],
    "cartTotalPrice": 16250,  # minor units — Shopify cents/pence: £162.50
    "cartCurrency": "GBP",
    "cartUrl": "https://inbuild-uk.myshopify.com/cart",
    "shopCurrency": "GBP",
    "shopDomain": "inbuild-uk.myshopify.com",
    "shopLocale": "en-GB",
}


# ---------------------------------------------------------------------------
# Extract the proactive helpers from chat.py and exec them in isolation
# ---------------------------------------------------------------------------

# PRD-141 US-005 moved ``_OPENER_CONTEXT_FIELDS`` and
# ``_format_opener_context_value`` to
# ``orchestrator/integrations/shopify/context_fields.py``. They're no
# longer extractable from chat.py but the AST-exec'd function bodies
# still close over them — ``_extract_chat_helpers`` imports them from
# their new home and seeds the namespace before exec'ing.
WANTED_NAMES = {
    "_build_proactive_opener_message",
    "_build_cart_idle_opener_message",
    "_resolve_graph_related_products",
    "_resolve_cart_recommendations",
}


def _extract_chat_helpers(graph: nx.Graph) -> dict:
    """Parse chat.py with ast, return a namespace containing the four helpers.

    Stubs ``modules.knowledge.graph_service.GraphifyService`` to hand back
    the supplied in-memory graph so ``_resolve_graph_related_products`` /
    ``_resolve_cart_recommendations`` can run without a real workspace.

    Seeds the namespace with ``_OPENER_CONTEXT_FIELDS`` and
    ``_format_opener_context_value`` (lifted to
    ``integrations.shopify.context_fields`` in US-005) so the exec'd
    ``_build_proactive_opener_message`` body can resolve them.
    """
    if str(ORCHESTRATOR_ROOT) not in sys.path:
        sys.path.insert(0, str(ORCHESTRATOR_ROOT))
    from integrations.shopify.context_fields import (
        _OPENER_CONTEXT_FIELDS,
        _format_opener_context_value,
    )

    src = CHAT_PY.read_text()
    tree = ast.parse(src)

    # Pre-populate the exec namespace with the names the function bodies
    # close over at module scope.
    ns: dict = {
        "Optional": Optional,
        "logger": logging.getLogger("fixture_generator"),
        "__name__": "_chat_extracted",
        "_OPENER_CONTEXT_FIELDS": _OPENER_CONTEXT_FIELDS,
        "_format_opener_context_value": _format_opener_context_value,
    }

    for node in tree.body:
        name: Optional[str] = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id

        if name in WANTED_NAMES:
            module = ast.Module(body=[node], type_ignores=[])
            code = compile(module, str(CHAT_PY), "exec")
            exec(code, ns)

    missing = WANTED_NAMES - set(ns)
    if missing:
        raise RuntimeError(
            f"Failed to extract from chat.py: {sorted(missing)}. "
            "Has chat.py been refactored ahead of this script?"
        )

    # Stub modules.knowledge.graph_service so the in-function imports resolve.
    fake_mod = types.ModuleType("modules.knowledge.graph_service")

    class _FakeGraphifyService:
        async def load_graph(self, workspace_id):  # noqa: D401
            return graph

    fake_mod.GraphifyService = _FakeGraphifyService
    # Ensure the parent packages exist in sys.modules so `from
    # modules.knowledge.graph_service import GraphifyService` can resolve
    # even if the orchestrator's `modules` package isn't on sys.path.
    sys.modules.setdefault("modules", types.ModuleType("modules"))
    sys.modules.setdefault("modules.knowledge", types.ModuleType("modules.knowledge"))
    sys.modules["modules.knowledge.graph_service"] = fake_mod

    return ns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[regen] Building synthetic graph...")
    graph = build_graph()
    print(
        f"[regen]   nodes={graph.number_of_nodes()} "
        f"edges={graph.number_of_edges()}"
    )

    print(f"[regen] Extracting helpers from {CHAT_PY.relative_to(ORCHESTRATOR_ROOT)}...")
    ns = _extract_chat_helpers(graph)

    # ---- Run product-page proactive opener ----
    print("[regen] Running _resolve_graph_related_products + _build_proactive_opener_message...")
    related = asyncio.run(
        ns["_resolve_graph_related_products"](
            "inbuild-fixture-workspace", PRODUCT_PAGE_CONTEXT
        )
    )
    print(f"[regen]   related_products = {related}")
    product_opener = ns["_build_proactive_opener_message"](
        PRODUCT_PAGE_CONTEXT, related
    )

    # ---- Run cart-idle opener ----
    print("[regen] Running _resolve_cart_recommendations + _build_cart_idle_opener_message...")
    cart_recs = asyncio.run(
        ns["_resolve_cart_recommendations"](
            "inbuild-fixture-workspace", CART_IDLE_CONTEXT
        )
    )
    print(f"[regen]   cart recommendations = {cart_recs}")
    cart_opener = ns["_build_cart_idle_opener_message"](
        CART_IDLE_CONTEXT, cart_recs
    )

    # ---- Persist ----
    graph_path = FIXTURES_DIR / "inbuild_graph_snapshot.json"
    product_ctx_path = FIXTURES_DIR / "product_page_context.json"
    cart_ctx_path = FIXTURES_DIR / "cart_idle_context.json"
    product_opener_path = FIXTURES_DIR / "expected_product_page_opener.txt"
    cart_opener_path = FIXTURES_DIR / "expected_cart_idle_opener.txt"

    # node_link_data round-trips cleanly via json_graph.node_link_graph.
    # edges="edges" silences the NetworkX 3.4 deprecation warning.
    graph_data = json_graph.node_link_data(graph, edges="edges")
    graph_path.write_text(json.dumps(graph_data, indent=2, sort_keys=True))
    product_ctx_path.write_text(json.dumps(PRODUCT_PAGE_CONTEXT, indent=2, sort_keys=True))
    cart_ctx_path.write_text(json.dumps(CART_IDLE_CONTEXT, indent=2, sort_keys=True))
    product_opener_path.write_text(product_opener)
    cart_opener_path.write_text(cart_opener)

    for p in (graph_path, product_ctx_path, cart_ctx_path,
              product_opener_path, cart_opener_path):
        print(f"[regen] wrote {p.relative_to(ORCHESTRATOR_ROOT)} ({p.stat().st_size} bytes)")

    print("[regen] Done.")
    print("[regen] --- product-page opener ---")
    print(product_opener)
    print("[regen] --- cart-idle opener ---")
    print(cart_opener)


if __name__ == "__main__":
    # Hash randomisation can perturb set iteration; lock it so candidate
    # ranking ties (if any) resolve consistently across runs.
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
