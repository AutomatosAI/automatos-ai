"""Shopify vertical plugin — TEMPORARY shim delegating to chat.py.

PRD-141 US-003. Registered as ``PLUGIN_REGISTRY["shopify"]`` and used by
any workspace whose ``settings.vertical == "shopify"``.

This module is a **shim**, not a rewrite. It encapsulates the dispatch
contract — ``handle_widget_message`` matching the
:class:`integrations.WidgetPlugin` protocol — and delegates to the
remaining inline Shopify helpers still living in
``orchestrator/api/widgets/chat.py``:

* ``_build_proactive_opener_message`` (product-page directive builder)
* ``_build_cart_idle_opener_message`` (cart-idle directive builder)

The two graph resolvers (``_resolve_graph_related_products`` lifted in
US-006, ``_resolve_cart_recommendations`` lifted in US-007) now live
in this file and the shim calls them locally.

US-008 will move the two builders here. US-010 will delete the chat.py
inline dispatch and route every widget chat request through
``PLUGIN_REGISTRY``. At that point the imports below become local
definitions and this docstring's "shim" framing goes away.

The remaining chat.py imports happen **inside**
``handle_widget_message``, beneath the early-return gate. Two reasons:

1. Circular-import safety. During Phase 1 there is a window where
   chat.py imports back from this module (US-006/007/008 move helpers
   progressively). Lazy imports avoid that window without changing
   behaviour.
2. Pass-through paths must not pay the cost of loading the FastAPI
   router module (which pulls in database / auth dependencies). The
   gate is the hot path; the rewrite is the rare path.

The ``PROACTIVE_TRIGGER_REASONS`` frozenset from chat.py is
intentionally NOT imported here — the two trigger strings are
hardcoded inline so the gate works without touching chat.py. The
duplication disappears in US-010 when the constant moves alongside
the helpers it gates.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from integrations import WidgetPluginResult

logger = logging.getLogger(__name__)


async def _resolve_graph_related_products(
    workspace_id: str,
    page_context: dict,
    *,
    max_per_relation: int = 1,
) -> list[dict]:
    """Pull top related products from the workspace knowledge graph.

    Looks up the seed product node by Shopify handle in page_context, then
    walks 1-hop edges by relation type:
      - frequently_bought_with (highest co_count wins — real customer signal)
      - in_collection (most-connected sibling — likely a category anchor)
      - by_vendor (any same-vendor sibling)

    Returns at most ``max_per_relation`` entries per relation type. Empty
    list when no seed product, no graph, or any failure — caller treats
    that as "fall back to plain Layer-1 opener".
    """
    handle = (page_context or {}).get("productHandle")
    title = (page_context or {}).get("productTitle")
    if not handle and not title:
        return []  # No seed (cart/checkout/collection page) — nothing to traverse

    try:
        from modules.knowledge.graph_service import GraphifyService

        gs = GraphifyService()
        graph = await gs.load_graph(workspace_id)
        if graph is None:
            return []

        # Find the seed node by handle (preferred) or label match.
        seed_id = None
        for node_id, attrs in graph.nodes(data=True):
            node_attrs = attrs.get("attrs") or {}
            if handle and node_attrs.get("handle") == handle:
                seed_id = node_id
                break
        if seed_id is None and title:
            for node_id, attrs in graph.nodes(data=True):
                if (attrs.get("file_type") == "shopify_product"
                        and attrs.get("label") == title):
                    seed_id = node_id
                    break
        if seed_id is None:
            return []

        # Walk 1-hop, group by relation type, sort each group by signal strength.
        by_relation: dict[str, list[dict]] = {}
        for u, v, edata in graph.edges(seed_id, data=True):
            rel = (edata.get("relation") or "").lower()
            if rel not in ("frequently_bought_with", "in_collection", "by_vendor"):
                continue
            other = v if u == seed_id else u
            other_attrs = graph.nodes[other]
            # Only return product nodes (skip variants/vendors as targets —
            # those aren't recommendable on their own to a shopper)
            if other_attrs.get("file_type") not in ("shopify_product", "shopify_collection"):
                # in_collection targets ARE collections; that's fine for catalog framing.
                if rel != "in_collection":
                    continue
            edge_attrs = edata.get("attrs") or {}
            by_relation.setdefault(rel, []).append({
                "relation": rel,
                "label": other_attrs.get("label") or other,
                "type": other_attrs.get("file_type", ""),
                "confidence": edata.get("confidence_score", 0),
                "co_count": edge_attrs.get("co_count"),
                "total_orders": edge_attrs.get("total_orders"),
                "weight": edata.get("weight", 0),
            })

        # FBT: sort by co_count (raw signal strength).
        by_relation.get("frequently_bought_with", []).sort(
            key=lambda p: -(p.get("co_count") or 0)
        )
        # Collection / vendor: arbitrary — take first.

        out: list[dict] = []
        for rel in ("frequently_bought_with", "in_collection", "by_vendor"):
            out.extend(by_relation.get(rel, [])[:max_per_relation])
        return out

    except Exception as e:  # noqa: BLE001 — opener falls back gracefully
        logger.warning("_resolve_graph_related_products failed: %s", e)
        return []


async def _resolve_cart_recommendations(
    workspace_id: str,
    page_context: dict,
    *,
    max_recs: int = 3,
) -> list[dict]:
    """Pull cross-sell recommendations for the items currently in the cart.

    PRD-008-B Feature C2 (cart-idle): walks FBT edges across every cart
    line-item, aggregates co_count across overlapping recommendations
    (an item that pairs with multiple cart items scores higher), removes
    products already in the cart, returns the top ``max_recs`` by score.

    Empty list when no cart items, no graph, or any failure — caller
    treats that as "fall back to merchant's static greeting".
    """
    cart_items = (page_context or {}).get("cartItems") or []
    if not isinstance(cart_items, list) or not cart_items:
        return []

    # Cart items may arrive as [{handle, id, title, qty}, ...] OR as bare
    # handles/strings — be defensive about shape.
    seed_handles: set[str] = set()
    for item in cart_items:
        if isinstance(item, dict):
            h = item.get("handle") or item.get("product_handle")
            if h:
                seed_handles.add(str(h))
        elif isinstance(item, str):
            seed_handles.add(item)
    if not seed_handles:
        return []

    try:
        from modules.knowledge.graph_service import GraphifyService

        gs = GraphifyService()
        graph = await gs.load_graph(workspace_id)
        if graph is None:
            return []

        # Locate every cart-item node by handle. Skip silently if a handle
        # isn't in the graph (newly-added product, catalog out of sync).
        seed_node_ids: set = set()
        cart_node_ids: set = set()  # for exclusion from recs
        for node_id, attrs in graph.nodes(data=True):
            node_attrs = attrs.get("attrs") or {}
            if attrs.get("file_type") == "shopify_product":
                h = node_attrs.get("handle")
                if h and h in seed_handles:
                    seed_node_ids.add(node_id)
                    cart_node_ids.add(node_id)
        if not seed_node_ids:
            return []

        # Aggregate FBT recommendations across all seeds. Score = sum of
        # co_count across seeds it pairs with. A product paired with 2 of
        # 3 cart items scores higher than one paired with just 1.
        candidates: dict = {}  # node_id -> {label, score, total_orders, paired_with}
        for seed_id in seed_node_ids:
            for u, v, edata in graph.edges(seed_id, data=True):
                rel = (edata.get("relation") or "").lower()
                if rel != "frequently_bought_with":
                    continue
                other = v if u == seed_id else u
                if other in cart_node_ids:
                    continue  # don't recommend what's already in the cart
                other_attrs = graph.nodes[other]
                if other_attrs.get("file_type") != "shopify_product":
                    continue
                edge_attrs = edata.get("attrs") or {}
                co_count = edge_attrs.get("co_count") or 0
                entry = candidates.setdefault(other, {
                    "label": other_attrs.get("label") or other,
                    "handle": (other_attrs.get("attrs") or {}).get("handle"),
                    "score": 0,
                    "total_orders": edge_attrs.get("total_orders") or 0,
                    "paired_with_count": 0,
                })
                entry["score"] += co_count
                entry["paired_with_count"] += 1
                # Track the widest total_orders denominator seen (for citation).
                if edge_attrs.get("total_orders", 0) > entry["total_orders"]:
                    entry["total_orders"] = edge_attrs["total_orders"]

        if not candidates:
            return []

        # Rank: prefer items paired with the MOST cart items (cross-cutting
        # adds = strongest signal), break ties on aggregate score.
        ranked = sorted(
            candidates.values(),
            key=lambda c: (-c["paired_with_count"], -c["score"]),
        )
        return ranked[:max_recs]

    except Exception as e:  # noqa: BLE001
        logger.warning("_resolve_cart_recommendations failed: %s", e)
        return []


async def handle_widget_message(
    *,
    message: str,
    page_context: Optional[dict],
    trigger_reason: Optional[str],
    workspace_id: UUID,
    db: Session,
) -> WidgetPluginResult:
    """Shim — replicates the inline chat.py proactive-rewrite block.

    Behaviour mirrors ``api/widgets/chat.py`` byte-for-byte:

    * ``trigger_reason`` is ``"proactive_opener"`` or ``"cart_idle"``
      (the two members of chat.py's ``PROACTIVE_TRIGGER_REASONS``
      frozenset) AND ``page_context`` is not ``None`` → call the
      matching resolver + builder and return the rewritten directive
      as ``message``.
    * any other case (no trigger, unknown trigger, missing context) →
      return ``message`` unchanged. This includes mid-conversation
      messages: the Shopify vertical does NOT prepend an opaque
      ``(Context: ...)`` block today; that behaviour is owned by the
      generic plugin only.

    ``telemetry`` carries the same counts chat.py's current
    ``PROACTIVE_REWRITE`` log line captures so US-010 can rebuild that
    log line from the plugin result without losing observability.
    """
    if page_context is None or trigger_reason not in ("proactive_opener", "cart_idle"):
        return WidgetPluginResult(message=message)

    workspace_str = str(workspace_id)

    if trigger_reason == "cart_idle":
        from api.widgets.chat import _build_cart_idle_opener_message

        recommendations = await _resolve_cart_recommendations(
            workspace_str, page_context,
        )
        rewritten = _build_cart_idle_opener_message(
            page_context,
            recommendations=recommendations,
        )
        return WidgetPluginResult(
            message=rewritten,
            context_note="shopify shim: cart_idle rewrite",
            telemetry={
                "trigger_reason": trigger_reason,
                "related_count": len(recommendations),
            },
        )

    from api.widgets.chat import _build_proactive_opener_message

    related_products = await _resolve_graph_related_products(
        workspace_str, page_context,
    )
    rewritten = _build_proactive_opener_message(
        page_context,
        related_products=related_products,
    )
    return WidgetPluginResult(
        message=rewritten,
        context_note="shopify shim: proactive_opener rewrite",
        telemetry={
            "trigger_reason": trigger_reason,
            "related_count": len(related_products),
        },
    )
