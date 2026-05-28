"""Shopify vertical plugin — proactive opener + cart-idle directive builders.

PRD-141. Registered as ``PLUGIN_REGISTRY["shopify"]`` and used by any
workspace whose ``settings.vertical == "shopify"``.

The module encapsulates the dispatch contract — ``handle_widget_message``
matching the :class:`integrations.WidgetPlugin` protocol — together with
the four Shopify-specific helpers it needs:

* :func:`_resolve_graph_related_products` — single-seed FBT / collection /
  vendor traversal for the product-page opener.
* :func:`_resolve_cart_recommendations` — multi-seed FBT aggregation for
  the cart-idle nudge.
* :func:`_build_proactive_opener_message` — product-page directive
  builder, closes over ``_OPENER_CONTEXT_FIELDS`` and
  ``_format_opener_context_value`` from :mod:`.context_fields`.
* :func:`_build_cart_idle_opener_message` — cart-idle directive builder;
  no context-field dependency.

After PRD-141 US-010, ``orchestrator/api/widgets/chat.py`` calls this
module only through ``PLUGIN_REGISTRY["shopify"].handle_widget_message``
— there are no direct imports of the underlying helpers from outside
this package, and chat.py contains zero Shopify identifiers.

The two proactive trigger strings (``proactive_opener``, ``cart_idle``)
are hardcoded inline below for the gate check. They mirror the
``PROACTIVE_TRIGGER_REASONS`` frozenset in chat.py, which controls the
generic LLM-call shape (text-only, no composio) and is deliberately
kept on the generic side: a barbershop opener would use the same
``proactive_opener`` value to flip the agent into opener mode.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from integrations import WidgetPluginResult
from integrations.shopify.context_fields import (
    _OPENER_CONTEXT_FIELDS,
    _format_opener_context_value,
)

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


def _build_proactive_opener_message(
    page_context: dict,
    related_products: Optional[list] = None,
) -> str:
    """Synthesize the user-side message for a proactive opener request.

    The widget never sends real user text for proactive openers — instead
    we synthesize a directive carrying the FULL page context the agent
    needs to ground a contextual one-line opener.

    PRD-007 v0.4: previously only pageType + productTitle/Type leaked into
    the directive; agents had to make up everything else (price, vendor,
    availability). Now every populated page_context field is forwarded so
    the agent can lean on real facts before reaching for a tool call.

    PRD-007 addendum (post PRD-009 Layer 2): when ``related_products`` is
    supplied — top FBT pair, top in-collection sibling, top vendor sibling
    — they're appended as facts the agent can weave into the opener. The
    agent is instructed to lead with FBT signal when present (real
    customer co-purchase pattern, highest leverage) and fall back to the
    catalog siblings otherwise. Always real data; never invented.
    """
    parts: list[str] = []
    for src_key, label in _OPENER_CONTEXT_FIELDS:
        rendered = _format_opener_context_value(label, page_context.get(src_key))
        if rendered is not None:
            parts.append(rendered)
    summary = ", ".join(parts) if parts else "no context"

    related_block = ""
    if related_products:
        # Render each related product as a one-line fact with provenance,
        # so the agent can cite naturally (e.g. "often bought with X — 12 of
        # 57 orders"). Order matters: FBT first (strongest signal), then
        # collection / vendor siblings as fall-backs.
        rel_order = {
            "frequently_bought_with": 0,
            "in_collection": 1,
            "by_vendor": 2,
        }
        sorted_rel = sorted(
            related_products,
            key=lambda p: rel_order.get(p.get("relation", ""), 99),
        )
        rendered_rel = []
        for p in sorted_rel:
            label = p.get("label", "?")
            rel = p.get("relation", "")
            if rel == "frequently_bought_with" and p.get("co_count"):
                rendered_rel.append(
                    f'"{label}" (bought together in {p["co_count"]} '
                    f'of {p.get("total_orders", "?")} orders)'
                )
            elif rel == "in_collection":
                rendered_rel.append(f'"{label}" (same collection)')
            elif rel == "by_vendor":
                rendered_rel.append(f'"{label}" (same vendor)')
            else:
                rendered_rel.append(f'"{label}"')
        related_block = (
            " Related from order/catalog graph (use these naturally — "
            "prefer the order-pair signal when present, else mention the "
            "collection/vendor sibling as a starter for conversation): "
            + "; ".join(rendered_rel)
        )

    return (
        "[PROACTIVE_OPENER] Generate a contextual one-sentence opener "
        "(≤140 chars). RETURN PLAIN TEXT ONLY — no tool calls, no JSON, "
        "no markdown, no greetings. Use the facts below as your source of "
        "truth — do NOT invent specs, compatibility, or pricing the context "
        "doesn't include. If a fact you'd want isn't here, ask a question "
        "instead of fabricating. "
        f"Context: {summary}.{related_block}"
    )


def _build_cart_idle_opener_message(
    page_context: dict,
    recommendations: Optional[list] = None,
) -> str:
    """Synthesize the directive for a cart-idle proactive popup.

    PRD-008-B Feature C2: a shopper has been idle on the cart page for
    `idle_seconds`. We want a graph-grounded nudge that references real
    FBT pairings — "customers who bought your stuff also added X" — not
    a generic "still there?" line.

    When ``recommendations`` is empty (no graph, cold start, or no FBT
    signal for cart items), the directive still produces a contextual
    nudge based on cart size/total — never fabricates products.
    """
    cart_count = page_context.get("cartItemCount") or 0
    cart_total = page_context.get("cartTotalPrice")
    currency = page_context.get("shopCurrency") or ""

    cart_summary_parts: list[str] = []
    if cart_count:
        cart_summary_parts.append(f"cart_item_count={cart_count}")
    if cart_total:
        # Shopify amounts are minor units (cents/pence). Render as e.g. "362.18 GBP".
        try:
            major = float(cart_total) / 100.0
            cart_summary_parts.append(f"cart_total={major:.2f} {currency}".strip())
        except (TypeError, ValueError):
            cart_summary_parts.append(f"cart_total={cart_total}")
    cart_summary = ", ".join(cart_summary_parts) if cart_summary_parts else "cart_idle"

    rec_block = ""
    if recommendations:
        rendered = []
        for r in recommendations:
            label = r.get("label", "?")
            paired = r.get("paired_with_count", 0)
            if paired > 1:
                rendered.append(
                    f'"{label}" (bought with {paired} of the items in this cart)'
                )
            elif r.get("score") and r.get("total_orders"):
                rendered.append(
                    f'"{label}" (added together in {r["score"]} '
                    f'of {r["total_orders"]} orders)'
                )
            else:
                rendered.append(f'"{label}"')
        rec_block = (
            " Frequently bought with what's in this cart (real order-graph "
            "data — pick ONE to mention, prefer the one paired with multiple "
            "cart items): "
            + "; ".join(rendered)
        )

    return (
        "[PROACTIVE_OPENER] [CART_IDLE] The shopper has been idle on the cart "
        "page. Generate a single helpful sentence (≤140 chars) that nudges "
        "them toward checkout OR offers a relevant add-on. RETURN PLAIN TEXT "
        "ONLY — no tool calls, no markdown, no greetings. Do NOT invent "
        "products that aren't named below. If no recommendation is provided, "
        "ask if they need help finishing their order — don't fabricate. "
        f"Context: {cart_summary}.{rec_block}"
    )


async def handle_widget_message(
    *,
    message: str,
    page_context: Optional[dict],
    trigger_reason: Optional[str],
    workspace_id: UUID,
    db: Session,
) -> WidgetPluginResult:
    """Build the Shopify proactive opener / cart-idle directive.

    Behaviour:

    * ``trigger_reason`` is ``"proactive_opener"`` or ``"cart_idle"``
      (mirrors chat.py's ``PROACTIVE_TRIGGER_REASONS`` frozenset) AND
      ``page_context`` is not ``None`` → call the matching resolver +
      builder and return the rewritten directive as ``message``.
    * any other case (no trigger, unknown trigger, missing context) →
      return ``message`` unchanged. This includes mid-conversation
      messages: the Shopify vertical does NOT prepend an opaque
      ``(Context: ...)`` block — that behaviour is owned by the
      generic plugin only.

    ``telemetry`` carries the counts chat.py's ``PROACTIVE_REWRITE``
    log line surfaces (``trigger_reason``, ``related_count``) so the
    dispatcher rebuilds the log line from the plugin result without
    losing observability.
    """
    if page_context is None or trigger_reason not in ("proactive_opener", "cart_idle"):
        return WidgetPluginResult(message=message)

    workspace_str = str(workspace_id)

    if trigger_reason == "cart_idle":
        recommendations = await _resolve_cart_recommendations(
            workspace_str, page_context,
        )
        rewritten = _build_cart_idle_opener_message(
            page_context,
            recommendations=recommendations,
        )
        return WidgetPluginResult(
            message=rewritten,
            context_note="shopify: cart_idle rewrite",
            telemetry={
                "trigger_reason": trigger_reason,
                "related_count": len(recommendations),
            },
        )

    related_products = await _resolve_graph_related_products(
        workspace_str, page_context,
    )
    rewritten = _build_proactive_opener_message(
        page_context,
        related_products=related_products,
    )
    return WidgetPluginResult(
        message=rewritten,
        context_note="shopify: proactive_opener rewrite",
        telemetry={
            "trigger_reason": trigger_reason,
            "related_count": len(related_products),
        },
    )
