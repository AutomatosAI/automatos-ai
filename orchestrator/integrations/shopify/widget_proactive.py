"""Shopify vertical plugin — TEMPORARY shim delegating to chat.py.

PRD-141 US-003. Registered as ``PLUGIN_REGISTRY["shopify"]`` and used by
any workspace whose ``settings.vertical == "shopify"``.

This module is a **shim**, not a rewrite. It encapsulates the dispatch
contract — ``handle_widget_message`` matching the
:class:`integrations.WidgetPlugin` protocol — and delegates to the
existing inline Shopify helpers still living in
``orchestrator/api/widgets/chat.py``:

* ``_resolve_graph_related_products`` (product-page FBT / collection /
  vendor walk)
* ``_resolve_cart_recommendations`` (cart-idle multi-seed FBT walk)
* ``_build_proactive_opener_message`` (product-page directive builder)
* ``_build_cart_idle_opener_message`` (cart-idle directive builder)

US-006/007/008 will move those four helpers into this file. US-010
will delete the chat.py inline dispatch and route every widget chat
request through ``PLUGIN_REGISTRY``. At that point the imports below
become local definitions and this docstring's "shim" framing goes
away.

The four chat.py imports happen **inside** ``handle_widget_message``,
beneath the early-return gate. Two reasons:

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

from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from integrations import WidgetPluginResult


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
        from api.widgets.chat import (
            _build_cart_idle_opener_message,
            _resolve_cart_recommendations,
        )

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

    from api.widgets.chat import (
        _build_proactive_opener_message,
        _resolve_graph_related_products,
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
        context_note="shopify shim: proactive_opener rewrite",
        telemetry={
            "trigger_reason": trigger_reason,
            "related_count": len(related_products),
        },
    )
