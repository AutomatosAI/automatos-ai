"""Shopify page-context field mapping for proactive openers.

PRD-141 US-005 — the first ACTUAL move out of
``orchestrator/api/widgets/chat.py``. This module hosts the
Shopify-shaped page-context primitives that the proactive opener
directive builder closes over:

* :data:`_OPENER_CONTEXT_FIELDS` — ordered Shopify ``page_context`` →
  directive-label mapping. Order matters: it dictates the field order
  rendered into the proactive directive seen by the agent.
* :func:`_format_opener_context_value` — single-value formatter used
  by the directive builder to skip empties and quote string values
  that contain whitespace or embedded quotes.

These were extracted verbatim from chat.py with no value or behaviour
changes. ``_build_proactive_opener_message`` (lifted to
``integrations/shopify/widget_proactive.py`` in US-008) imports both
from here so its rendered output is byte-equal to the pre-lift
baseline captured in
``orchestrator/integrations/shopify/tests/fixtures/``.

Why this module exists at all: chat.py is the generic widget-chat
router; ``productHandle``, ``cartItems`` and other Shopify-shaped
keys must not appear in generic surfaces (PRD-141 §12). Hosting them
under ``integrations/shopify/`` lets the CI grep gate (US-012) enforce
that boundary.
"""

from __future__ import annotations

from typing import Optional


# Fields from page_context that the agent gets to ground openers on.
# Order matters — first match per group wins. Numeric/boolean fields are
# coerced to strings only when present + non-default.
_OPENER_CONTEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("pageType",          "page_type"),
    ("template",          "template"),
    # Product
    ("productTitle",      "product"),
    ("productType",       "product_type"),
    ("productVendor",     "vendor"),
    ("productPrice",      "price"),
    ("productAvailable",  "in_stock"),
    ("productHandle",     "product_handle"),
    # Collection (when on a collection page)
    ("collectionTitle",   "collection"),
    ("collectionHandle",  "collection_handle"),
    # Shop / locale
    ("shopDomain",        "shop"),
    ("shopCurrency",      "currency"),
    ("shopLocale",        "locale"),
    # Customer / cart
    ("customerId",        "logged_in_customer_id"),
    ("customerTags",      "customer_tags"),
    ("cartItemCount",     "cart_item_count"),
    ("cartTotalPrice",    "cart_total"),
)


def _format_opener_context_value(key: str, value) -> Optional[str]:
    """Render a single page-context value into the directive. Returns None
    if the value is empty/zero/false and shouldn't be sent to the agent."""
    if value is None or value == "" or value == 0 or value is False:
        return None
    if isinstance(value, str):
        return f'{key}="{value}"' if " " in value or '"' in value else f"{key}={value}"
    return f"{key}={value}"
