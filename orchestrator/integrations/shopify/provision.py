"""Shopify vertical provisioner (PRD-183 S5, F076).

Moves the Shopify-specific provisioning *declarations* out of ``api/shopify.py``
and behind the generic :class:`integrations.provisioning.VerticalProvisioner`
interface. ``api/shopify.py`` keeps its thin internal-key-authed HTTP routes for
the existing Remix app, but the lifecycle KNOBS — the agent roster, the default
proactive widget config, the minted-key permissions, the ops-manager slug, the
site type, and the allowed-origins rule — now live here, so the generic
``POST /api/verticals/{v}/provision`` path can stand up a Shopify workspace
without any of that logic being Shopify-shaped in a generic file.

Importing ``integrations.shopify`` self-registers this provisioner into
``integrations.provisioning.PROVISIONER_REGISTRY`` and registers the
catalog/orders graph-source mappers into ``GRAPH_SOURCE_MAPPERS`` — the widget
plugin registration already happens the same way.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ── Roster ───────────────────────────────────────────────────────────
# The Shopify marketplace agent slugs seeded into a merchant workspace.
SHOPIFY_AGENT_SLUGS: List[str] = [
    "shopify-ops",
    "shopify-support",
    "shopify-product-expert",
    "shopify-merchandiser",
    "shopify-review-analyst",
    "shopify-gift-concierge",
    "shopify-seo-content",
    "shopify-business-analyst",
    "shopify-inventory-watchdog",
]


# ── Default proactive widget config (PRD-007) ────────────────────────
# Seeded into workspace.settings.widget_proactive at provision time. The
# merchant flips `enabled: true` from the dashboard to activate.
DEFAULT_WIDGET_PROACTIVE_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "page_types": ["product"],
    "triggers": [
        {"type": "time_on_page", "seconds": 20},
    ],
    "frequency_cap": {"scope": "product_session", "max_pops": 1},
    "greeting_source": "agent_with_canned_fallback",
    "canned_fallback": "Need a hand finding the right product?",
    "agent_timeout_ms": 1500,
    "popup_style": "corner_bubble",
    "respect_consent": True,
    "dismissal_persistence": "session",
}


class ShopifyProvisioner:
    """VerticalProvisioner implementation for the Shopify vertical."""

    vertical = "shopify"
    agent_slugs = SHOPIFY_AGENT_SLUGS
    ops_manager_slug = "shopify-ops"
    default_widget_config = DEFAULT_WIDGET_PROACTIVE_CONFIG
    key_permissions = ["chat", "documents:read", "agents:read", "agents:execute"]
    key_type = "public"
    site_type = "shopify"
    # The other Shopify routes (/connect, /deactivate, /sync, /events) resolve a
    # workspace by settings.shopify_domain, so the generic flow must stamp that
    # key too — not only the canonical source_external_id.
    external_id_key = "shopify_domain"
    # Rotate-on-reprovision is Shopify's documented key-recovery path
    # (runbooks/client-onboarding) — keep minting a fresh key every call.
    reuse_existing_key = False

    def allowed_domains(self, external_id: str, metadata: Dict[str, Any]) -> List[str]:
        """Origins permitted to use the minted widget key.

        Always allows the shop's ``*.myshopify.com`` domains. When the shop has
        a custom primary domain the storefront serves the widget from there, so
        allow that host plus its apex and sibling subdomains — otherwise the
        blog/chat widgets 403 on custom-domain stores.
        """
        shop = external_id
        domains = [f"https://{shop}", f"https://*.{shop}", "https://*.myshopify.com"]

        primary = (metadata or {}).get("domain")
        if primary:
            host = primary.split("://", 1)[-1].strip("/").split("/", 1)[0]
            if host and "myshopify.com" not in host:
                apex = host[4:] if host.startswith("www.") else host
                domains += [f"https://{host}", f"https://{apex}", f"https://*.{apex}"]

        seen: set[str] = set()
        return [d for d in domains if not (d in seen or seen.add(d))]

    def on_provisioned(self, db: Session, workspace: Any) -> None:
        """Ensure the workspace has a Site of type=shopify (cart-aware panels)."""
        try:
            from services.sites import ensure_shopify_site_for_workspace
            ensure_shopify_site_for_workspace(db, workspace)
        except Exception as e:  # noqa: BLE001 — never fail provision on heal error
            logger.warning(
                "ensure_shopify_site_for_workspace failed for workspace %s: %s",
                getattr(workspace, "id", "?"), e,
            )


# Module-level singleton — the registered provisioner instance.
provisioner = ShopifyProvisioner()


def register() -> None:
    """Self-register the Shopify provisioner + graph-source mappers.

    Called at package import time (``integrations/shopify/__init__.py``).
    """
    from integrations.provisioning import (
        PROVISIONER_REGISTRY,
        register_graph_source_mappers,
    )
    from modules.knowledge.graph_extraction import map_shopify_catalog, map_shopify_orders

    PROVISIONER_REGISTRY["shopify"] = provisioner
    register_graph_source_mappers(
        "shopify",
        {"catalog": map_shopify_catalog, "orders": map_shopify_orders},
    )
