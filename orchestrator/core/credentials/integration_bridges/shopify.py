"""
Shopify credential bridge — turns a saved n8n-style Credential into a working
Composio connected_account for the workspace.

Three credential types live in the credential_types table:
  - shopifyAccessTokenApi  → Shopify Custom App admin access token (shpat_*)
                             Maps to Composio API_KEY auth (instant ACTIVE).
  - shopifyOAuth2Api       → Shopify Partner App (client_id + client_secret)
                             Maps to per-workspace OAUTH2 auth config; returns
                             oauth_redirect_url for the frontend popup.
  - shopifyApi             → Legacy Shopify private-app (api key + password)
                             Composio dropped support — reported as unsupported.

The default OAuth scope list mirrors scripts/composio-setup.mjs (FULL_SCOPES)
so the Partner app must be configured with the same scopes; merchants who
need more can edit the credential's `scope` field.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from core.composio.client import get_composio_client
from core.composio.entity_manager import EntityManager
from core.database.database import SessionLocal

from . import register
from .base import BridgeContext, BridgeResult

logger = logging.getLogger(__name__)

# Shared API_KEY auth config (already exists in the Composio account — see
# scripts/composio-list-configs probe). Created via the Composio dashboard
# and reused for every merchant connecting via shpat_ token.
SHARED_API_KEY_AUTH_CONFIG = "ac_wwcaUIBEt9bX"

# Mirrors FULL_SCOPES in scripts/composio-setup.mjs — keep in lockstep with
# shopify.app.toml [access_scopes] when scopes change.
DEFAULT_OAUTH_SCOPES = ",".join([
    "read_products", "write_products",
    "read_orders", "write_orders", "read_all_orders",
    "read_customers", "write_customers",
    "read_inventory", "write_inventory",
    "read_content", "write_content",
    "read_discounts", "write_discounts",
    "read_price_rules", "write_price_rules",
    "read_fulfillments", "write_fulfillments",
    "read_gift_cards", "write_gift_cards",
    "read_draft_orders", "write_draft_orders",
    "read_shipping", "write_shipping",
    "read_analytics", "read_reports",
    "read_marketing_events", "write_marketing_events",
    "read_themes", "write_themes",
    "read_script_tags", "write_script_tags",
    "read_checkouts", "write_checkouts",
    "read_product_listings",
    "read_locations",
])


def _normalise_subdomain(raw: str) -> str:
    """Accept either 'innobuilduk' or 'innobuilduk.myshopify.com' — store the bare subdomain."""
    s = (raw or "").strip().lower()
    if s.endswith(".myshopify.com"):
        s = s[: -len(".myshopify.com")]
    return s


def _resolve_entity_id(workspace_id) -> Optional[str]:
    """Get-or-create the Composio entity for the workspace and return its composio_entity_id."""
    db = SessionLocal()
    try:
        em = EntityManager(db)
        entity = em.get_or_create_entity(workspace_id)
        return entity.get("composio_entity_id")
    finally:
        db.close()


def _persist_connection(
    workspace_id,
    app_name: str,
    status: str,
    connection_id: Optional[str],
    auth_config_id: Optional[str],
    auth_scheme: Optional[str],
    credential_id: int,
) -> None:
    """Mirror the bridge result into composio_connections.connection_metadata."""
    db = SessionLocal()
    try:
        em = EntityManager(db)
        entity = em.get_or_create_entity(workspace_id)
        em.add_connection(
            entity_id=entity["id"],
            app_name=app_name,
            status=status,
            connection_id=connection_id,
            metadata={
                "auth_config_id": auth_config_id,
                "auth_scheme": auth_scheme,
                "credential_id": credential_id,
                "source": "credential_bridge",
            },
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# shopifyAccessTokenApi — admin API token (instant ACTIVE)
# ---------------------------------------------------------------------------

@register("shopifyAccessTokenApi")
def shopify_access_token(ctx: BridgeContext) -> BridgeResult:
    data = ctx.decrypted_data or {}
    subdomain = _normalise_subdomain(data.get("shopSubdomain", ""))
    token = (data.get("accessToken") or "").strip()

    if not subdomain or not token:
        return BridgeResult(
            status="bridge_error",
            error="shopSubdomain and accessToken are required",
        )
    if not token.startswith("shpat_"):
        return BridgeResult(
            status="bridge_error",
            error="accessToken must be a Shopify admin API token (shpat_...). "
                  "Mint one at Shopify admin → Settings → Apps → Develop apps.",
        )

    entity_id = _resolve_entity_id(ctx.workspace_id)
    if not entity_id:
        return BridgeResult(status="bridge_error", error="No Composio entity for workspace")

    client = get_composio_client()
    composio = client.composio
    if not composio:
        return BridgeResult(status="bridge_error", error="Composio client not initialised (COMPOSIO_API_KEY missing)")

    connection = composio.connected_accounts.initiate(
        entity_id,
        SHARED_API_KEY_AUTH_CONFIG,
        config={
            "authScheme": "API_KEY",
            "val": {"subdomain": subdomain, "generic_api_key": token},
        },
    )

    conn_id = getattr(connection, "id", None) or getattr(connection, "connectionId", None)
    conn_status = (getattr(connection, "status", "") or "").upper()
    persisted_status = "active" if conn_status == "ACTIVE" else "pending"

    _persist_connection(
        workspace_id=ctx.workspace_id,
        app_name="SHOPIFY",
        status=persisted_status,
        connection_id=conn_id,
        auth_config_id=SHARED_API_KEY_AUTH_CONFIG,
        auth_scheme="API_KEY",
        credential_id=ctx.credential_id,
    )

    return BridgeResult(
        status="connected" if conn_status == "ACTIVE" else "pending",
        connection_id=conn_id,
        auth_config_id=SHARED_API_KEY_AUTH_CONFIG,
        auth_scheme="API_KEY",
        extra={"shop": f"{subdomain}.myshopify.com"},
    )


# ---------------------------------------------------------------------------
# shopifyOAuth2Api — per-workspace OAuth via merchant's own Partner app
# ---------------------------------------------------------------------------

@register("shopifyOAuth2Api")
def shopify_oauth2(ctx: BridgeContext) -> BridgeResult:
    data = ctx.decrypted_data or {}
    subdomain = _normalise_subdomain(data.get("shopSubdomain", ""))
    client_id = (data.get("clientId") or "").strip()
    client_secret = (data.get("clientSecret") or "").strip()
    scope = (data.get("scope") or "").strip() or DEFAULT_OAUTH_SCOPES

    if not subdomain or not client_id or not client_secret:
        return BridgeResult(
            status="bridge_error",
            error="shopSubdomain, clientId and clientSecret are required",
        )

    entity_id = _resolve_entity_id(ctx.workspace_id)
    if not entity_id:
        return BridgeResult(status="bridge_error", error="No Composio entity for workspace")

    client = get_composio_client()
    composio = client.composio
    if not composio:
        return BridgeResult(status="bridge_error", error="Composio client not initialised (COMPOSIO_API_KEY missing)")

    # Create a per-workspace OAuth2 auth config so each merchant owns their
    # own Partner-app credentials. Composio keeps these encrypted server-side.
    auth_config = composio.auth_configs.create(
        toolkit="SHOPIFY",
        options={
            "type": "use_custom_auth",
            "authScheme": "OAUTH2",
            "credentials": {
                "client_id": client_id,
                "client_secret": client_secret,
                "scopes": scope,
            },
            "name": f"Shopify OAuth (credential={ctx.credential_id})",
        },
    )
    auth_config_id = getattr(auth_config, "id", None)
    if not auth_config_id:
        return BridgeResult(status="bridge_error", error="Composio did not return an auth_config id")

    # Initiate the hosted OAuth bounce — Composio renders Shopify's install
    # page with the merchant's own credentials and stored scopes.
    link = composio.connected_accounts.link(
        user_id=entity_id,
        auth_config_id=auth_config_id,
        callback_url=None,
        config={"val": {"shop": subdomain}},
    )

    redirect_url = getattr(link, "redirect_url", None) or getattr(link, "redirectUrl", None)

    _persist_connection(
        workspace_id=ctx.workspace_id,
        app_name="SHOPIFY",
        status="pending",
        connection_id=getattr(link, "id", None),
        auth_config_id=auth_config_id,
        auth_scheme="OAUTH2",
        credential_id=ctx.credential_id,
    )

    return BridgeResult(
        status="pending_oauth",
        connection_id=getattr(link, "id", None),
        auth_config_id=auth_config_id,
        auth_scheme="OAUTH2",
        oauth_redirect_url=redirect_url,
        extra={"shop": f"{subdomain}.myshopify.com"},
    )


# ---------------------------------------------------------------------------
# shopifyApi — legacy private app (deprecated by Shopify, unsupported by Composio)
# ---------------------------------------------------------------------------

@register("shopifyApi")
def shopify_legacy(ctx: BridgeContext) -> BridgeResult:
    return BridgeResult(
        status="unsupported",
        error=(
            "Shopify legacy private apps (API key + password) were deprecated by Shopify "
            "and are not supported by Composio. Create a Shopify Custom App and use the "
            "'Shopify Access Token API' credential type instead."
        ),
    )
