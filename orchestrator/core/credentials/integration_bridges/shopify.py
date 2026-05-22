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
    """
    Repurposed: this single credential type now handles BOTH paths so the
    merchant only sees one form regardless of where they got their Shopify
    values from.

      - accessToken starts with 'shpat_'  → use it directly as an Admin API
        token (API_KEY flow). appSecretKey stays for webhook HMAC.
      - Otherwise (Partner App credentials from Partner Dashboard):
          accessToken    → Partner App Client ID
          appSecretKey   → Partner App Client Secret
        Bridge creates a per-workspace OAuth2 auth_config in Composio and
        returns redirect_url → frontend opens popup → merchant clicks Install
        on Shopify → ACTIVE shpat_ comes back automatically.
    """
    data = ctx.decrypted_data or {}
    subdomain = _normalise_subdomain(data.get("shopSubdomain", ""))
    primary = (data.get("accessToken") or "").strip()
    secondary = (data.get("appSecretKey") or "").strip()

    if not subdomain or not primary:
        return BridgeResult(
            status="bridge_error",
            error="shopSubdomain and the first credential field are required",
        )

    # Branch: shpat_ goes to API_KEY direct, everything else goes to OAuth
    # bounce treating (primary, secondary) as (clientId, clientSecret).
    if not primary.startswith("shpat_"):
        if not secondary:
            return BridgeResult(
                status="bridge_error",
                error=(
                    "Looks like Partner App credentials but no Client Secret was "
                    "provided. Paste Client Secret into the second field."
                ),
            )
        return _shopify_oauth_bounce(
            ctx,
            subdomain=subdomain,
            client_id=primary,
            client_secret=secondary,
        )

    # ---- shpat_ direct path (Custom App admin API token) ---------------
    token = primary

    entity_id = _resolve_entity_id(ctx.workspace_id)
    if not entity_id:
        return BridgeResult(status="bridge_error", error="No Composio entity for workspace")

    client = get_composio_client()
    composio = client.composio
    if not composio:
        return BridgeResult(status="bridge_error", error="Composio client not initialised (COMPOSIO_API_KEY missing)")

    # Sweep stale connections for this (entity, auth_config) before creating a
    # new one — Composio rejects "Multiple connected accounts" otherwise. We
    # delete anything not ACTIVE (EXPIRED, INITIATED, FAILED) since the user
    # is explicitly re-saving the credential to reconnect. If an ACTIVE one
    # already matches the token, just return success without creating another.
    try:
        existing = composio.connected_accounts.list(user_ids=[entity_id])
        for acct in getattr(existing, "items", []) or []:
            acct_auth = getattr(acct.auth_config, "id", None) if hasattr(acct, "auth_config") else None
            if acct_auth != SHARED_API_KEY_AUTH_CONFIG:
                continue
            acct_status = (getattr(acct, "status", "") or "").upper()
            if acct_status in ("EXPIRED", "INITIATED", "FAILED", "INACTIVE"):
                logger.info("[bridge:shopify] deleting stale %s connection %s", acct_status, acct.id)
                try:
                    composio.connected_accounts.delete(nanoid=acct.id)
                except Exception as del_err:
                    logger.warning("[bridge:shopify] couldn't delete %s: %s", acct.id, del_err)
            elif acct_status == "ACTIVE":
                # Already connected — return success with the existing id.
                logger.info("[bridge:shopify] reusing ACTIVE connection %s", acct.id)
                _persist_connection(
                    workspace_id=ctx.workspace_id,
                    app_name="SHOPIFY",
                    status="active",
                    connection_id=acct.id,
                    auth_config_id=SHARED_API_KEY_AUTH_CONFIG,
                    auth_scheme="API_KEY",
                    credential_id=ctx.credential_id,
                )
                return BridgeResult(
                    status="connected",
                    connection_id=acct.id,
                    auth_config_id=SHARED_API_KEY_AUTH_CONFIG,
                    auth_scheme="API_KEY",
                    extra={"shop": f"{subdomain}.myshopify.com", "reused": True},
                )
    except Exception as sweep_err:
        # Sweep is best-effort — if it fails, fall through to initiate and let
        # Composio's "multiple accounts" error reach the user with full context.
        logger.warning("[bridge:shopify] stale-connection sweep failed: %s", sweep_err)

    try:
        connection = composio.connected_accounts.initiate(
            entity_id,
            SHARED_API_KEY_AUTH_CONFIG,
            config={
                "authScheme": "API_KEY",
                "val": {"subdomain": subdomain, "generic_api_key": token},
            },
        )
    except Exception as e:
        # Surface Composio's actual error — it's the authority on what's wrong.
        logger.exception("[bridge:shopify] Composio rejected token")
        err_msg = str(e)
        # Strip generic SDK noise so the UI alert reads cleanly.
        for chunk in ("Composio API Error: ", "API Error: "):
            if chunk in err_msg:
                err_msg = err_msg.split(chunk, 1)[1]
        return BridgeResult(
            status="bridge_error",
            error=f"Composio rejected the credential: {err_msg}",
        )

    conn_id = getattr(connection, "id", None) or getattr(connection, "connectionId", None)
    conn_status = (getattr(connection, "status", "") or "").upper()

    # Composio doesn't validate tokens at create-time — it stores them and
    # marks the connection ACTIVE blindly. The token is only proven good when
    # a tool call actually goes through. Fire SHOPIFY_GET_SHOP_DETAILS now to
    # catch a wrong token (Storefront vs Admin, expired, etc.) so the merchant
    # gets the failure in the same UI alert instead of a silently-broken setup
    # they only discover when the widget can't list products.
    if conn_status == "ACTIVE":
        try:
            probe = composio.tools.execute(
                "SHOPIFY_GET_SHOP_DETAILS",
                user_id=entity_id,
                arguments={},
            )
            if not getattr(probe, "successful", True) and getattr(probe, "successful", None) is not None:
                err_text = getattr(probe, "error", None) or "Shopify rejected the request"
                logger.warning("[bridge:shopify] post-create probe failed: %s", err_text)
                # Clean up the dead connection so retries don't trip the sweep.
                try:
                    composio.connected_accounts.delete(nanoid=conn_id)
                except Exception:
                    pass
                hint = ""
                if "401" in str(err_text) or "Invalid API key" in str(err_text):
                    hint = (
                        " — Shopify says the token isn't valid for the Admin API. "
                        "Make sure you copied the 'Admin API access token' (shpat_…), "
                        "not the Storefront API token (shpss_…). Rotate the token by "
                        "toggling a scope in the Custom App if you can't see it anymore."
                    )
                return BridgeResult(
                    status="bridge_error",
                    error=f"Shopify rejected the credential{hint}",
                )
        except Exception as probe_err:
            # Probe is best-effort — if it raises, treat the create as success
            # but log so we can diagnose later.
            logger.warning("[bridge:shopify] post-create probe raised: %s", probe_err)

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

def _shopify_oauth_bounce(
    ctx: BridgeContext,
    subdomain: str,
    client_id: str,
    client_secret: str,
    scope: Optional[str] = None,
) -> BridgeResult:
    """
    Shared OAuth2 bounce: creates a per-workspace Composio auth_config with
    Partner App credentials, returns a hosted-install redirect_url. Called by
    BOTH the shopifyAccessTokenApi handler (when given Partner App creds
    instead of an shpat_ token) AND the shopifyOAuth2Api handler.
    """
    entity_id = _resolve_entity_id(ctx.workspace_id)
    if not entity_id:
        return BridgeResult(status="bridge_error", error="No Composio entity for workspace")

    client = get_composio_client()
    composio = client.composio
    if not composio:
        return BridgeResult(status="bridge_error", error="Composio client not initialised (COMPOSIO_API_KEY missing)")

    auth_config = composio.auth_configs.create(
        toolkit="SHOPIFY",
        options={
            "type": "use_custom_auth",
            "authScheme": "OAUTH2",
            "credentials": {
                "client_id": client_id,
                "client_secret": client_secret,
                "scopes": (scope or "").strip() or DEFAULT_OAUTH_SCOPES,
            },
            "name": f"Shopify OAuth (credential={ctx.credential_id})",
        },
    )
    auth_config_id = getattr(auth_config, "id", None)
    if not auth_config_id:
        return BridgeResult(status="bridge_error", error="Composio did not return an auth_config id")

    # Use .initiate() not .link() — initiate accepts a config dict carrying the
    # OAuth state (authScheme + val.shop), which is required for Shopify since
    # the shop subdomain is part of the OAuth authorize URL Composio builds.
    # allow_multiple=True so re-saves after a previously-INITIATED-but-never-
    # completed install don't trip "Multiple connected accounts" rejection.
    link = composio.connected_accounts.initiate(
        user_id=entity_id,
        auth_config_id=auth_config_id,
        config={"authScheme": "OAUTH2", "val": {"shop": subdomain}},
        allow_multiple=True,
    )

    redirect_url = getattr(link, "redirect_url", None) or getattr(link, "redirectUrl", None)
    conn_id = getattr(link, "id", None)

    _persist_connection(
        workspace_id=ctx.workspace_id,
        app_name="SHOPIFY",
        status="pending",
        connection_id=conn_id,
        auth_config_id=auth_config_id,
        auth_scheme="OAUTH2",
        credential_id=ctx.credential_id,
    )

    return BridgeResult(
        status="pending_oauth",
        connection_id=conn_id,
        auth_config_id=auth_config_id,
        auth_scheme="OAUTH2",
        oauth_redirect_url=redirect_url,
        extra={"shop": f"{subdomain}.myshopify.com"},
    )


@register("shopifyOAuth2Api")
def shopify_oauth2(ctx: BridgeContext) -> BridgeResult:
    data = ctx.decrypted_data or {}
    subdomain = _normalise_subdomain(data.get("shopSubdomain", ""))
    client_id = (data.get("clientId") or "").strip()
    client_secret = (data.get("clientSecret") or "").strip()
    scope = (data.get("scope") or "").strip()

    if not subdomain or not client_id or not client_secret:
        return BridgeResult(
            status="bridge_error",
            error="shopSubdomain, clientId and clientSecret are required",
        )

    return _shopify_oauth_bounce(
        ctx,
        subdomain=subdomain,
        client_id=client_id,
        client_secret=client_secret,
        scope=scope,
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
