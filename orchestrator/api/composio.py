"""
Composio API Endpoints (PRD-36)
==============================

API routes for Composio integration:
- App discovery and listing
- OAuth connection management
- Action listing
- Webhook handling for triggers
"""

import asyncio
import hmac
import hashlib
import logging
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Header
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from core.database.database import get_db, get_db_session
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.auth.dependencies import RequestContext
from core.composio.client import get_composio_client, ComposioClient
from core.composio.entity_manager import EntityManager
from core.composio.tool_executor import ComposioToolExecutor
from core.models.composio import AgentAppFeature, ComposioConnection, ComposioEntity, TriggerSubscription
from core.models.core import WorkflowTemplate as WorkflowRecipe, RecipeExecution
from core.models.routing import UnroutedEvent
from core.routing.cache import get_routing_cache
from core.routing.engine import UniversalRouter
from core.routing.ingestors.jira_trigger import JiraTriggerIngestor
from services import webhook_dedup
from config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/composio", tags=["Composio"])


# =============================================================================
# Request/Response Models
# =============================================================================

class AppResponse(BaseModel):
    """Composio app information."""
    name: str
    display_name: str
    description: Optional[str] = None
    logo_url: Optional[str] = None
    categories: List[str] = []
    is_connected: bool = False
    action_count: int = 0
    auth_schemes: List[str] = []
    triggers: List[Dict[str, Any]] = []
    trigger_count: int = 0


class ActionResponse(BaseModel):
    """Composio action information."""
    name: str
    display_name: str
    description: Optional[str] = None
    app_name: str
    parameters: dict = {}
    enabled: bool = True


class ConnectionResponse(BaseModel):
    """App connection status."""
    app_name: str
    status: str
    connected_at: Optional[datetime] = None
    connection_id: Optional[str] = None


class InitiateConnectionRequest(BaseModel):
    """Request to initiate a hosted-auth connection."""
    callback_url: Optional[str] = None
    # Optional per-workspace overrides for hosted-auth scheme selection.
    # Pass auth_scheme="API_KEY" for Shopify merchants where managed-install
    # breaks the OAuth bounce; auth_config_id pins a specific Composio config.
    auth_scheme: Optional[str] = None
    auth_config_id: Optional[str] = None


class InitiateConnectionResponse(BaseModel):
    """Hosted-auth redirect URL + the chosen auth config (persisted on the workspace)."""
    redirect_url: str
    app_name: str
    auth_config_id: Optional[str] = None
    auth_scheme: Optional[str] = None


class FeatureToggleRequest(BaseModel):
    """Toggle action enabled state."""
    action_name: str
    enabled: bool


class BatchFeatureToggleRequest(BaseModel):
    """Batch toggle actions."""
    features: List[FeatureToggleRequest]


class TriggerSubscriptionRequest(BaseModel):
    """Subscribe to a trigger."""
    trigger_name: str
    agent_id: Optional[int] = None
    workflow_id: Optional[int] = None


class WebhookPayload(BaseModel):
    """Incoming webhook data."""
    trigger_name: str
    entity_id: str
    app_name: str
    event_data: dict = {}


# =============================================================================
# App Discovery Endpoints
# =============================================================================

@router.get("/apps", response_model=List[AppResponse])
async def list_available_apps(
    category: Optional[str] = Query(None, description="Filter by category"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    List all available Composio apps.
    
    Returns apps with connection status for the current workspace.
    """
    client = get_composio_client()
    
    try:
        apps = client.get_available_apps()
    except Exception as e:
        logger.error(f"Failed to fetch Composio apps: {e}")
        raise HTTPException(status_code=503, detail="Composio service unavailable")
    
    # Get connected apps for this workspace
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    connected_apps = set()
    
    if entity:
        connections = entity_manager.get_entity_connections(entity["id"])
        connected_apps = {
            c["app_name"] for c in connections 
            if c["status"] == "active"
        }
    
    # Filter by category if provided
    if category:
        apps = [a for a in apps if category.lower() in [c.lower() for c in a.get("categories", [])]]
    
    return [
        AppResponse(
            name=app["name"],
            display_name=app.get("display_name", app["name"]),
            description=app.get("description"),
            logo_url=app.get("logo_url"),
            categories=app.get("categories", []),
            is_connected=app["name"].upper() in {a.upper() for a in connected_apps},
            # Never fetch actions per-app in list endpoint (expensive).
            action_count=app.get("action_count", 0),
            auth_schemes=app.get("auth_schemes", []),
            triggers=app.get("triggers", []),
            trigger_count=app.get("trigger_count", 0)
        )
        for app in apps
    ]


@router.get("/apps/{app_name}/actions", response_model=List[ActionResponse])
async def list_app_actions(
    app_name: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    List all actions available for an app.
    
    Includes enabled status for each action based on agent configuration.
    """
    client = get_composio_client()
    
    try:
        actions = client.get_app_actions(app_name.upper())
    except Exception as e:
        logger.error(f"Failed to fetch actions for {app_name}: {e}")
        raise HTTPException(status_code=503, detail="Composio service unavailable")
    
    return [
        ActionResponse(
            name=action["name"],
            display_name=action.get("display_name", action["name"]),
            description=action.get("description"),
            app_name=app_name.upper(),
            parameters=action.get("parameters", {})
        )
        for action in actions
    ]


# =============================================================================
# Connection Management Endpoints
# =============================================================================

@router.get("/connections", response_model=List[ConnectionResponse])
async def list_connections(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    List all app connections for the current workspace.
    Syncs status with Composio API for pending connections.
    """
    client = get_composio_client()
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    
    if not entity:
        return []
    
    all_connections = entity_manager.get_entity_connections(entity["id"])

    # Include all workspace connections: active, pending OAuth, and disconnected (added).
    # 'added' connections are shown so users can reconnect after disconnecting auth.
    connections = [c for c in all_connections if c.get("status") in ("active", "pending", "added")]

    result = []

    for conn in connections:
        status = conn["status"]

        # For pending connections, check Composio API for actual status
        if status == "pending":
            try:
                composio_status = client.get_connection_status(
                    entity_id=entity["composio_entity_id"],
                    app=conn["app_name"]
                )
                if composio_status and composio_status.get("status") in ("ACTIVE", "INITIATED"):
                    # Connection completed on Composio side — upgrade to active
                    entity_manager.update_connection_status(
                        entity_id=entity["id"],
                        app_name=conn["app_name"],
                        status="active",
                        connection_id=composio_status.get("id")
                    )
                    status = "active"
                    logger.info(f"Connection {conn['app_name']} synced to active (was {composio_status.get('status')})")
            except Exception as e:
                logger.warning(f"Failed to sync pending connection {conn['app_name']}: {e}")
        
        result.append(ConnectionResponse(
            app_name=conn["app_name"],
            status=status,
            connected_at=conn.get("connected_at"),
            connection_id=conn.get("connection_id")
        ))
    
    return result


@router.get("/linkedin/test-upload-init")
async def test_linkedin_upload_init():
    """Smoke test: call LinkedIn initializeUpload directly (no Composio).
    Verifies credential store + OAuth token + API version work."""
    import httpx as _httpx
    from core.composio.linkedin_image_workaround import (
        _get_access_token, _initialize_image_upload, _load_linkedin_credentials,
    )

    try:
        creds = _load_linkedin_credentials()
        org_urn = creds.get("organization_urn", "")
        async with _httpx.AsyncClient(timeout=15) as http:
            token = await _get_access_token(http)
            upload_url, image_urn = await _initialize_image_upload(http, token, org_urn)
            return {
                "ok": bool(upload_url and image_urn),
                "upload_url": (upload_url or "")[:80] + "..." if upload_url else None,
                "image_urn": image_urn,
                "org_urn": org_urn,
                "credential_found": True,
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}




def _record_first_integration_if_first(
    db: Session, entity_manager: EntityManager, workspace_id: Any
) -> None:
    """PRD-222 US-019: emit ``first_integration_connected`` on the 0 → 1 crossing.

    The trigger is the workspace's ACTIVE connection count reaching exactly 1 —
    the 0 → 1 transition — read from the same connection store the Tools page
    uses, NOT the OAuth popup closing. The stamp itself is idempotent (once per
    workspace, ever, via ``record_first_integration_connected``), so a reconnect
    after a full disconnect never re-fires. Best effort: a funnel miss must never
    break a real connection, so any failure is logged and swallowed.
    """
    try:
        active = entity_manager.get_connected_apps(workspace_id)
        if len(active) != 1:
            return
        from core.models.workspaces import Workspace
        from services.onboarding_state import record_first_integration_connected

        ws = db.query(Workspace).get(workspace_id)
        if ws is not None:
            record_first_integration_connected(db, ws)
    except Exception:
        logger.warning(
            "[FUNNEL] first_integration_connected check failed for workspace %s",
            workspace_id, exc_info=True,
        )


@router.post("/connect/{app_name}", response_model=InitiateConnectionResponse, dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def initiate_connection(
    app_name: str,
    request: InitiateConnectionRequest = None,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Initiate OAuth connection for an app.
    
    Returns a redirect URL for the Composio hosted OAuth flow.
    """
    client = get_composio_client()
    entity_manager = EntityManager(db)
    
    # Get or create entity for workspace
    entity = entity_manager.get_or_create_entity(ctx.workspace_id)
    composio_entity_id = entity["composio_entity_id"]
    
    # NO_AUTH apps (e.g. composio_search) don't need OAuth — activate immediately
    if client.is_no_auth_app(app_name):
        entity_manager.add_connection(
            entity_id=entity["id"],
            app_name=app_name.upper(),
            status="active",
        )
        logger.info(f"[CONNECT] {app_name.upper()} is NO_AUTH — activated immediately")
        _record_first_integration_if_first(db, entity_manager, ctx.workspace_id)
        return InitiateConnectionResponse(
            redirect_url="",
            app_name=app_name.upper(),
        )

    # Default callback URL
    callback_url = request.callback_url if request else None
    if not callback_url:
        # Use frontend callback URL (popup callback page)
        frontend_url = config.FRONTEND_URL or "http://localhost:3000"
        callback_url = f"{frontend_url}/tools/callback?connected={app_name.upper()}"

    # If the workspace already has a stored auth_config_id/scheme for this
    # app (from a prior connect), reuse it unless the caller is explicitly
    # overriding. This makes reconnects sticky to the scheme that worked.
    explicit_scheme = request.auth_scheme if request else None
    explicit_id = request.auth_config_id if request else None
    if not explicit_scheme and not explicit_id:
        prior_meta = entity_manager.get_connection_metadata(
            entity_id=entity["id"], app_name=app_name.upper()
        )
        if prior_meta:
            explicit_scheme = prior_meta.get("auth_scheme") or explicit_scheme
            explicit_id = prior_meta.get("auth_config_id") or explicit_id

    try:
        link = client.initiate_connection(
            entity_id=composio_entity_id,
            app=app_name.upper(),
            callback_url=callback_url,
            auth_config_id=explicit_id,
            auth_scheme=explicit_scheme,
        )
    except Exception as e:
        logger.error(f"Failed to initiate connection for {app_name}: {e}")
        raise HTTPException(status_code=503, detail="Failed to initiate OAuth connection")

    # Store pending connection + pin the chosen auth_config_id / scheme so
    # subsequent reconnects pick the same one without the caller specifying.
    entity_manager.add_connection(
        entity_id=entity["id"],
        app_name=app_name.upper(),
        status="pending",
        metadata={
            "auth_config_id": link.get("auth_config_id"),
            "auth_scheme": link.get("auth_scheme"),
        },
    )

    return InitiateConnectionResponse(
        redirect_url=link["redirect_url"],
        app_name=app_name.upper(),
        auth_config_id=link.get("auth_config_id"),
        auth_scheme=link.get("auth_scheme"),
    )


@router.post("/connect/{app_name}/callback", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def connection_callback(
    app_name: str,
    connection_id: Optional[str] = Query(None, description="Composio connection ID"),
    status: str = Query("active", description="Connection status"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    OAuth callback handler.

    Called after user completes OAuth flow to update connection status.
    The connection_id is optional — if not provided by the OAuth redirect,
    we attempt to resolve it from the Composio API.
    """
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)

    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    resolved_connection_id = connection_id

    # If no connection_id from OAuth redirect, try to resolve from Composio API
    if not resolved_connection_id:
        try:
            client = get_composio_client()
            composio_status = client.get_connection_status(
                entity_id=entity["composio_entity_id"],
                app=app_name.upper()
            )
            if composio_status:
                resolved_connection_id = composio_status.get("id")
                logger.info(f"[CALLBACK] Resolved connection_id from Composio API: {resolved_connection_id}")
        except Exception as e:
            logger.warning(f"[CALLBACK] Could not resolve connection_id from Composio: {e}")

    # Fallback: scan all entity connections for this app (covers API_KEY apps
    # where get_connection_status fails due to missing auth_config_id cache)
    if not resolved_connection_id:
        try:
            connections = entity_manager.get_entity_connections(str(entity["id"]))
            for conn in connections:
                if (conn.get("app_name") or "").upper() == app_name.upper():
                    resolved_connection_id = conn.get("connection_id")
                    if resolved_connection_id:
                        logger.info(f"[CALLBACK] Resolved connection_id from entity connections: {resolved_connection_id}")
                    break
        except Exception as e:
            logger.warning(f"[CALLBACK] Entity connections fallback failed: {e}")

    # Normalize status — treat "success" as "active"
    normalized_status = "active" if status in ("success", "active") else status

    # Use add_connection (upsert) — creates if missing, updates if exists.
    # This is safer than update_connection_status which returns False if no row.
    entity_manager.add_connection(
        entity_id=str(entity["id"]),
        app_name=app_name.upper(),
        status=normalized_status,
        connection_id=resolved_connection_id,
    )

    logger.info(f"[CALLBACK] {app_name.upper()} connection updated to {normalized_status} (connection_id={resolved_connection_id})")

    # PRD-222 US-019: the first integration a workspace connects is a first-class
    # funnel event — fire it here (the real OAuth completion), off the 0 → 1
    # active-connection crossing, not the popup closing.
    if normalized_status == "active":
        _record_first_integration_if_first(db, entity_manager, ctx.workspace_id)

    return {"status": "success", "app_name": app_name.upper(), "connected": normalized_status == "active"}


@router.delete("/connections/{app_name}", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def disconnect_app(
    app_name: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Disconnect OAuth for an app (keeps app in workspace with status='added').
    This allows the app to stay in the workspace but disconnects the OAuth authorization.
    Use remove-from-workspace endpoint to completely remove the app.
    """
    client = get_composio_client()
    entity_manager = EntityManager(db)

    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    if not entity:
        raise HTTPException(status_code=404, detail="No connections found")

    # Disconnect from Composio (revoke OAuth)
    try:
        client.disconnect_app(entity["composio_entity_id"], app_name.upper())
        logger.info(f"Disconnected {app_name} OAuth from Composio")
    except Exception as e:
        logger.warning(f"Failed to disconnect from Composio (may already be disconnected): {e}")

    # Update status to 'added' instead of removing - keeps app in workspace but disconnected
    entity_manager.update_connection_status(
        entity_id=entity["id"],
        app_name=app_name.upper(),
        status="added",
        connection_id=None
    )

    logger.info(f"[DISCONNECT] {app_name} OAuth disconnected but kept in workspace")

    return {"status": "disconnected", "app_name": app_name.upper()}


# =============================================================================
# Agent Feature Management
# =============================================================================

@router.get("/agents/{agent_id}/apps/{app_name}/features", response_model=List[ActionResponse])
async def get_agent_app_features(
    agent_id: int,
    app_name: str,
    db: Session = Depends(get_db)
):
    """
    Get enabled features for an agent-app combination.
    """
    client = get_composio_client()
    executor = ComposioToolExecutor(db)
    
    # Get all actions for the app
    actions = client.get_app_actions(app_name.upper())
    
    # Get enabled actions for this agent
    enabled_actions = set(executor.get_agent_enabled_actions(agent_id, app_name.upper()))
    
    # If no features configured yet, all are enabled by default
    has_config = db.query(AgentAppFeature).filter(
        AgentAppFeature.agent_id == agent_id,
        AgentAppFeature.app_name == app_name.upper()
    ).count() > 0
    
    return [
        ActionResponse(
            name=action["name"],
            display_name=action.get("display_name", action["name"]),
            description=action.get("description"),
            app_name=app_name.upper(),
            parameters=action.get("parameters", {}),
            enabled=action["name"] in enabled_actions if has_config else True
        )
        for action in actions
    ]


@router.put("/agents/{agent_id}/apps/{app_name}/features", dependencies=[Depends(require_workspace_permission("agents:update"))])
async def update_agent_app_features(
    agent_id: int,
    app_name: str,
    request: BatchFeatureToggleRequest,
    db: Session = Depends(get_db)
):
    """
    Update enabled features for an agent-app combination.
    """
    executor = ComposioToolExecutor(db)
    
    updates = [
        {"action_name": f.action_name, "enabled": f.enabled}
        for f in request.features
    ]
    
    count = executor.batch_set_agent_actions(agent_id, app_name.upper(), updates)
    
    return {"updated": count, "app_name": app_name.upper()}


@router.post("/agents/{agent_id}/apps/{app_name}/enable-all", dependencies=[Depends(require_workspace_permission("agents:update"))])
async def enable_all_features(
    agent_id: int,
    app_name: str,
    db: Session = Depends(get_db)
):
    """
    Enable all features for an agent-app combination.
    """
    client = get_composio_client()
    executor = ComposioToolExecutor(db)
    
    actions = client.get_app_actions(app_name.upper())
    updates = [{"action_name": a["name"], "enabled": True} for a in actions]
    
    count = executor.batch_set_agent_actions(agent_id, app_name.upper(), updates)
    
    return {"enabled": count, "app_name": app_name.upper()}


@router.post("/agents/{agent_id}/apps/{app_name}/disable-all", dependencies=[Depends(require_workspace_permission("agents:update"))])
async def disable_all_features(
    agent_id: int,
    app_name: str,
    db: Session = Depends(get_db)
):
    """
    Disable all features for an agent-app combination.
    """
    client = get_composio_client()
    executor = ComposioToolExecutor(db)
    
    actions = client.get_app_actions(app_name.upper())
    updates = [{"action_name": a["name"], "enabled": False} for a in actions]
    
    count = executor.batch_set_agent_actions(agent_id, app_name.upper(), updates)
    
    return {"disabled": count, "app_name": app_name.upper()}


# =============================================================================
# Webhook Handler
# =============================================================================

@router.post("/webhook")
async def handle_webhook(
    request: Request,
    x_composio_signature: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Handle incoming Composio webhook events.

    Supports both V2 (legacy) and V3 payload formats:
      V2: {"trigger_name": ..., "entity_id": ..., "event_data": {...}}
      V3: {"type": ..., "data": {...}, "timestamp": ..., "log_id": ...}
           + composio.trigger.message wrapper
    """
    # Get raw body for signature verification
    body = await request.body()

    # Verify signature — supports both legacy (x-composio-signature)
    # and V3 (webhook-signature with "v1,<base64>" format).
    # When a secret is configured, a valid signature is mandatory:
    # mismatch, verification error, or missing header ⇒ 401 (P2-13).
    webhook_secret = config.COMPOSIO_WEBHOOK_SECRET
    v3_signature = request.headers.get("webhook-signature")
    if webhook_secret and v3_signature:
        # V3 format: "v1,<base64_signature>"
        try:
            import base64
            webhook_id = request.headers.get("webhook-id", "")
            webhook_ts = request.headers.get("webhook-timestamp", "")
            signed_content = f"{webhook_id}.{webhook_ts}.".encode() + body
            expected_sig = base64.b64encode(
                hmac.new(base64.b64decode(webhook_secret), signed_content, hashlib.sha256).digest()
            ).decode()
            sig_value = v3_signature.split(",", 1)[-1] if "," in v3_signature else v3_signature
            signature_ok = hmac.compare_digest(sig_value, expected_sig)
        except Exception:
            logger.warning("V3 signature verification error — rejecting", exc_info=True)
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        if not signature_ok:
            logger.warning("V3 webhook signature mismatch — rejecting")
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
    elif webhook_secret and x_composio_signature:
        # Legacy format: plain HMAC hex digest
        expected_sig = hmac.new(
            webhook_secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(x_composio_signature, expected_sig):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
    elif webhook_secret:
        logger.warning("Webhook rejected: COMPOSIO_WEBHOOK_SECRET is set but no signature header present")
        raise HTTPException(status_code=401, detail="Missing webhook signature")

    # Replay guard + event dedup (PRD-194 S2, P2-13). Composio V3 sends
    # webhook-id + webhook-timestamp on every delivery, and both are part of
    # the signed content above — so a valid-signature replay of an old
    # capture still carries its ORIGINAL timestamp (⇒ stale ⇒ 401), and a
    # redelivery of an already-accepted webhook-id is a fast no-op ack:
    # nothing parsed, nothing routed, nothing dispatched.
    if webhook_dedup.timestamp_is_stale(request.headers.get("webhook-timestamp")):
        logger.warning("Webhook rejected: stale webhook-timestamp (replay guard)")
        raise HTTPException(status_code=401, detail="Stale webhook timestamp")
    dedup_event_id = request.headers.get("webhook-id")
    if await webhook_dedup.seen_before("composio", dedup_event_id):
        logger.info("Duplicate Composio webhook %s — no-op ack", dedup_event_id)
        return {"status": "duplicate", "webhook_id": dedup_event_id}

    # Parse payload
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    logger.info("Webhook received — raw keys: %s", list(payload.keys()))

    # Normalise V3 payload format.
    # Real V3 structure: {id, timestamp, type: "composio.trigger.message",
    #   metadata: {log_id, trigger_slug, trigger_id, connected_account_id, auth_config_id, user_id},
    #   data: {actual event fields like issue_key, summary, etc.}}
    if "type" in payload and "data" in payload:
        event_type = payload.get("type", "")
        inner = payload.get("data", {})
        meta = payload.get("metadata", {})

        logger.info("V3 webhook: event_type=%s metadata_keys=%s data_keys=%s connected_account=%s trigger_slug=%s",
                     event_type, list(meta.keys()) if isinstance(meta, dict) else "N/A",
                     list(inner.keys()) if isinstance(inner, dict) else "N/A",
                     meta.get("connected_account_id", "N/A") if isinstance(meta, dict) else "N/A",
                     meta.get("trigger_slug", "N/A") if isinstance(meta, dict) else "N/A")

        # Trigger name: check metadata.trigger_slug (real V3), then camelCase variants, then data, then event_type
        trigger_name = (
            (meta.get("trigger_slug") if isinstance(meta, dict) else None)
            or (meta.get("triggerName") if isinstance(meta, dict) else None)
            or (inner.get("trigger_name") if isinstance(inner, dict) else None)
            or (inner.get("triggerName") if isinstance(inner, dict) else None)
            or (inner.get("triggerSlug") if isinstance(inner, dict) else None)
            or event_type
        )

        # Entity ID: real V3 doesn't include entityId directly.
        # Try explicit entity fields first, then resolve via connected_account_id.
        entity_id = (
            (meta.get("entityId") if isinstance(meta, dict) else None)
            or (meta.get("entity_id") if isinstance(meta, dict) else None)
            or (inner.get("entity_id") if isinstance(inner, dict) else None)
            or (inner.get("entityId") if isinstance(inner, dict) else None)
            or payload.get("entity_id", "")
            or ""
        )

        # V3 fallback: resolve entity_id from connected_account_id via DB
        if not entity_id and isinstance(meta, dict):
            connected_account_id = meta.get("connected_account_id") or meta.get("connectionId")
            if connected_account_id:
                conn_row = (
                    db.query(ComposioConnection)
                    .filter(ComposioConnection.connection_id == connected_account_id)
                    .first()
                )
                if conn_row:
                    entity_row = db.query(ComposioEntity).filter(ComposioEntity.id == conn_row.entity_id).first()
                    if entity_row:
                        entity_id = entity_row.composio_entity_id
                        logger.info("V3: resolved entity_id=%s from connected_account_id=%s",
                                    entity_id, connected_account_id)
                    else:
                        logger.warning("V3: ComposioConnection found (id=%s) but no ComposioEntity for entity_id=%s",
                                       conn_row.id, conn_row.entity_id)
                else:
                    logger.warning("V3: No ComposioConnection found for connected_account_id=%s", connected_account_id)

        # Event data: the actual trigger payload is in "data"
        event_data = inner if isinstance(inner, dict) else {}

    else:
        # V2 / legacy format
        trigger_name = payload.get("trigger_name") or payload.get("triggerSlug", "")
        entity_id = payload.get("entity_id") or payload.get("entityId", "")
        event_data = payload.get("event_data") or payload.get("payload", {})

    if not trigger_name:
        logger.warning("Webhook missing trigger_name — full payload: %s", str(payload)[:500])
        raise HTTPException(status_code=400, detail="Missing trigger_name")
    
    logger.info(f"Received webhook: {trigger_name} for entity {entity_id}")

    # Determine ingestor by trigger_name prefix and build RequestEnvelope
    if trigger_name.startswith("JIRA_"):
        ingestor = JiraTriggerIngestor()
        envelope = ingestor.ingest(
            trigger_name=trigger_name,
            entity_id=entity_id,
            event_data=event_data,
            db=db,
        )
    else:
        # Unsupported trigger type — store as unrouted and return
        logger.warning(f"No ingestor for trigger prefix: {trigger_name}")
        try:
            unrouted = UnroutedEvent(
                workspace_id=UUID(int=0),
                source=trigger_name,
                content=str(event_data)[:2000],
                raw_payload=payload,
                reason=f"No ingestor for trigger: {trigger_name}",
            )
            db.add(unrouted)
            db.commit()
        except Exception:
            logger.exception("Failed to store unrouted webhook event")
            db.rollback()
        return {
            "status": "unrouted",
            "trigger_name": trigger_name,
            "reason": f"No ingestor for trigger: {trigger_name}",
        }

    # Route the envelope through the universal router
    universal_router = UniversalRouter(db, cache=get_routing_cache())
    try:
        decision = await universal_router.route(envelope)
    except Exception:
        logger.exception("Router failed for webhook event")
        decision = None

    if decision is None:
        # No route found — already stored as unrouted by router
        return {
            "status": "unrouted",
            "trigger_name": trigger_name,
            "entity_id": entity_id,
            "reason": "No route found",
        }

    # Dispatch based on routing decision (async background task)
    if decision.route_type == "agent" and decision.agent_id is not None:
        asyncio.create_task(
            _dispatch_agent(
                agent_id=decision.agent_id,
                content=envelope.content,
                metadata=envelope.metadata,
            )
        )
    elif decision.route_type == "workflow" and decision.workflow_id is not None:
        asyncio.create_task(
            _dispatch_workflow(
                workflow_id=decision.workflow_id,
                envelope=envelope,
            )
        )

    return {
        "status": "dispatched",
        "trigger_name": trigger_name,
        "entity_id": entity_id,
        "route_type": decision.route_type,
        "agent_id": decision.agent_id,
        "workflow_id": decision.workflow_id,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning[:200],
    }


# =============================================================================
# Webhook Dispatch Helpers
# =============================================================================

async def _dispatch_agent(
    agent_id: int,
    content: str,
    metadata: Dict[str, Any],
) -> None:
    """Execute an agent in the background for a webhook-triggered request."""
    try:
        from modules.agents.factory.agent_factory import AgentFactory

        with get_db_session() as db:
            factory = AgentFactory(db_session=db)
            result = await factory.execute_with_prompt(
                agent=agent_id,
                prompt=content,
                context=metadata,
            )
            status = result.get("status", "unknown")
            logger.info(
                f"[webhook] Agent {agent_id} execution completed: status={status}"
            )
    except Exception:
        logger.exception(f"[webhook] Agent {agent_id} dispatch failed")


async def _dispatch_workflow(
    workflow_id: int,
    envelope: "RequestEnvelope",
) -> None:
    """Execute a workflow in the background for a webhook-triggered request.

    If a recipe is registered for *workflow_id* in the recipe registry,
    the recipe is executed directly.  Otherwise, falls back to the
    standard 9-stage workflow pipeline.
    """
    try:
        from modules.workflows.recipes import get_recipe

        recipe_cls = get_recipe(workflow_id=workflow_id)
        if recipe_cls is not None:
            # Recipe-based dispatch (Python-registered)
            with get_db_session() as db:
                recipe = recipe_cls()
                result = await recipe.execute(envelope, db, envelope.workspace_id)
                logger.info(
                    "[webhook] Recipe for workflow %d completed: success=%s",
                    workflow_id,
                    result.success if hasattr(result, "success") else "unknown",
                )
            return

        # UI-created recipe dispatch (workflow_recipes table)
        with get_db_session() as db:
            recipe_row = db.query(WorkflowRecipe).filter(
                WorkflowRecipe.id == workflow_id
            ).first()
            if recipe_row:
                from uuid import uuid4 as _uuid4
                # PRD-142 W3-S12: composio trigger dispatch goes via the engine seam.
                from services.playbook_engine import get_playbook_engine

                execution_id = f"trigger-{_uuid4().hex[:12]}"
                execution = RecipeExecution(
                    execution_id=execution_id,
                    recipe_id=recipe_row.id,
                    workspace_id=envelope.workspace_id,
                    status='pending',
                    input_data={"content": envelope.content, "metadata": envelope.metadata},
                    triggered_by="composio_trigger",
                    execution_metadata={
                        'execution_type': 'trigger_dispatch',
                        'total_steps': len(recipe_row.steps or []),
                        'trigger_source': envelope.source,
                    },
                )
                db.add(execution)
                recipe_row.use_count += 1
                recipe_row.last_used_at = datetime.utcnow()
                db.commit()

                logger.info(
                    "[webhook] Dispatching to UI recipe %d (%s), execution=%s",
                    recipe_row.id, recipe_row.name, execution_id,
                )
                await get_playbook_engine().execute_direct(
                    recipe_execution_id=execution_id,
                    recipe_id=recipe_row.id,
                    workspace_id=envelope.workspace_id,
                    input_data={"content": envelope.content, **envelope.metadata},
                )
                return

        # Standard workflow dispatch (no matching recipe)
        from api.workflows import execute_workflow_with_progress
        from core.models.core import WorkflowExecution, ExecutionStatus

        with get_db_session() as db:
            execution = WorkflowExecution(
                workflow_id=workflow_id,
                input_data={
                    "content": envelope.content,
                    "metadata": envelope.metadata,
                },
                status=ExecutionStatus.PENDING.value,
            )
            db.add(execution)
            db.commit()
            db.refresh(execution)
            execution_id = execution.id

        await execute_workflow_with_progress(execution_id, {})
        logger.info(
            "[webhook] Workflow %d execution %d completed",
            workflow_id,
            execution_id,
        )
    except Exception:
        logger.exception("[webhook] Workflow %d dispatch failed", workflow_id)


# =============================================================================
# Trigger Subscription Endpoints
# =============================================================================

@router.post("/triggers/subscribe", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def subscribe_to_trigger(
    request: TriggerSubscriptionRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Subscribe to a Composio trigger.
    """
    client = get_composio_client()
    entity_manager = EntityManager(db)
    
    entity = entity_manager.get_or_create_entity(ctx.workspace_id)
    
    # Build callback URL
    backend_url = config.BACKEND_URL or "http://localhost:8000"
    callback_url = f"{backend_url}/api/composio/webhook"

    try:
        result = client.subscribe_to_trigger(
            entity_id=entity["composio_entity_id"],
            trigger_name=request.trigger_name,
            callback_url=callback_url
        )
    except Exception as e:
        logger.error(f"Failed to subscribe to trigger: {e}")
        raise HTTPException(status_code=503, detail="Failed to subscribe to trigger")
    
    # Store subscription
    subscription = TriggerSubscription(
        entity_id=entity["id"],
        trigger_name=request.trigger_name,
        callback_url=callback_url,
        agent_id=request.agent_id,
        workflow_id=request.workflow_id,
        composio_subscription_id=result.get("id"),
        is_active=True
    )
    db.add(subscription)
    db.commit()
    
    return {
        "status": "subscribed",
        "trigger_name": request.trigger_name,
        "subscription_id": subscription.id
    }


@router.get("/triggers")
async def list_trigger_subscriptions(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    List all trigger subscriptions for the workspace.
    """
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    
    if not entity:
        return []
    
    subscriptions = db.query(TriggerSubscription).filter(
        TriggerSubscription.entity_id == entity["id"],
        TriggerSubscription.is_active == True
    ).all()
    
    return [
        {
            "id": s.id,
            "trigger_name": s.trigger_name,
            "agent_id": s.agent_id,
            "workflow_id": s.workflow_id,
            "created_at": s.created_at
        }
        for s in subscriptions
    ]


@router.delete("/triggers/{subscription_id}", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def unsubscribe_trigger(
    subscription_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Unsubscribe from a trigger.
    """
    client = get_composio_client()
    
    subscription = db.query(TriggerSubscription).filter(
        TriggerSubscription.id == subscription_id
    ).first()
    
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    # Unsubscribe from Composio
    if subscription.composio_subscription_id:
        try:
            client.unsubscribe_trigger(subscription.composio_subscription_id)
        except Exception as e:
            logger.warning(f"Failed to unsubscribe from Composio: {e}")
    
    # Mark as inactive
    subscription.is_active = False
    db.commit()
    
    return {"status": "unsubscribed", "subscription_id": subscription_id}
