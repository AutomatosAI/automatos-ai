

"""
Tools API (Rewrite)

This router is the new source-of-truth for the Tools page.
It serves marketplace metadata from local cache tables for fast loads.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.composio.client import get_composio_client
from core.composio.entity_manager import EntityManager
from core.database.database import get_db
from core.models.composio_cache import ComposioActionCache, ComposioAppCache, ComposioStatsCache
from services.metadata_sync_service import MetadataSyncService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tools", tags=["Tools"])


def _assert_workspace_admin(ctx: RequestContext) -> None:
    """Raise 403 unless the user has admin or owner role."""
    role = getattr(ctx.user, 'role', 'user') if ctx.user else 'user'
    if role not in ('admin', 'owner'):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

INTERNAL_APP_NAMES = {"RAG", "MEMORY", "NL2SQL", "CODEGRAPH"}


class AppOut(BaseModel):
    id: int
    app_name: str
    display_name: str
    description: Optional[str] = None
    logo_url: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    auth_schemes: List[str] = Field(default_factory=list)
    action_count: int = 0
    trigger_count: int = 0
    status: str = "ACTIVE"
    is_connected: bool = False
    triggers: List[Dict[str, Any]] = Field(default_factory=list)  # Include triggers in response


class MarketplaceOut(BaseModel):
    apps: List[AppOut]
    total_apps: int
    total_actions: int
    categories: Dict[str, Any]
    last_synced: Optional[str] = None


class StatsOut(BaseModel):
    total_apps: int
    total_actions: int
    connected_apps: int
    categories: Dict[str, Any]
    last_synced: Optional[str] = None


class ConnectIn(BaseModel):
    app_name: str
    callback_url: Optional[str] = None


@router.get("/marketplace", response_model=MarketplaceOut)
async def marketplace(
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    # Connected apps for this workspace (from DB ONLY - no API calls)
    # PERFORMANCE FIX: Removed pending connection sync from page loads.
    # Pending connections are now only synced:
    # 1. After OAuth callback (in composio.py)
    # 2. Via manual refresh endpoint
    # This eliminates 48+ Composio API calls on every page load.
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    connected_set = set()
    if entity:
        connections = entity_manager.get_entity_connections(entity["id"])
        # Just read active connections from DB - no API calls
        connected_set = {
            (c.get("app_name") or "").upper()
            for c in connections
            if (c.get("status") or "").lower() == "active"
        }

    # Read stats from composio_stats_cache (populated by /api/tools/sync)
    stats_rows = db.query(ComposioStatsCache).all()
    stats = {r.stat_key: r.stat_value for r in stats_rows}
    last_synced = (stats.get("last_full_sync") or {}).get("timestamp")

    # Cache-only marketplace (no live fallback).
    # Internal tools should never appear here.
    q = db.query(ComposioAppCache).filter(
        ComposioAppCache.status == "ACTIVE",
        ~ComposioAppCache.app_name.in_(INTERNAL_APP_NAMES),
    )
    if category:
        q = q.filter(ComposioAppCache.categories.contains([category]))
    if search:
        st = f"%{search}%"
        q = q.filter(
            (ComposioAppCache.display_name.ilike(st))
            | (ComposioAppCache.app_name.ilike(st))
            | (ComposioAppCache.description.ilike(st))
        )

    total_apps = q.count()
    apps = (
        q.order_by(ComposioAppCache.action_count.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    # Build response with triggers from app_metadata
    apps_out = []
    for a in apps:
        # Extract triggers from app_metadata JSONB field
        meta = a.app_metadata or {}
        triggers = meta.get("triggers") or []
        
        # If triggers are missing but trigger_count > 0, log a warning
        # This indicates triggers weren't synced properly
        if not triggers and a.trigger_count > 0:
            logger.warning(
                f"App {a.app_name} has trigger_count={a.trigger_count} but no triggers in metadata. "
                f"Re-sync may be needed: POST /api/tools/sync"
            )
        
        apps_out.append(
            AppOut(
                id=a.id,
                app_name=a.app_name,
                display_name=a.display_name,
                description=a.description,
                logo_url=a.logo_url,
                categories=a.categories or [],
                auth_schemes=a.auth_schemes or [],
                action_count=a.action_count or 0,
                trigger_count=a.trigger_count or 0,
                status=a.status,
                is_connected=a.app_name.upper() in connected_set,
                triggers=triggers,  # Include triggers array
            )
        )
    
    return MarketplaceOut(
        apps=apps_out,
        total_apps=total_apps,
        total_actions=(stats.get("total_actions") or {}).get("count", 0),
        categories=stats.get("categories") or {},
        last_synced=last_synced,
    )


@router.get("/stats", response_model=StatsOut)
async def stats(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    stats_rows = db.query(ComposioStatsCache).all()
    stats = {r.stat_key: r.stat_value for r in stats_rows}

    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    connected = 0
    if entity:
        connected = sum(
            1
            for c in entity_manager.get_entity_connections(entity["id"])
            if c.get("status") == "active"
        )

    return StatsOut(
        total_apps=(stats.get("total_apps") or {}).get("count", 0),
        total_actions=(stats.get("total_actions") or {}).get("count", 0),
        connected_apps=connected,
        categories=stats.get("categories") or {},
        last_synced=(stats.get("last_full_sync") or {}).get("timestamp"),
    )


@router.get("/connected")
async def connected(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    if not entity:
        return {"apps": [], "total": 0}

    connections = entity_manager.get_entity_connections(entity["id"])

    # PERFORMANCE FIX: Removed pending sync from page loads (see marketplace endpoint comment)

    # Include both "active" (connected) and "added" (in workspace but not connected yet) apps
    active = [c for c in connections if (c.get("status") or "").lower() in ("active", "added")]

    # Enrich with cached metadata if present
    conn_app_names = [c.get("app_name") for c in active if c.get("app_name")]
    app_names_upper = [(a or "").upper() for a in conn_app_names]
    cache = {
        a.app_name: a
        for a in db.query(ComposioAppCache).filter(
            ComposioAppCache.app_name.in_(list(set(app_names_upper)))
        ).all()
    }
    out = []
    for c in active:
        app_name = (c.get("app_name") or "").upper()
        cached = cache.get(app_name)
        meta = (cached.app_metadata or {}) if cached else {}
        triggers = meta.get("triggers") or []
        action_count = cached.action_count if cached else 0
        if action_count == 0:
            n = (
                db.query(ComposioActionCache)
                .filter(ComposioActionCache.app_name == app_name)
                .count()
            )
            action_count = n
        app_data = {
            "id": cached.id if cached else None,
            "app_name": app_name,
            "status": c.get("status"),  # This should be 'active' or 'added'
            "connected_at": c.get("connected_at"),
            "connection_id": c.get("connection_id"),
            "display_name": cached.display_name if cached else app_name,
            "description": cached.description if cached else None,
            "logo_url": cached.logo_url if cached else None,
            "categories": cached.categories if cached else [],
            "action_count": action_count,
            "trigger_count": cached.trigger_count if cached else 0,
            "triggers": triggers if isinstance(triggers, list) else [],
        }
        logger.info(f"[CONNECTED_APPS] {app_name}: status={c.get('status')}, connection_id={c.get('connection_id')}")
        out.append(app_data)
    logger.info(f"[CONNECTED_APPS] Returning {len(out)} apps total")
    return {"apps": out, "total": len(out)}


@router.get("/{app_name}/actions")
async def app_actions(
    app_name: str,
    search: Optional[str] = Query(None),
    # NOTE: Many apps (e.g., GitHub) exceed 500 actions.
    # Default higher so UI can show "all actions" without implementing pagination first.
    limit: int = Query(5000, ge=1, le=20000),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get actions for an app with enabled state for this workspace."""
    from core.models.tool_assignments import WorkspaceToolConfig

    # DB-backed: read from composio_actions_cache (populated by /api/tools/sync)
    q = db.query(ComposioActionCache).filter(ComposioActionCache.app_name == app_name.upper())
    if search:
        st = f"%{search}%"
        q = q.filter(
            (ComposioActionCache.action_name.ilike(st))
            | (ComposioActionCache.display_name.ilike(st))
            | (ComposioActionCache.description.ilike(st))
        )

    clean_tool_id = app_name.lower()
    config = (
        db.query(WorkspaceToolConfig)
        .filter(WorkspaceToolConfig.workspace_id == ctx.workspace_id, WorkspaceToolConfig.tool_id == clean_tool_id)
        .first()
    )
    enabled_actions = set()
    if config and config.configuration:
        enabled_actions = set(config.configuration.get("enabled_actions", []))

    rows = q.order_by(ComposioActionCache.action_name.asc()).offset(offset).limit(limit).all()
    out: List[Dict[str, Any]] = []
    for a in rows:
        out.append(
            {
                "name": a.action_name,
                "display_name": a.display_name,
                "description": a.description,
                "app_name": a.app_name,
                "parameters": a.parameters or {},
                "enabled": a.action_name in enabled_actions,
            }
        )
    return out


@router.get("/{app_name}/triggers")
async def app_triggers(
    app_name: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Get triggers for an app.

    Triggers are synced into `composio_apps_cache.metadata["triggers"]` during `/api/tools/sync`.
    This avoids schema changes while still exposing all Composio trigger types in the UI.
    """
    cached = db.query(ComposioAppCache).filter(ComposioAppCache.app_name == app_name.upper()).first()
    if not cached:
        return []
    meta = cached.app_metadata or {}
    triggers = meta.get("triggers") or []
    return triggers if isinstance(triggers, list) else []


@router.post("/{app_name}/actions")
async def save_app_actions(
    app_name: str,
    payload: Dict[str, Any],
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Persist enabled actions for an app in this workspace."""
    from core.models.tool_assignments import WorkspaceToolConfig

    enabled_actions = payload.get("actions", []) or []
    clean_tool_id = app_name.lower()

    config = (
        db.query(WorkspaceToolConfig)
        .filter(WorkspaceToolConfig.workspace_id == ctx.workspace_id, WorkspaceToolConfig.tool_id == clean_tool_id)
        .first()
    )

    if not config:
        config = WorkspaceToolConfig(
            workspace_id=ctx.workspace_id,
            tool_id=clean_tool_id,
            display_name=app_name,
            enabled=True,
            configuration={"enabled_actions": enabled_actions},
        )
        db.add(config)
    else:
        current_config = dict(config.configuration or {})
        current_config["enabled_actions"] = enabled_actions
        config.configuration = current_config
        config.enabled = True

    db.commit()
    return {"status": "success", "enabled_count": len(enabled_actions), "app_name": app_name.upper()}


@router.post("/connect")
async def connect_app(
    payload: ConnectIn,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    # Delegate to existing Composio client + entity manager logic
    client = get_composio_client()
    entity_manager = EntityManager(db)
    entity = entity_manager.get_or_create_entity(ctx.workspace_id)

    app_name = payload.app_name.upper()
    try:
        redirect_url = client.initiate_connection(
            entity_id=entity["composio_entity_id"],
            app=app_name,
            callback_url=payload.callback_url,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to initiate OAuth: {str(e)}")

    # Store pending connection in DB
    entity_manager.add_connection(entity_id=entity["id"], app_name=app_name, status="pending")
    return {"redirect_url": redirect_url, "app_name": app_name}


@router.get("/workspace")
async def workspace_tools(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Get ALL tools in workspace (both connected and not connected).
    This is separate from /connected which only shows active connections.
    """
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    if not entity:
        return {"apps": [], "total": 0}

    connections = entity_manager.get_entity_connections(entity["id"])

    # Include active, added, and pending (all workspace tools)
    workspace_tools = [c for c in connections if (c.get("status") or "").lower() in ("active", "added", "pending")]

    # Enrich with metadata
    conn_app_names = [c.get("app_name") for c in workspace_tools if c.get("app_name")]
    app_names_upper = [(a or "").upper() for a in conn_app_names]
    cache = {
        a.app_name: a
        for a in db.query(ComposioAppCache).filter(
            ComposioAppCache.app_name.in_(list(set(app_names_upper)))
        ).all()
    }

    out = []
    for c in workspace_tools:
        app_name = (c.get("app_name") or "").upper()
        cached = cache.get(app_name)
        meta = (cached.app_metadata or {}) if cached else {}
        triggers = meta.get("triggers") or []
        action_count = cached.action_count if cached else 0

        out.append({
            "id": cached.id if cached else None,
            "app_name": app_name,
            "status": c.get("status"),  # This is important - shows actual status
            "connected_at": c.get("connected_at"),
            "connection_id": c.get("connection_id"),
            "display_name": cached.display_name if cached else app_name,
            "description": cached.description if cached else None,
            "logo_url": cached.logo_url if cached else None,
            "categories": cached.categories if cached else [],
            "action_count": action_count,
            "trigger_count": cached.trigger_count if cached else 0,
            "triggers": triggers if isinstance(triggers, list) else [],
        })

    return {"apps": out, "total": len(out)}


@router.post("/add-to-workspace")
async def add_to_workspace(
    payload: ConnectIn,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Add a tool to workspace without OAuth connection.
    This creates a workspace record so the tool appears in Applications tab.
    User can then click Connect to initiate OAuth.
    """
    logger.info(f"[ADD_TO_WORKSPACE] app={payload.app_name}, workspace={ctx.workspace_id}, user_id={ctx.user.id if ctx.user else 'None'}")

    logger.info("[ADD_TO_WORKSPACE] Step 1: Creating EntityManager...")
    entity_manager = EntityManager(db)

    logger.info("[ADD_TO_WORKSPACE] Step 2: Getting or creating entity...")
    entity = entity_manager.get_or_create_entity(ctx.workspace_id)
    logger.info(f"[ADD_TO_WORKSPACE] Entity result: {entity}")

    app_name = payload.app_name.upper()
    logger.info(f"[ADD_TO_WORKSPACE] Step 3: Normalized app_name: {app_name}")

    # Check if already exists
    logger.info("[ADD_TO_WORKSPACE] Step 4: Checking existing connections...")
    existing = entity_manager.get_entity_connections(entity["id"])
    logger.info(f"[ADD_TO_WORKSPACE] Found {len(existing)} existing connections")
    logger.info(f"[ADD_TO_WORKSPACE] Existing connections: {[c.get('app_name') for c in existing]}")

    for conn in existing:
        if (conn.get("app_name") or "").upper() == app_name:
            current_status = (conn.get("status") or "").lower()

            # Allow overwriting PENDING connections (OAuth failed/abandoned)
            # This lets users retry OAuth flows that didn't complete
            if current_status == "pending":
                logger.info(f"[ADD_TO_WORKSPACE] Overwriting pending connection for {app_name}")
                entity_manager.update_connection_status(
                    entity_id=entity["id"],
                    app_name=app_name,
                    status="added"
                )
                db.commit()
                logger.info(f"[ADD_TO_WORKSPACE] ✅ Updated {app_name} from pending to added")
                logger.info("=" * 80)
                return {
                    "status": "success",
                    "message": f"{app_name} is ready to connect. Click Connect to authorize.",
                    "app_name": app_name
                }

            # For active/added connections, return already_added
            logger.info(f"[ADD_TO_WORKSPACE] ⚠️  App already exists: {app_name} (status: {current_status})")
            logger.info("=" * 80)
            return {
                "status": "already_added",
                "message": f"{app_name} is already in your workspace",
                "app_name": app_name
            }

    # Add connection with status "added" (not connected yet)
    logger.info(f"[ADD_TO_WORKSPACE] Step 5: Adding connection to database...")
    logger.info(f"[ADD_TO_WORKSPACE] entity_id={entity['id']}, app_name={app_name}, status='added'")
    entity_manager.add_connection(entity_id=entity["id"], app_name=app_name, status="added")

    # Ensure changes are committed to database
    logger.info("[ADD_TO_WORKSPACE] Step 6: Committing to database...")
    db.commit()
    logger.info("[ADD_TO_WORKSPACE] ✅ Database commit successful!")

    logger.info(f"[ADD_TO_WORKSPACE] ✅ Added {app_name} to workspace for entity {entity['id']}")
    logger.info("=" * 80)

    return {
        "status": "success",
        "message": f"{app_name} added to workspace. Click Connect to authorize.",
        "app_name": app_name
    }


@router.get("/debug/connections")
async def debug_connections(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    DEBUG: Show all connection records for this workspace
    """
    _assert_workspace_admin(ctx)
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    if not entity:
        return {"connections": [], "total": 0}

    connections = entity_manager.get_entity_connections(entity["id"])

    # Format for readability
    formatted = []
    for conn in connections:
        formatted.append({
            "app_name": conn.get("app_name"),
            "status": conn.get("status"),
            "created_at": str(conn.get("created_at")),
            "connected_at": str(conn.get("connected_at")) if conn.get("connected_at") else None,
            "connection_id": conn.get("connection_id")
        })

    return {
        "workspace_id": str(ctx.workspace_id),
        "entity_id": entity["id"],
        "connections": formatted,
        "total": len(formatted)
    }


@router.delete("/remove-from-workspace/{app_name}")
async def remove_from_workspace(
    app_name: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Remove an app from workspace (deletes the connection record).
    Works for both connected and unconnected apps.
    """
    _assert_workspace_admin(ctx)
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    if not entity:
        raise HTTPException(status_code=404, detail="No entity found for workspace")

    app_upper = app_name.upper()

    # Delete the connection record using entity_manager
    success = entity_manager.remove_connection(str(entity["id"]), app_upper)

    if not success:
        raise HTTPException(status_code=404, detail=f"{app_upper} not found in workspace")

    logger.info(f"[REMOVE_FROM_WORKSPACE] Removed {app_upper} from workspace for entity {entity['id']}")

    return {
        "status": "success",
        "message": f"{app_upper} removed from workspace"
    }


@router.delete("/debug/connections/{app_name}")
async def delete_connection_record(
    app_name: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    DEBUG: Delete a specific connection record
    """
    _assert_workspace_admin(ctx)
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    if not entity:
        return {"error": "No entity found"}

    # Delete the connection
    from core.models.composio_cache import ComposioConnection
    deleted = db.query(ComposioConnection).filter(
        ComposioConnection.entity_id == entity["id"],
        ComposioConnection.app_name.ilike(app_name)
    ).delete()

    db.commit()

    return {
        "deleted": deleted,
        "app_name": app_name
    }


@router.post("/cleanup-pending")
async def cleanup_pending_connections(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Clean up stale pending connections (OAuth started but never completed).

    Removes connections with status='pending' that are older than 1 hour.
    This allows users to retry OAuth flows that failed or were abandoned.
    """
    _assert_workspace_admin(ctx)
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    if not entity:
        return {"error": "No entity found"}

    from core.models.composio import ComposioConnection
    from datetime import datetime, timedelta

    # Find pending connections older than 1 hour
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)

    stale_pending = db.query(ComposioConnection).filter(
        ComposioConnection.entity_id == entity["id"],
        ComposioConnection.status == 'pending',
        ComposioConnection.updated_at < one_hour_ago
    ).all()

    deleted_apps = [c.app_name for c in stale_pending]
    deleted_count = len(stale_pending)

    # Delete stale pending connections
    for conn in stale_pending:
        db.delete(conn)

    db.commit()

    logger.info(f"Cleaned up {deleted_count} stale pending connections for workspace {ctx.workspace_id}: {deleted_apps}")

    return {
        "deleted_count": deleted_count,
        "deleted_apps": deleted_apps,
        "message": f"Removed {deleted_count} stale pending connections (older than 1 hour)"
    }


@router.post("/debug/cleanup")
async def cleanup_connections(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    DEBUG: Clean up all non-active connection records (removes failed, error, pending, added)
    Keeps only 'active' connections
    """
    _assert_workspace_admin(ctx)
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    if not entity:
        return {"error": "No entity found"}

    from core.models.composio import ComposioConnection

    # Get all non-active connections
    connections_to_delete = db.query(ComposioConnection).filter(
        ComposioConnection.entity_id == entity["id"],
        ComposioConnection.status != 'active'
    ).all()

    deleted_apps = [c.app_name for c in connections_to_delete]

    # Delete them
    deleted_count = db.query(ComposioConnection).filter(
        ComposioConnection.entity_id == entity["id"],
        ComposioConnection.status != 'active'
    ).delete()

    db.commit()

    logger.info(f"Cleaned up {deleted_count} non-active connections for workspace {ctx.workspace_id}")

    return {
        "deleted_count": deleted_count,
        "deleted_apps": deleted_apps,
        "message": f"Removed {deleted_count} non-active connection records"
    }


@router.post("/sync")
async def sync(
    sync_type: str = Query("full", description="full or incremental"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    logger.info(f"Sync requested: type={sync_type}, workspace_id={ctx.workspace_id}")
    service = MetadataSyncService(db)
    if sync_type == "incremental":
        result = service.run_incremental_sync()
    else:
        result = service.run_full_sync()
    logger.info(f"Sync completed: {result}")
    return result


@router.post("/refresh-connections")
async def refresh_connections(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Manually refresh pending connections from Composio.

    This endpoint checks all pending connections and updates their status.
    Use this after OAuth callbacks to ensure UI reflects the latest state.

    PERFORMANCE NOTE: This makes API calls to Composio and should NOT
    be called on every page load. Use only when explicitly needed.
    """
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)

    if not entity:
        return {"synced": 0, "updated": 0, "message": "No entity found"}

    from datetime import datetime, timedelta
    connections = entity_manager.get_entity_connections(entity["id"])

    # Find pending connections to sync
    pending_to_sync = [
        conn for conn in connections
        if (conn.get("status") or "").lower() == "pending"
    ]

    if not pending_to_sync:
        return {"synced": 0, "updated": 0, "message": "No pending connections"}

    logger.info(f"[REFRESH] Syncing {len(pending_to_sync)} pending connections for workspace {ctx.workspace_id}")

    client = get_composio_client()
    updated_count = 0

    for conn in pending_to_sync:
        try:
            composio_status = client.get_connection_status(
                entity_id=entity["composio_entity_id"],
                app=(conn.get("app_name") or "").upper(),
            )
            if composio_status and composio_status.get("status") == "ACTIVE":
                entity_manager.update_connection_status(
                    entity_id=entity["id"],
                    app_name=conn.get("app_name") or "",
                    status="active",
                    connection_id=composio_status.get("id"),
                )
                updated_count += 1
                logger.info(f"[REFRESH] Updated {conn.get('app_name')} to active")
            else:
                # Mark as checked to avoid retry storms
                entity_manager.update_connection_status(
                    entity_id=entity["id"],
                    app_name=conn.get("app_name") or "",
                    status="pending",
                )
        except Exception as e:
            logger.error(f"[REFRESH] Failed to sync {conn.get('app_name')}: {e}")

    return {
        "synced": len(pending_to_sync),
        "updated": updated_count,
        "message": f"Synced {len(pending_to_sync)} connections, {updated_count} updated to active"
    }


@router.get("/sync/history")
async def sync_history(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    service = MetadataSyncService(db)
    return {"jobs": service.get_sync_history(limit), "count": limit}


# PRD-40: Dynamic Tool Suggestions
class SuggestionsOut(BaseModel):
    """Response model for tool suggestions"""
    app: str
    suggestions: List[str]
    source: str  # "curated" or "generated"


@router.get("/{app_name}/suggestions", response_model=SuggestionsOut)
async def get_tool_suggestions(
    app_name: str,
    db: Session = Depends(get_db),
):
    """
    Get suggestion prompts for a specific tool/app.

    Returns curated suggestions if available, otherwise generates
    basic suggestions from action schemas.

    Args:
        app_name: The Composio app name (case-insensitive)

    Returns:
        SuggestionsOut with app name, suggestions list, and source type
    """
    # Normalize app name to uppercase (Composio convention)
    app_name_upper = app_name.upper()

    # Try to get curated suggestions from cache
    # Note: app_suggestions is stored per-action but same for all actions of an app
    cached = db.query(ComposioActionCache).filter(
        ComposioActionCache.app_name == app_name_upper
    ).first()

    if cached and cached.app_suggestions and len(cached.app_suggestions) > 0:
        return SuggestionsOut(
            app=app_name_upper,
            suggestions=cached.app_suggestions,
            source="curated"
        )

    # Fallback: Generate from action schemas
    suggestions = _generate_suggestions_from_schema(app_name_upper, db)
    return SuggestionsOut(
        app=app_name_upper,
        suggestions=suggestions,
        source="generated"
    )


def _generate_suggestions_from_schema(app_name: str, db: Session) -> List[str]:
    """
    Generate basic suggestions from action descriptions.

    This is a fallback for apps without curated suggestions.
    Analyzes action descriptions to create natural language prompts.

    Args:
        app_name: The app name to generate suggestions for
        db: Database session

    Returns:
        List of generated suggestion strings (3-4 suggestions)
    """
    # Get top 10 actions for this app
    actions = db.query(ComposioActionCache).filter(
        ComposioActionCache.app_name == app_name.upper()
    ).limit(10).all()

    if not actions:
        return [f"Get started with {app_name}"]

    suggestions = set()

    for action in actions:
        desc = (action.description or "").lower()
        action_display = action.display_name or action.action_name

        # Pattern matching to generate prompts
        if any(verb in desc for verb in ["list", "fetch", "get", "retrieve", "show"]):
            suggestions.add(f"Show my {app_name.lower()} items")
        if any(verb in desc for verb in ["send", "create", "post", "add"]):
            suggestions.add(f"Create a new {app_name.lower()} item")
        if "search" in desc or "find" in desc:
            suggestions.add(f"Search {app_name.lower()} for...")
        if any(verb in desc for verb in ["update", "edit", "modify", "change"]):
            suggestions.add(f"Update a {app_name.lower()} item")
        if any(verb in desc for verb in ["delete", "remove", "revoke"]):
            suggestions.add(f"Delete a {app_name.lower()} item")

    # Return up to 4 suggestions
    result = list(suggestions)[:4]

    # If we have fewer than 3, add generic ones
    if len(result) < 3:
        result.extend([
            f"Connect to {app_name}",
            f"Explore {app_name} features",
            f"Get help with {app_name}"
        ])

    return result[:4]
