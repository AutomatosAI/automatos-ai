

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
    # Connected apps for this workspace (from DB)
    # NOTE: Some Composio hosted-auth callbacks don't always include `connection_id`.
    # To avoid "connected in Composio but not showing in UI", opportunistically
    # sync any pending connections to ACTIVE on read.
    # OPTIMIZATION: Only sync pending connections that haven't been checked recently (within last 30s)
    # to reduce redundant Composio API calls (which also reduces telemetry volume).
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    connected_set = set()
    if entity:
        from datetime import datetime, timedelta
        connections = entity_manager.get_entity_connections(entity["id"])
        pending_to_sync = []
        for conn in connections:
            status = (conn.get("status") or "").lower()
            if status == "pending":
                # Only sync if last_synced_at is older than 30 seconds or doesn't exist
                last_synced = conn.get("last_synced_at")
                if not last_synced or (isinstance(last_synced, datetime) and (datetime.utcnow() - last_synced) > timedelta(seconds=30)):
                    pending_to_sync.append(conn)
            elif status == "active":
                connected_set.add((conn.get("app_name") or "").upper())
        
        # Batch sync pending connections (only if needed)
        if pending_to_sync:
            client = get_composio_client()
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
                        connected_set.add((conn.get("app_name") or "").upper())
                except Exception:
                    # best-effort sync only - mark as checked even on failure to avoid retry storms
                    try:
                        entity_manager.update_connection_status(
                            entity_id=entity["id"],
                            app_name=conn.get("app_name") or "",
                            status="pending",  # Keep as pending but update last_synced_at
                        )
                    except Exception:
                        pass
        else:
            # No pending connections to sync - just read active ones
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
    
    # OPTIMIZATION: Only sync pending connections that haven't been checked recently (within last 30s)
    # to reduce redundant Composio API calls (which also reduces telemetry volume).
    from datetime import datetime, timedelta
    pending_to_sync = []
    for conn in connections:
        status = (conn.get("status") or "").lower()
        if status == "pending":
            last_synced = conn.get("last_synced_at")
            if not last_synced or (isinstance(last_synced, datetime) and (datetime.utcnow() - last_synced) > timedelta(seconds=30)):
                pending_to_sync.append(conn)
    
    # Batch sync pending connections (only if needed)
    if pending_to_sync:
        client = get_composio_client()
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
            except Exception:
                # Mark as checked even on failure to avoid retry storms
                try:
                    entity_manager.update_connection_status(
                        entity_id=entity["id"],
                        app_name=conn.get("app_name") or "",
                        status="pending",
                    )
                except Exception:
                    pass
        # Re-read after potential updates
        connections = entity_manager.get_entity_connections(entity["id"])
    
    active = [c for c in connections if (c.get("status") or "").lower() == "active"]

    # Enrich with cached metadata if present
    cache = {
        a.app_name: a
        for a in db.query(ComposioAppCache).filter(ComposioAppCache.app_name.in_([c["app_name"] for c in active])).all()
    }
    out = []
    for c in active:
        app_name = (c.get("app_name") or "").upper()
        cached = cache.get(app_name)
        meta = (cached.app_metadata or {}) if cached else {}
        triggers = meta.get("triggers") or []
        out.append(
            {
                "id": cached.id if cached else None,
                "app_name": app_name,
                "status": c.get("status"),
                "connected_at": c.get("connected_at"),
                "connection_id": c.get("connection_id"),
                "display_name": cached.display_name if cached else app_name,
                "description": cached.description if cached else None,
                "logo_url": cached.logo_url if cached else None,
                "categories": cached.categories if cached else [],
                "action_count": cached.action_count if cached else 0,
                "trigger_count": cached.trigger_count if cached else 0,
                "triggers": triggers if isinstance(triggers, list) else [],
            }
        )
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


@router.get("/sync/history")
async def sync_history(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    service = MetadataSyncService(db)
    return {"jobs": service.get_sync_history(limit), "count": limit}
