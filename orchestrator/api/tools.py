

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
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    connected_set = set()
    if entity:
        connected_set = {
            (c.get("app_name") or "").upper()
            for c in entity_manager.get_entity_connections(entity["id"])
            if c.get("status") == "active"
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

    return MarketplaceOut(
        apps=[
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
            )
            for a in apps
        ],
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
    active = [c for c in connections if c.get("status") == "active"]

    # Enrich with cached metadata if present
    cache = {
        a.app_name: a
        for a in db.query(ComposioAppCache).filter(ComposioAppCache.app_name.in_([c["app_name"] for c in active])).all()
    }
    out = []
    for c in active:
        app_name = (c.get("app_name") or "").upper()
        cached = cache.get(app_name)
        out.append(
            {
                "id": cached.id if cached else None,
                "app_name": app_name,
                "status": c.get("status"),
                "connected_at": c.get("connected_at"),
                "connection_id": c.get("connection_id"),
                "display_name": cached.display_name if cached else app_name,
                "logo_url": cached.logo_url if cached else None,
                "categories": cached.categories if cached else [],
                "action_count": cached.action_count if cached else 0,
            }
        )
    return {"apps": out, "total": len(out)}


@router.get("/{app_name}/actions")
async def app_actions(
    app_name: str,
    search: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
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
    service = MetadataSyncService(db)
    if sync_type == "incremental":
        result = service.run_incremental_sync()
    else:
        result = service.run_full_sync()
    return result


@router.get("/sync/history")
async def sync_history(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    service = MetadataSyncService(db)
    return {"jobs": service.get_sync_history(limit), "count": limit}
