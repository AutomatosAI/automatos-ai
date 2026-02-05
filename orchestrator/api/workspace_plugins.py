"""
Workspace Plugin Enablement API Endpoints (PRD-42: US-011)
==========================================================

Provides REST APIs for workspace owners to enable/disable marketplace plugins:
- List enabled plugins for a workspace
- Enable a plugin for a workspace
- Disable a plugin for a workspace (cascades to agent assignments)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workspaces", tags=["Workspace Plugins"])


# ===================================================================
# Pydantic Models
# ===================================================================

class EnablePluginBody(BaseModel):
    plugin_id: UUID = Field(..., description="ID of the marketplace plugin to enable")


class WorkspacePluginOut(BaseModel):
    plugin_id: str
    slug: str
    name: str
    version: str
    description: Optional[str] = None
    category_name: Optional[str] = None
    skills_count: int = 0
    commands_count: int = 0
    token_estimate: int = 0
    enable_count: int = 0
    security_status: Optional[str] = None
    enabled_at: Optional[str] = None

    class Config:
        from_attributes = True


# ===================================================================
# Endpoints
# ===================================================================

@router.get("/{workspace_id}/plugins", response_model=None)
async def list_workspace_plugins(
    workspace_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List plugins enabled for a workspace (joined with marketplace_plugins for details)."""
    # Validate workspace matches authenticated user's workspace
    if ctx.workspace_id != workspace_id:
        raise HTTPException(status_code=403, detail="Access denied: workspace mismatch")

    try:
        from core.models.marketplace_plugins import (
            MarketplacePlugin,
            PluginCategory,
            WorkspaceEnabledPlugin,
        )

        # Join WorkspaceEnabledPlugin with MarketplacePlugin for details
        rows = (
            db.query(WorkspaceEnabledPlugin, MarketplacePlugin)
            .join(
                MarketplacePlugin,
                MarketplacePlugin.id == WorkspaceEnabledPlugin.plugin_id,
            )
            .filter(WorkspaceEnabledPlugin.workspace_id == workspace_id)
            .all()
        )

        items = []
        for wep, plugin in rows:
            cat_name = None
            if plugin.category_id:
                cat = db.query(PluginCategory).filter(
                    PluginCategory.id == plugin.category_id,
                ).first()
                if cat:
                    cat_name = cat.name

            items.append(WorkspacePluginOut(
                plugin_id=str(plugin.id),
                slug=plugin.slug,
                name=plugin.name,
                version=plugin.version,
                description=plugin.description,
                category_name=cat_name,
                skills_count=plugin.skills_count or 0,
                commands_count=plugin.commands_count or 0,
                token_estimate=plugin.token_estimate or 0,
                enable_count=plugin.enable_count or 0,
                security_status=plugin.security_status,
                enabled_at=wep.enabled_at.isoformat() if wep.enabled_at else None,
            ))

        return {"items": [item.model_dump() for item in items]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing workspace plugins for %s: %s", workspace_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to list workspace plugins: {e}")


@router.post("/{workspace_id}/plugins", status_code=201)
async def enable_plugin(
    workspace_id: UUID,
    body: EnablePluginBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Enable a marketplace plugin for a workspace."""
    # Validate workspace matches authenticated user's workspace
    if ctx.workspace_id != workspace_id:
        raise HTTPException(status_code=403, detail="Access denied: workspace mismatch")

    try:
        from core.models.marketplace_plugins import (
            MarketplacePlugin,
            WorkspaceEnabledPlugin,
        )

        # Validate plugin exists, is approved, and is active
        plugin = db.query(MarketplacePlugin).filter(
            MarketplacePlugin.id == body.plugin_id,
            MarketplacePlugin.approval_status == "approved",
            MarketplacePlugin.is_active == True,
        ).first()

        if not plugin:
            raise HTTPException(
                status_code=404,
                detail="Plugin not found or not approved/active",
            )

        # Check if already enabled
        existing = db.query(WorkspaceEnabledPlugin).filter(
            WorkspaceEnabledPlugin.workspace_id == workspace_id,
            WorkspaceEnabledPlugin.plugin_id == body.plugin_id,
        ).first()

        if existing:
            raise HTTPException(status_code=409, detail="Plugin is already enabled for this workspace")

        # Resolve user id for enabled_by (Integer FK to users table)
        enabled_by = None
        user_id_str = getattr(ctx.user, "id", None)
        if user_id_str:
            try:
                enabled_by = int(user_id_str)
            except (ValueError, TypeError):
                # Clerk user id or non-integer — skip FK
                enabled_by = None

        # Create junction record
        junction = WorkspaceEnabledPlugin(
            workspace_id=workspace_id,
            plugin_id=body.plugin_id,
            enabled_by=enabled_by,
        )
        db.add(junction)

        # Increment enable_count on the plugin
        plugin.enable_count = (plugin.enable_count or 0) + 1

        db.commit()

        return {
            "success": True,
            "message": f"Plugin '{plugin.name}' enabled for workspace",
            "plugin_id": str(plugin.id),
            "workspace_id": str(workspace_id),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error enabling plugin for workspace %s: %s", workspace_id, e)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to enable plugin: {e}")


@router.delete("/{workspace_id}/plugins/{plugin_id}")
async def disable_plugin(
    workspace_id: UUID,
    plugin_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Disable a plugin for a workspace. Also removes agent assignments for this workspace's agents."""
    # Validate workspace matches authenticated user's workspace
    if ctx.workspace_id != workspace_id:
        raise HTTPException(status_code=403, detail="Access denied: workspace mismatch")

    try:
        from core.models.marketplace_plugins import (
            AgentAssignedPlugin,
            MarketplacePlugin,
            WorkspaceEnabledPlugin,
        )

        # Check the junction record exists
        junction = db.query(WorkspaceEnabledPlugin).filter(
            WorkspaceEnabledPlugin.workspace_id == workspace_id,
            WorkspaceEnabledPlugin.plugin_id == plugin_id,
        ).first()

        if not junction:
            raise HTTPException(status_code=404, detail="Plugin is not enabled for this workspace")

        # Remove agent assignments for this plugin from agents in this workspace
        from core.models.core import Agent

        workspace_agent_ids = [
            a.id for a in db.query(Agent.id).filter(
                Agent.workspace_id == workspace_id,
            ).all()
        ]

        removed_agent_count = 0
        if workspace_agent_ids:
            removed_agent_count = db.query(AgentAssignedPlugin).filter(
                AgentAssignedPlugin.agent_id.in_(workspace_agent_ids),
                AgentAssignedPlugin.plugin_id == plugin_id,
            ).delete(synchronize_session="fetch")

        # Remove the workspace junction record
        db.delete(junction)

        # Decrement enable_count on the plugin
        plugin = db.query(MarketplacePlugin).filter(
            MarketplacePlugin.id == plugin_id,
        ).first()
        if plugin and plugin.enable_count and plugin.enable_count > 0:
            plugin.enable_count = plugin.enable_count - 1

        db.commit()

        return {
            "success": True,
            "message": f"Plugin disabled for workspace",
            "plugin_id": str(plugin_id),
            "workspace_id": str(workspace_id),
            "agents_unassigned": removed_agent_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error disabling plugin %s for workspace %s: %s", plugin_id, workspace_id, e)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to disable plugin: {e}")
