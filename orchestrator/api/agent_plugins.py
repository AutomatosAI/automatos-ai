"""
Agent Plugin Assignment API Endpoints (PRD-42: US-012)
======================================================

Provides REST APIs for agent builders to assign marketplace plugins to agents:
- List plugins assigned to an agent
- Update (replace) plugin assignments for an agent
"""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["Agent Plugins"])


# ===================================================================
# Pydantic Models
# ===================================================================

class UpdateAgentPluginsBody(BaseModel):
    plugin_ids: List[UUID] = Field(..., description="List of plugin UUIDs to assign to the agent")


class AgentPluginOut(BaseModel):
    plugin_id: str
    slug: str
    name: str
    version: str
    description: Optional[str] = None
    skills_count: int = 0
    commands_count: int = 0
    token_estimate: int = 0
    priority: int = 0
    assigned_at: Optional[str] = None

    class Config:
        from_attributes = True


# ===================================================================
# Endpoints
# ===================================================================

@router.get("/{agent_id}/plugins", response_model=None)
async def list_agent_plugins(
    agent_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List plugins assigned to an agent (joined with marketplace_plugins for details)."""
    try:
        from core.models.core import Agent
        from core.models.marketplace_plugins import (
            AgentAssignedPlugin,
            MarketplacePlugin,
        )

        # Validate agent exists and belongs to authenticated user's workspace
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        if agent.workspace_id and agent.workspace_id != ctx.workspace_id:
            raise HTTPException(status_code=403, detail="Access denied: workspace mismatch")

        # Join AgentAssignedPlugin with MarketplacePlugin for details
        rows = (
            db.query(AgentAssignedPlugin, MarketplacePlugin)
            .join(
                MarketplacePlugin,
                MarketplacePlugin.id == AgentAssignedPlugin.plugin_id,
            )
            .filter(AgentAssignedPlugin.agent_id == agent_id)
            .order_by(AgentAssignedPlugin.priority.asc())
            .all()
        )

        items = []
        for aap, plugin in rows:
            items.append(AgentPluginOut(
                plugin_id=str(plugin.id),
                slug=plugin.slug,
                name=plugin.name,
                version=plugin.version,
                description=plugin.description,
                skills_count=plugin.skills_count or 0,
                commands_count=plugin.commands_count or 0,
                token_estimate=plugin.token_estimate or 0,
                priority=aap.priority or 0,
                assigned_at=aap.assigned_at.isoformat() if aap.assigned_at else None,
            ))

        return {"items": [item.model_dump() for item in items]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing agent plugins for agent %s: %s", agent_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to list agent plugins: {e}")


@router.put("/{agent_id}/plugins", response_model=None)
async def update_agent_plugins(
    agent_id: int,
    body: UpdateAgentPluginsBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update plugin assignments for an agent. Replaces existing assignments."""
    try:
        from core.models.core import Agent
        from core.models.marketplace_plugins import (
            AgentAssignedPlugin,
            MarketplacePlugin,
            WorkspaceEnabledPlugin,
        )

        # Validate agent exists and belongs to authenticated user's workspace
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        if agent.workspace_id and agent.workspace_id != ctx.workspace_id:
            raise HTTPException(status_code=403, detail="Access denied: workspace mismatch")

        workspace_id = agent.workspace_id or ctx.workspace_id

        # Validate all plugin_ids are enabled for the agent's workspace
        if body.plugin_ids:
            enabled_plugin_ids = {
                row.plugin_id
                for row in db.query(WorkspaceEnabledPlugin.plugin_id).filter(
                    WorkspaceEnabledPlugin.workspace_id == workspace_id,
                ).all()
            }

            not_enabled = [
                str(pid) for pid in body.plugin_ids if pid not in enabled_plugin_ids
            ]

            if not_enabled:
                raise HTTPException(
                    status_code=400,
                    detail=f"Plugins not enabled for workspace: {', '.join(not_enabled)}",
                )

        # Remove existing assignments
        db.query(AgentAssignedPlugin).filter(
            AgentAssignedPlugin.agent_id == agent_id,
        ).delete(synchronize_session="fetch")

        # Create new assignments with priority based on list order
        for priority, plugin_id in enumerate(body.plugin_ids):
            assignment = AgentAssignedPlugin(
                agent_id=agent_id,
                plugin_id=plugin_id,
                priority=priority,
            )
            db.add(assignment)

        db.commit()

        return {
            "success": True,
            "message": f"Agent plugins updated ({len(body.plugin_ids)} assigned)",
            "agent_id": agent_id,
            "plugin_ids": [str(pid) for pid in body.plugin_ids],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating plugins for agent %s: %s", agent_id, e)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update agent plugins: {e}")
