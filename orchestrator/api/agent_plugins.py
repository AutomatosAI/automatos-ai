"""
Agent Plugin Assignment API Endpoints (PRD-42: US-012, US-016)
==============================================================

Provides REST APIs for agent builders to assign marketplace plugins to agents:
- List plugins assigned to an agent
- Update (replace) plugin assignments for an agent
- Get assembled agent context (persona + plugin skills + tools)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
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


class AssembledContextOut(BaseModel):
    agent_id: int
    model: str
    temperature: float
    system_prompt: str
    persona: Dict[str, Any]
    plugins_loaded: List[str]
    tools: List[Dict[str, Any]]
    token_estimate: int


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
        logger.exception("Error listing agent plugins for agent %s: %s", agent_id, e)
        raise HTTPException(status_code=500, detail="Failed to list agent plugins")


@router.put("/{agent_id}/plugins", response_model=None, dependencies=[Depends(require_workspace_permission("agents:update"))])
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

        # Deduplicate plugin_ids while preserving order
        seen: set = set()
        unique_plugin_ids: list = []
        for pid in body.plugin_ids:
            if pid not in seen:
                seen.add(pid)
                unique_plugin_ids.append(pid)

        # Validate all plugin_ids are enabled for the agent's workspace
        if unique_plugin_ids:
            enabled_plugin_ids = {
                row.plugin_id
                for row in db.query(WorkspaceEnabledPlugin.plugin_id).filter(
                    WorkspaceEnabledPlugin.workspace_id == workspace_id,
                ).all()
            }

            not_enabled = [
                str(pid) for pid in unique_plugin_ids if pid not in enabled_plugin_ids
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
        for priority, plugin_id in enumerate(unique_plugin_ids):
            assignment = AgentAssignedPlugin(
                agent_id=agent_id,
                plugin_id=plugin_id,
                priority=priority,
            )
            db.add(assignment)

        # PRD-71: Auto-assign materialized skills from assigned plugins
        from core.models.core import Skill, agent_skills as agent_skills_table
        from core.models.marketplace_plugins import WorkspaceEnabledSkill

        # Collect candidate skill IDs from plugins, then validate against Skill table
        candidate_skill_ids: set = set()
        if unique_plugin_ids:
            plugins = db.query(MarketplacePlugin).filter(
                MarketplacePlugin.id.in_(unique_plugin_ids),
            ).all()
            for plugin in plugins:
                for sid in (plugin.materialized_skill_ids or []):
                    candidate_skill_ids.add(sid)

        if candidate_skill_ids:
            # Resolve to only IDs that actually exist in the skills table
            valid_skill_ids = {
                row[0] for row in db.query(Skill.id).filter(
                    Skill.id.in_(candidate_skill_ids),
                ).all()
            }
            if len(valid_skill_ids) < len(candidate_skill_ids):
                stale = candidate_skill_ids - valid_skill_ids
                logger.warning(
                    "Skipping %d stale materialized_skill_ids: %s",
                    len(stale), stale,
                )

            if valid_skill_ids:
                # Idempotent upsert: WorkspaceEnabledSkill has composite PK
                # (workspace_id, skill_id) so ON CONFLICT DO NOTHING works
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                for sid in valid_skill_ids:
                    stmt = pg_insert(WorkspaceEnabledSkill).values(
                        workspace_id=workspace_id,
                        skill_id=sid,
                    ).on_conflict_do_nothing()
                    db.execute(stmt)

                # agent_skills has no unique constraint — bulk check then insert
                from sqlalchemy import select as sa_select
                existing_skill_ids = {
                    row[0] for row in db.execute(
                        sa_select(agent_skills_table.c.skill_id).where(
                            agent_skills_table.c.agent_id == agent_id,
                        )
                    ).fetchall()
                }
                new_ids = valid_skill_ids - existing_skill_ids
                for sid in new_ids:
                    db.execute(
                        agent_skills_table.insert().values(
                            agent_id=agent_id, skill_id=sid,
                        )
                    )

                logger.info(
                    "Auto-assigned %d materialized skill(s) to agent %s",
                    len(valid_skill_ids), agent_id,
                )

        db.commit()

        return {
            "success": True,
            "message": f"Agent plugins updated ({len(unique_plugin_ids)} assigned)",
            "agent_id": agent_id,
            "plugin_ids": [str(pid) for pid in unique_plugin_ids],
            "materialized_skill_ids": sorted(int(sid) for sid in valid_skill_ids) if candidate_skill_ids else [],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error updating plugins for agent %s: %s", agent_id, e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update agent plugins")


@router.get("/{agent_id}/assembled-context", response_model=None)
async def get_assembled_context(
    agent_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return the fully assembled agent context including persona prompt,
    plugin skills, and tool definitions.  Used by AgentFactory at runtime."""
    try:
        from core.models.core import Agent
        from core.models.composio_cache import AgentAppAssignment, ComposioAppCache

        # ------------------------------------------------------------------
        # 1. Load agent & validate workspace
        # ------------------------------------------------------------------
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        if agent.workspace_id and agent.workspace_id != ctx.workspace_id:
            raise HTTPException(status_code=403, detail="Access denied: workspace mismatch")

        # ------------------------------------------------------------------
        # 2. Resolve model & temperature from agent model_config
        # ------------------------------------------------------------------
        model_cfg = agent.model_config or {}
        model_id = model_cfg.get("model_id") if isinstance(model_cfg, dict) else None
        if not model_id:
            raise HTTPException(400, "Agent has no model configured. Set a model in Agent > Model tab.")
        temperature = model_cfg.get("temperature", 0.7) if isinstance(model_cfg, dict) else 0.7

        # ------------------------------------------------------------------
        # 3. Build persona section
        # ------------------------------------------------------------------
        persona_info: Dict[str, Any] = {"name": None, "source": None}
        persona_prompt = ""

        if agent.use_custom_persona and agent.custom_persona_prompt:
            persona_prompt = agent.custom_persona_prompt
            persona_info = {"name": "Custom Persona", "source": "custom"}
        elif agent.persona_id:
            from core.models.personas import Persona
            persona = db.query(Persona).filter(Persona.id == agent.persona_id).first()
            if persona:
                persona_prompt = persona.system_prompt or ""
                temperature = persona.suggested_temperature or temperature
                persona_info = {
                    "name": persona.name,
                    "slug": persona.slug,
                    "source": persona.source,
                    "category": persona.category,
                    "voice_description": persona.voice_description,
                }

        # ------------------------------------------------------------------
        # 4. Load assigned plugin context via PluginContextService
        # ------------------------------------------------------------------
        from core.services.plugin_context_service import PluginContextService

        plugin_svc = PluginContextService(db)
        plugin_rows = plugin_svc.get_assigned_plugins(agent_id)
        plugins_loaded: List[str] = [p.slug for _, p in plugin_rows]

        tier1 = plugin_svc.build_tier1_summary(plugin_rows)
        tier2 = await plugin_svc.build_tier2_content(plugin_rows)

        # ------------------------------------------------------------------
        # 5. Assemble system_prompt
        # ------------------------------------------------------------------
        sections: List[str] = []

        if persona_prompt:
            sections.append(persona_prompt)

        if tier1:
            sections.append(tier1)
        if tier2:
            sections.append(tier2)

        system_prompt = "\n\n".join(sections)

        # ------------------------------------------------------------------
        # 6. Include tool definitions from agent_app_assignments
        # ------------------------------------------------------------------
        tools: List[Dict[str, Any]] = []
        assignments = (
            db.query(AgentAppAssignment)
            .filter(AgentAppAssignment.agent_id == agent_id, AgentAppAssignment.is_active == True)
            .all()
        )
        if assignments:
            app_names = [a.app_name.upper() for a in assignments if a.app_name]
            cache = {
                a.app_name: a
                for a in db.query(ComposioAppCache).filter(ComposioAppCache.app_name.in_(app_names)).all()
            }
            for assignment in assignments:
                app_name = (assignment.app_name or "").upper()
                cached = cache.get(app_name)
                tools.append({
                    "id": cached.id if cached else None,
                    "name": app_name,
                    "description": (cached.description if cached else "") or "",
                    "provider": "Composio" if cached else None,
                    "category": ((cached.categories or [None])[0] if cached else None),
                    "configuration": assignment.config or {},
                })

        # ------------------------------------------------------------------
        # 7. Estimate token count (rough: len / 4)
        # ------------------------------------------------------------------
        token_estimate = len(system_prompt) // 4

        return AssembledContextOut(
            agent_id=agent.id,
            model=model_id,
            temperature=temperature,
            system_prompt=system_prompt,
            persona=persona_info,
            plugins_loaded=plugins_loaded,
            tools=tools,
            token_estimate=token_estimate,
        ).model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error assembling context for agent %s: %s", agent_id, e)
        raise HTTPException(status_code=500, detail="Failed to assemble agent context")
