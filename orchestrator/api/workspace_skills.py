"""
Workspace Skill Enablement API Endpoints (PRD-71)
==================================================

Provides REST APIs for workspace owners to enable/disable marketplace skills:
- List enabled skills for a workspace
- Enable a skill for a workspace
- Disable a skill for a workspace (cascades to agent assignments)
- List all available marketplace skills (for browsing)

Mirrors workspace_plugins.py — same pattern for skills.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workspaces", tags=["Workspace Skills"])


def _is_admin(ctx: RequestContext) -> bool:
    """System admins can operate on any workspace (mirrors admin_plugins.py)."""
    return getattr(ctx.user, "system_role", "user") == "admin"


def _assert_workspace_access(ctx: RequestContext, workspace_id: UUID) -> None:
    """Allow own workspace or system admin; otherwise 403."""
    if ctx.workspace_id != workspace_id and not _is_admin(ctx):
        raise HTTPException(status_code=403, detail="Access denied: workspace mismatch")


# ===================================================================
# Pydantic Models
# ===================================================================

class EnableSkillBody(BaseModel):
    skill_id: int = Field(..., description="ID of the marketplace skill to enable")


class WorkspaceSkillOut(BaseModel):
    skill_id: int
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    skill_version: Optional[str] = None
    tags: Optional[List[str]] = None
    estimated_tokens: int = 0
    skill_source: Optional[str] = None
    enabled_at: Optional[str] = None

    class Config:
        from_attributes = True


class MarketplaceSkillOut(BaseModel):
    """Skill available in the marketplace (workspace_id IS NULL)."""
    id: int
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    skill_version: Optional[str] = None
    tags: Optional[List[str]] = None
    estimated_tokens: int = 0
    skill_source: Optional[str] = None
    is_enabled: bool = False

    class Config:
        from_attributes = True


# ===================================================================
# Endpoints
# ===================================================================

@router.get("/{workspace_id}/skills", response_model=None)
async def list_workspace_skills(
    workspace_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List skills enabled for a workspace."""
    _assert_workspace_access(ctx, workspace_id)

    try:
        from core.models.core import Skill
        from core.models.marketplace_plugins import WorkspaceEnabledSkill

        rows = (
            db.query(WorkspaceEnabledSkill, Skill)
            .join(Skill, Skill.id == WorkspaceEnabledSkill.skill_id)
            .filter(WorkspaceEnabledSkill.workspace_id == workspace_id)
            .all()
        )

        items = []
        for wes, skill in rows:
            tokens = len(skill.prompt_template or "") // 4 if skill.prompt_template else 0
            items.append(WorkspaceSkillOut(
                skill_id=skill.id,
                name=skill.name,
                description=skill.description,
                category=skill.category,
                skill_version=skill.skill_version,
                tags=skill.tags,
                estimated_tokens=tokens,
                skill_source=skill.skill_source,
                enabled_at=wes.enabled_at.isoformat() if wes.enabled_at else None,
            ))

        return {"items": [item.model_dump() for item in items]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing workspace skills for %s: %s", workspace_id, e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{workspace_id}/skills/available", response_model=None)
async def list_available_skills(
    workspace_id: UUID,
    q: Optional[str] = Query(None, description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List marketplace skills available to enable (workspace_id IS NULL, is_active=True)."""
    _assert_workspace_access(ctx, workspace_id)

    try:
        from core.models.core import Skill
        from core.models.marketplace_plugins import WorkspaceEnabledSkill

        # Get all marketplace/global skills
        query = db.query(Skill).filter(
            Skill.workspace_id.is_(None),
            Skill.is_active == True,
        )

        if q:
            search = f"%{q}%"
            query = query.filter(
                (Skill.name.ilike(search)) | (Skill.description.ilike(search))
            )

        if category:
            query = query.filter(Skill.category == category)

        all_skills = query.order_by(Skill.name).all()

        # Check which are already enabled for this workspace
        enabled_ids = set(
            row[0] for row in
            db.query(WorkspaceEnabledSkill.skill_id)
            .filter(WorkspaceEnabledSkill.workspace_id == workspace_id)
            .all()
        )

        items = []
        for skill in all_skills:
            tokens = len(skill.prompt_template or "") // 4 if skill.prompt_template else 0
            items.append(MarketplaceSkillOut(
                id=skill.id,
                name=skill.name,
                description=skill.description,
                category=skill.category,
                skill_version=skill.skill_version,
                tags=skill.tags,
                estimated_tokens=tokens,
                skill_source=skill.skill_source,
                is_enabled=skill.id in enabled_ids,
            ))

        return {
            "items": [item.model_dump() for item in items],
            "total": len(items),
            "enabled_count": sum(1 for i in items if i.is_enabled),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing available skills for %s: %s", workspace_id, e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{workspace_id}/skills", status_code=201)
async def enable_skill(
    workspace_id: UUID,
    body: EnableSkillBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Enable a marketplace skill for a workspace."""
    _assert_workspace_access(ctx, workspace_id)

    try:
        from core.models.core import Skill
        from core.models.marketplace_plugins import WorkspaceEnabledSkill

        # Validate skill exists, is marketplace/global, and is active
        skill = db.query(Skill).filter(
            Skill.id == body.skill_id,
            Skill.workspace_id.is_(None),
            Skill.is_active == True,
        ).first()

        if not skill:
            raise HTTPException(
                status_code=404,
                detail="Skill not found or not available in marketplace",
            )

        # Check if already enabled
        existing = db.query(WorkspaceEnabledSkill).filter(
            WorkspaceEnabledSkill.workspace_id == workspace_id,
            WorkspaceEnabledSkill.skill_id == body.skill_id,
        ).first()

        if existing:
            raise HTTPException(status_code=409, detail="Skill is already enabled for this workspace")

        # Resolve user id
        enabled_by = None
        user_id_str = getattr(ctx.user, "id", None)
        if user_id_str:
            try:
                enabled_by = int(user_id_str)
            except (ValueError, TypeError):
                enabled_by = None

        junction = WorkspaceEnabledSkill(
            workspace_id=workspace_id,
            skill_id=body.skill_id,
            enabled_by=enabled_by,
        )
        db.add(junction)
        db.commit()

        return {
            "success": True,
            "message": f"Skill '{skill.name}' enabled for workspace",
            "skill_id": skill.id,
            "workspace_id": str(workspace_id),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error enabling skill for workspace %s: %s", workspace_id, e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{workspace_id}/skills/{skill_id}")
async def disable_skill(
    workspace_id: UUID,
    skill_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Disable a skill for a workspace. Also removes agent assignments."""
    _assert_workspace_access(ctx, workspace_id)

    try:
        from core.models.core import Agent, agent_skills_table
        from core.models.marketplace_plugins import WorkspaceEnabledSkill

        junction = db.query(WorkspaceEnabledSkill).filter(
            WorkspaceEnabledSkill.workspace_id == workspace_id,
            WorkspaceEnabledSkill.skill_id == skill_id,
        ).first()

        if not junction:
            raise HTTPException(status_code=404, detail="Skill is not enabled for this workspace")

        # Remove agent assignments for this skill from agents in this workspace
        workspace_agent_ids = [
            a.id for a in db.query(Agent.id).filter(
                Agent.workspace_id == workspace_id,
            ).all()
        ]

        removed_agent_count = 0
        if workspace_agent_ids:
            from sqlalchemy import and_, delete
            stmt = delete(agent_skills_table).where(
                and_(
                    agent_skills_table.c.agent_id.in_(workspace_agent_ids),
                    agent_skills_table.c.skill_id == skill_id,
                )
            )
            result = db.execute(stmt)
            removed_agent_count = result.rowcount

        db.delete(junction)
        db.commit()

        return {
            "success": True,
            "message": "Skill disabled for workspace",
            "skill_id": skill_id,
            "workspace_id": str(workspace_id),
            "agents_unassigned": removed_agent_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error disabling skill %s for workspace %s: %s", skill_id, workspace_id, e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")
