"""
Persona API Endpoints (PRD-42: US-013)
======================================

Provides REST APIs for persona CRUD operations and agent persona assignment:
- List available personas (global + workspace-custom)
- Get persona details
- Create/update/delete workspace personas
- Set/get agent persona
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import asc, or_
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Personas"])


# ===================================================================
# Pydantic Models
# ===================================================================

class PersonaOut(BaseModel):
    id: str
    slug: str
    name: str
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    voice_description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    suggested_temperature: float = 0.7
    suggested_models: Optional[List[str]] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    scope: str = "global"
    workspace_id: Optional[str] = None
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class PersonaSummaryOut(BaseModel):
    id: str
    slug: str
    name: str
    description: Optional[str] = None
    voice_description: Optional[str] = None
    category: Optional[str] = None
    suggested_temperature: float = 0.7
    scope: str = "global"
    is_active: bool = True

    class Config:
        from_attributes = True


class CreatePersonaBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    voice_description: Optional[str] = Field(None, max_length=500)
    category: Optional[str] = Field(None, max_length=100)
    suggested_temperature: float = Field(0.7, ge=0.0, le=2.0)


class UpdatePersonaBody(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    voice_description: Optional[str] = Field(None, max_length=500)
    category: Optional[str] = Field(None, max_length=100)
    suggested_temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


class SetAgentPersonaBody(BaseModel):
    persona_id: Optional[UUID] = None
    custom_prompt: Optional[str] = None
    use_custom: bool = False


class AgentPersonaOut(BaseModel):
    agent_id: int
    persona_id: Optional[str] = None
    persona_name: Optional[str] = None
    persona_slug: Optional[str] = None
    system_prompt: Optional[str] = None
    voice_description: Optional[str] = None
    custom_persona_prompt: Optional[str] = None
    use_custom_persona: bool = False


# ===================================================================
# Helper
# ===================================================================

def _slugify(name: str) -> str:
    """Generate a slug from a name."""
    import re
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "-", slug)
    return slug[:100]


def _persona_to_out(p) -> PersonaOut:
    """Convert a Persona model instance to PersonaOut."""
    return PersonaOut(
        id=str(p.id),
        slug=p.slug,
        name=p.name,
        description=p.description,
        system_prompt=p.system_prompt,
        voice_description=p.voice_description,
        category=p.category,
        tags=p.tags or [],
        suggested_temperature=p.suggested_temperature or 0.7,
        suggested_models=p.suggested_models or [],
        source=p.source,
        source_url=p.source_url,
        scope=p.scope or "global",
        workspace_id=str(p.workspace_id) if p.workspace_id else None,
        is_active=p.is_active if p.is_active is not None else True,
        created_at=p.created_at.isoformat() if p.created_at else None,
        updated_at=p.updated_at.isoformat() if p.updated_at else None,
    )


def _persona_to_summary(p) -> PersonaSummaryOut:
    """Convert a Persona model instance to PersonaSummaryOut."""
    return PersonaSummaryOut(
        id=str(p.id),
        slug=p.slug,
        name=p.name,
        description=p.description,
        voice_description=p.voice_description,
        category=p.category,
        suggested_temperature=p.suggested_temperature or 0.7,
        scope=p.scope or "global",
        is_active=p.is_active if p.is_active is not None else True,
    )


# ===================================================================
# Endpoints
# ===================================================================

@router.get("/personas", response_model=None)
async def list_personas(
    category: Optional[str] = Query(None, description="Filter by category"),
    scope: Optional[str] = Query(None, description="Filter by scope: all, global, workspace"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List available personas (global + workspace-custom filtered by user's workspace)."""
    try:
        from core.models.personas import Persona

        query = db.query(Persona).filter(Persona.is_active == True)

        # Scope filtering
        if scope == "global":
            query = query.filter(Persona.scope == "global")
        elif scope == "workspace":
            query = query.filter(
                Persona.scope == "workspace",
                Persona.workspace_id == ctx.workspace_id,
            )
        else:
            # Default: show global + user's workspace personas
            query = query.filter(
                or_(
                    Persona.scope == "global",
                    Persona.workspace_id == ctx.workspace_id,
                )
            )

        # Category filter
        if category:
            query = query.filter(Persona.category == category)

        query = query.order_by(asc(Persona.category), asc(Persona.name))
        personas = query.all()

        return {
            "items": [_persona_to_summary(p).model_dump() for p in personas],
            "total": len(personas),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing personas: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to list personas: {e}")


@router.get("/personas/{persona_id}", response_model=PersonaOut)
async def get_persona(
    persona_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get persona details including full system_prompt."""
    try:
        from core.models.personas import Persona

        persona = db.query(Persona).filter(
            Persona.id == persona_id,
            Persona.is_active == True,
        ).first()

        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")

        # Workspace personas only visible to that workspace
        if persona.scope == "workspace" and persona.workspace_id != ctx.workspace_id:
            raise HTTPException(status_code=403, detail="Access denied: workspace mismatch")

        return _persona_to_out(persona)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching persona %s: %s", persona_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch persona: {e}")


@router.post("/workspaces/{workspace_id}/personas", status_code=201, response_model=PersonaOut)
async def create_workspace_persona(
    workspace_id: UUID,
    body: CreatePersonaBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create a custom persona for a workspace."""
    if ctx.workspace_id != workspace_id:
        raise HTTPException(status_code=403, detail="Access denied: workspace mismatch")

    try:
        from core.models.personas import Persona

        # Generate slug from name
        slug = _slugify(body.name)
        # Ensure uniqueness by appending short uuid if needed
        existing = db.query(Persona).filter(Persona.slug == slug).first()
        if existing:
            slug = f"{slug}-{uuid4().hex[:8]}"

        persona = Persona(
            slug=slug,
            name=body.name,
            description=body.description,
            system_prompt=body.system_prompt,
            voice_description=body.voice_description,
            category=body.category,
            suggested_temperature=body.suggested_temperature,
            source="workspace",
            scope="workspace",
            workspace_id=workspace_id,
        )
        db.add(persona)
        db.commit()
        db.refresh(persona)

        return _persona_to_out(persona)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating workspace persona: %s", e)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create persona: {e}")


@router.put("/workspaces/{workspace_id}/personas/{persona_id}", response_model=PersonaOut)
async def update_workspace_persona(
    workspace_id: UUID,
    persona_id: UUID,
    body: UpdatePersonaBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update a workspace persona. Only workspace-scoped personas can be updated."""
    if ctx.workspace_id != workspace_id:
        raise HTTPException(status_code=403, detail="Access denied: workspace mismatch")

    try:
        from core.models.personas import Persona

        persona = db.query(Persona).filter(
            Persona.id == persona_id,
            Persona.scope == "workspace",
            Persona.workspace_id == workspace_id,
        ).first()

        if not persona:
            raise HTTPException(
                status_code=404,
                detail="Persona not found or not a workspace persona",
            )

        # Apply updates (only set provided fields)
        if body.name is not None:
            persona.name = body.name
        if body.description is not None:
            persona.description = body.description
        if body.system_prompt is not None:
            persona.system_prompt = body.system_prompt
        if body.voice_description is not None:
            persona.voice_description = body.voice_description
        if body.category is not None:
            persona.category = body.category
        if body.suggested_temperature is not None:
            persona.suggested_temperature = body.suggested_temperature

        db.commit()
        db.refresh(persona)

        return _persona_to_out(persona)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating persona %s: %s", persona_id, e)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update persona: {e}")


@router.delete("/workspaces/{workspace_id}/personas/{persona_id}")
async def delete_workspace_persona(
    workspace_id: UUID,
    persona_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Delete a workspace persona. Only workspace-scoped personas can be deleted."""
    if ctx.workspace_id != workspace_id:
        raise HTTPException(status_code=403, detail="Access denied: workspace mismatch")

    try:
        from core.models.personas import Persona

        persona = db.query(Persona).filter(
            Persona.id == persona_id,
            Persona.scope == "workspace",
            Persona.workspace_id == workspace_id,
        ).first()

        if not persona:
            raise HTTPException(
                status_code=404,
                detail="Persona not found or not a workspace persona",
            )

        # Clear persona references from agents in this workspace
        from core.models.core import Agent

        db.query(Agent).filter(
            Agent.workspace_id == workspace_id,
            Agent.persona_id == persona_id,
        ).update(
            {"persona_id": None, "use_custom_persona": False},
            synchronize_session="fetch",
        )

        db.delete(persona)
        db.commit()

        return {
            "success": True,
            "message": f"Persona '{persona.name}' deleted",
            "persona_id": str(persona_id),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting persona %s: %s", persona_id, e)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete persona: {e}")


@router.put("/agents/{agent_id}/persona", response_model=AgentPersonaOut)
async def set_agent_persona(
    agent_id: int,
    body: SetAgentPersonaBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Set agent persona. Can select a predefined persona, use a custom prompt, or clear."""
    try:
        from core.models.core import Agent

        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        if agent.workspace_id and agent.workspace_id != ctx.workspace_id:
            raise HTTPException(status_code=403, detail="Access denied: workspace mismatch")

        # Validate persona_id if provided
        if body.persona_id:
            from core.models.personas import Persona

            persona = db.query(Persona).filter(
                Persona.id == body.persona_id,
                Persona.is_active == True,
            ).first()

            if not persona:
                raise HTTPException(status_code=404, detail="Persona not found")

            # Workspace personas only assignable to same workspace agents
            if persona.scope == "workspace" and persona.workspace_id != ctx.workspace_id:
                raise HTTPException(status_code=403, detail="Cannot assign workspace persona from another workspace")

        agent.persona_id = body.persona_id
        agent.custom_persona_prompt = body.custom_prompt
        agent.use_custom_persona = body.use_custom

        db.commit()
        db.refresh(agent)

        # Build response
        persona_name = None
        persona_slug = None
        system_prompt = None
        voice_description = None

        if agent.persona_id:
            from core.models.personas import Persona
            p = db.query(Persona).filter(Persona.id == agent.persona_id).first()
            if p:
                persona_name = p.name
                persona_slug = p.slug
                system_prompt = p.system_prompt
                voice_description = p.voice_description

        return AgentPersonaOut(
            agent_id=agent.id,
            persona_id=str(agent.persona_id) if agent.persona_id else None,
            persona_name=persona_name,
            persona_slug=persona_slug,
            system_prompt=system_prompt,
            voice_description=voice_description,
            custom_persona_prompt=agent.custom_persona_prompt,
            use_custom_persona=agent.use_custom_persona or False,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error setting persona for agent %s: %s", agent_id, e)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to set agent persona: {e}")


@router.get("/agents/{agent_id}/persona", response_model=AgentPersonaOut)
async def get_agent_persona(
    agent_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get agent's current persona info."""
    try:
        from core.models.core import Agent

        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        if agent.workspace_id and agent.workspace_id != ctx.workspace_id:
            raise HTTPException(status_code=403, detail="Access denied: workspace mismatch")

        persona_name = None
        persona_slug = None
        system_prompt = None
        voice_description = None

        if agent.persona_id:
            from core.models.personas import Persona
            p = db.query(Persona).filter(Persona.id == agent.persona_id).first()
            if p:
                persona_name = p.name
                persona_slug = p.slug
                system_prompt = p.system_prompt
                voice_description = p.voice_description

        return AgentPersonaOut(
            agent_id=agent.id,
            persona_id=str(agent.persona_id) if agent.persona_id else None,
            persona_name=persona_name,
            persona_slug=persona_slug,
            system_prompt=system_prompt,
            voice_description=voice_description,
            custom_persona_prompt=agent.custom_persona_prompt,
            use_custom_persona=agent.use_custom_persona or False,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching persona for agent %s: %s", agent_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch agent persona: {e}")
