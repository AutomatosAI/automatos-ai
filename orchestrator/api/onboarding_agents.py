"""
Onboarding Agents API
=====================

Admin-only endpoints for managing the hidden system agents used during
Mission Zero onboarding (VOYAGER, BLUEPRINT, SCRIBE, FORGE).

Exposed in Settings > Onboarding Agents tab.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.auth.hybrid import get_request_context_hybrid, RequestContext
from core.database.database import get_db
from core.models.core import Agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/settings/onboarding-agents", tags=["onboarding-agents"])

ONBOARDING_TAG = "onboarding"


def _require_admin(ctx: RequestContext):
    """Only admins can manage onboarding agents."""
    if not ctx.user or ctx.user.system_role not in ("admin", "super_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")


def _get_onboarding_agents(db: Session):
    """Fetch all onboarding agents by tag."""
    return (
        db.query(Agent)
        .filter(
            Agent.is_system_agent.is_(True),
            Agent.required_role == "onboarding",
            Agent.tags.contains(["onboarding"]),
        )
        .order_by(Agent.id)
        .all()
    )


class OnboardingAgentUpdate(BaseModel):
    model_id: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    custom_persona_prompt: Optional[str] = None
    status: Optional[str] = None  # "active" | "inactive"


class OnboardingAgentResponse(BaseModel):
    id: int
    slug: str
    name: str
    description: str
    status: str
    job_title: Optional[str]
    team: Optional[str]
    model_id: str
    temperature: float
    max_tokens: int
    custom_persona_prompt: str
    tags: list
    configuration: Optional[dict]

    class Config:
        from_attributes = True


@router.get("")
async def list_onboarding_agents(
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """List all onboarding agents for the Settings tab."""
    _require_admin(ctx)

    agents = _get_onboarding_agents(db)
    result = []
    for a in agents:
        mc = a.model_config or {}
        result.append({
            "id": a.id,
            "slug": a.slug,
            "name": a.name,
            "description": a.description,
            "status": a.status,
            "job_title": a.job_title,
            "team": a.team,
            "model_id": mc.get("model_id", ""),
            "provider": mc.get("provider", "openrouter"),
            "temperature": mc.get("temperature", 0.7),
            "max_tokens": mc.get("max_tokens", 8000),
            "custom_persona_prompt": a.custom_persona_prompt or "",
            "tags": a.tags or [],
            "configuration": a.configuration or {},
        })

    return {"agents": result}


@router.put("/{slug}")
async def update_onboarding_agent(
    slug: str,
    payload: OnboardingAgentUpdate,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Update an onboarding agent's model, persona, or status."""
    _require_admin(ctx)

    agent = (
        db.query(Agent)
        .filter(
            Agent.slug == slug,
            Agent.is_system_agent.is_(True),
            Agent.required_role == "onboarding",
        )
        .first()
    )
    if not agent:
        raise HTTPException(status_code=404, detail=f"Onboarding agent '{slug}' not found")

    mc = dict(agent.model_config or {})

    if payload.model_id is not None:
        mc["model_id"] = payload.model_id
    if payload.temperature is not None:
        mc["temperature"] = payload.temperature
    if payload.max_tokens is not None:
        mc["max_tokens"] = payload.max_tokens
    if payload.custom_persona_prompt is not None:
        agent.custom_persona_prompt = payload.custom_persona_prompt
    if payload.status is not None:
        agent.status = payload.status

    agent.model_config = mc
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(agent, "model_config")

    db.commit()
    logger.info("Updated onboarding agent %s", slug)

    return {"status": "updated", "slug": slug}
