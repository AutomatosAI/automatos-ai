"""PRD-158 S1 — Teams API.

List + create the workspace's teams. Writes normalize through the single
``core.team_access`` helper, so 'Support'/'support' resolve to one team.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.team_access import get_or_create_team, list_teams, normalize_team

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/teams", tags=["teams"])


class TeamResponse(BaseModel):
    id: int
    name: str
    normalized_name: str

    class Config:
        from_attributes = True


class CreateTeamRequest(BaseModel):
    name: str


@router.get("", response_model=List[TeamResponse])
async def list_workspace_teams(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """All teams in the caller's workspace (canonical-name order)."""
    return list_teams(db, ctx.workspace_id)


@router.post("", response_model=TeamResponse, status_code=201)
async def create_workspace_team(
    body: CreateTeamRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create (or return the existing) team for the normalized name."""
    if not normalize_team(body.name or ""):
        raise HTTPException(status_code=400, detail="Team name cannot be empty")
    team = get_or_create_team(db, ctx.workspace_id, body.name)
    db.commit()
    db.refresh(team)
    return team
