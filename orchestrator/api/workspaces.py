"""
PRD-37: Workspace context endpoints.

The frontend expects `GET /api/workspaces/current` to return the active workspace.
In this codebase, most resources are filtered by `workspace_id`, and the auth
dependency (`get_request_context_hybrid`) provides a request-scoped workspace UUID.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models import Agent
from core.models.workspaces import Workspace

from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])


@router.get("/current")
async def get_current_workspace(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Return the currently active workspace.

    The auth dependency auto-provisions a personal workspace for new Clerk users,
    so ctx.workspace_id should always point to a valid workspace.

    Returns `is_new_workspace: true` when the workspace has no agents yet,
    signalling the frontend to trigger the onboarding flow.
    """
    workspace = db.query(Workspace).get(ctx.workspace_id)

    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Detect brand-new workspace: no agents created yet
    agent_count = db.query(Agent).filter(Agent.workspace_id == workspace.id).count()

    return {
        "id": str(workspace.id),
        "name": workspace.name,
        "slug": workspace.slug,
        "plan": workspace.plan,
        "role": ctx.user.role,
        "plan_limits": workspace.plan_limits or {},
        "is_new_workspace": agent_count == 0,
    }

