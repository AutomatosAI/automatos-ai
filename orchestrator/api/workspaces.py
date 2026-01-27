"""
PRD-37: Workspace context endpoints.

The frontend expects `GET /api/workspaces/current` to return the active workspace.
In this codebase, most resources are filtered by `workspace_id`, and the auth
dependency (`get_request_context_hybrid`) provides a request-scoped workspace UUID.
"""

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models import Agent
from core.models.workspaces import Workspace
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

    For now, we model the workspace as the request context's `workspace_id`.
    The frontend uses this to set `last_active_workspace` which is then sent as
    `X-Workspace-ID` on subsequent API calls.
    """

    # If the request context is using the "dev fallback" workspace UUID, try to
    # discover an existing workspace_id from the database so the UI doesn't
    # accidentally pin itself to an empty workspace.
    effective_workspace_id: UUID = ctx.workspace_id
    dev_fallback = UUID("00000000-0000-0000-0000-000000000001")
    
    workspace = db.query(Workspace).get(effective_workspace_id)
    
    if not workspace and ctx.workspace_id == dev_fallback:
        # Try to find a real workspace if we are in fallback mode
        workspace = db.query(Workspace).filter(Workspace.is_active == True).order_by(Workspace.updated_at.desc()).first()
        if workspace:
            effective_workspace_id = workspace.id

    if not workspace:
        # Should not happen given get_request_context logic, but safe fallback
        return {
            "id": str(effective_workspace_id),
            "name": "Default Workspace",
            "slug": "default",
            "plan": "starter",
            "role": "owner",
            "plan_limits": {
                "max_agents": 100,
                "max_workflows": 100,
                "max_documents": 1000,
                "max_members": 25,
            },
        }

    return {
        "id": str(workspace.id),
        "name": workspace.name,
        "slug": workspace.slug,
        "plan": workspace.plan,
        "role": ctx.user.role, # Role in this workspace
        "plan_limits": workspace.plan_limits or {},
    }

