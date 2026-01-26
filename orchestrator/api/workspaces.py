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
    if ctx.workspace_id == dev_fallback:
        ids = (
            db.query(Agent.workspace_id)
            .filter(Agent.workspace_id.isnot(None))
            .distinct()
            .limit(2)
            .all()
        )
        distinct_ids = [row[0] for row in ids if row and row[0]]
        if len(distinct_ids) == 1:
            effective_workspace_id = distinct_ids[0]

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

