"""
Workspace Exec API
====================
Proxies interactive terminal commands to the workspace worker.

  POST /api/workspaces/{workspace_id}/exec
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.workspace_client import WorkspaceClient

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/workspaces/{workspace_id}",
    tags=["workspace-exec"],
)


class ExecRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=4096)
    cwd: str | None = None
    timeout: int = Field(default=120, ge=1, le=300)


@router.post("/exec")
async def exec_command(
    workspace_id: str,
    body: ExecRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Run a shell command in the workspace."""
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    client = WorkspaceClient(workspace_id)
    result = await client.exec_command(
        command=body.command,
        cwd=body.cwd,
        timeout=body.timeout,
    )

    if result.get("success") is False:
        status = result.get("status_code", 503)
        raise HTTPException(status_code=status, detail=result["error"])

    return result
