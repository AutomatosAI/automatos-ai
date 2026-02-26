"""
Workspace File & Exec API
============================
PRD-66: Code Viewer Widget + Interactive Terminal

Proxies file-browsing and command execution requests to the workspace
worker's HTTP server, which has the persistent volume mounted.

  GET  /api/workspaces/{workspace_id}/files          — directory listing
  GET  /api/workspaces/{workspace_id}/files/content   — file content
  POST /api/workspaces/{workspace_id}/exec            — run command
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.workspace_client import WorkspaceClient

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/workspaces/{workspace_id}",
    tags=["workspace-files"],
)


# ---------------------------------------------------------------------------
# GET /api/workspaces/{workspace_id}/files — directory listing (proxied)
# ---------------------------------------------------------------------------
@router.get("/files")
async def list_files(
    workspace_id: str,
    path: str = Query(".", description="Relative path inside workspace"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """List files and directories at *path* inside the workspace."""
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    client = WorkspaceClient(workspace_id)
    result = await client.list_dir(path)

    if result.get("success") is False:
        status = result.get("status_code", 503)
        raise HTTPException(status_code=status, detail=result["error"])

    return result


# ---------------------------------------------------------------------------
# GET /api/workspaces/{workspace_id}/files/content — file content (proxied)
# ---------------------------------------------------------------------------
@router.get("/files/content")
async def get_file_content(
    workspace_id: str,
    path: str = Query(..., description="Relative file path inside workspace"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Return the text content of a file for the code viewer."""
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    client = WorkspaceClient(workspace_id)
    result = await client.read_file(path)

    if result.get("success") is False:
        status = result.get("status_code", 503)
        raise HTTPException(status_code=status, detail=result["error"])

    return result


# ---------------------------------------------------------------------------
# POST /api/workspaces/{workspace_id}/exec — run command (proxied)
# ---------------------------------------------------------------------------
class ExecRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=4096)
    cwd: Optional[str] = None
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
