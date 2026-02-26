"""
Workspace File Browser API
============================
PRD-66 Phase 1: Code Viewer Widget

Proxies file-browsing requests to the workspace worker's HTTP server,
which has the persistent volume mounted. The backend keeps auth/ownership
checks; the worker does the actual filesystem I/O.

Uses the shared WorkspaceClient (core/workspace_client.py) to avoid
duplicating httpx connection logic.

  GET /api/workspaces/{workspace_id}/files          — directory listing
  GET /api/workspaces/{workspace_id}/files/content   — file content
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.workspace_client import WorkspaceClient

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/workspaces/{workspace_id}/files",
    tags=["workspace-files"],
)


# ---------------------------------------------------------------------------
# GET /api/workspaces/{workspace_id}/files — directory listing (proxied)
# ---------------------------------------------------------------------------
@router.get("")
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
@router.get("/content")
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
