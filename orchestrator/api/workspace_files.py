"""
Workspace File Browser API
============================
PRD-66 Phase 1: Code Viewer Widget

Proxies file-browsing requests to the workspace worker's HTTP server,
which has the persistent volume mounted. The backend keeps auth/ownership
checks; the worker does the actual filesystem I/O.

  GET /api/workspaces/{workspace_id}/files          — directory listing
  GET /api/workspaces/{workspace_id}/files/content   — file content
"""

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from config import config
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/workspaces/{workspace_id}/files",
    tags=["workspace-files"],
)

# Shared httpx client (reused across requests for connection pooling)
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=15.0)
    return _http_client


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

    worker_url = f"{config.WORKER_INTERNAL_URL}/workspaces/{workspace_id}/files"
    client = _get_http_client()

    try:
        resp = await client.get(worker_url, params={"path": path})
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Workspace worker is unreachable. Files are only available when the worker service is running.",
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Workspace worker request timed out")

    if resp.status_code != 200:
        data = resp.json()
        raise HTTPException(status_code=resp.status_code, detail=data.get("error", "Worker error"))

    return resp.json()


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

    worker_url = f"{config.WORKER_INTERNAL_URL}/workspaces/{workspace_id}/files/content"
    client = _get_http_client()

    try:
        resp = await client.get(worker_url, params={"path": path})
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Workspace worker is unreachable. Files are only available when the worker service is running.",
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Workspace worker request timed out")

    if resp.status_code != 200:
        data = resp.json()
        raise HTTPException(status_code=resp.status_code, detail=data.get("error", "Worker error"))

    return resp.json()
