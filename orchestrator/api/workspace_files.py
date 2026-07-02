"""
Workspace File & Exec API
============================
PRD-66: Code Viewer Widget + Interactive Terminal

Proxies file-browsing and command execution requests to the workspace
worker's HTTP server, which has the persistent volume mounted.

  GET    /api/workspaces/{workspace_id}/files            — directory listing
  GET    /api/workspaces/{workspace_id}/files/content    — file content
  POST   /api/workspaces/{workspace_id}/exec             — run command
  POST   /api/workspaces/{workspace_id}/canvas/sessions  — start/resume SDK session (PRD-170 S1)
  GET    /api/workspaces/{workspace_id}/canvas/sessions  — session status
  DELETE /api/workspaces/{workspace_id}/canvas/sessions  — stop session
  GET    /api/workspaces/{workspace_id}/canvas/events    — SSE event stream (PRD-170 S3)
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.graph_storage import DbWorkspaceClient
from core.workspace_client import WorkspaceClient

logger = logging.getLogger(__name__)

# PRD-170 S3: the worker publishes canvas session events to this per-workspace
# Redis channel (mirror of services/workspace-worker/main.py CANVAS_EVENTS_CHANNEL);
# the SSE proxy below subscribes and re-emits them to the browser.
_CANVAS_EVENTS_CHANNEL = "workspace:ws:{workspace_id}:canvas:events"

router = APIRouter(
    prefix="/api/workspaces/{workspace_id}",
    tags=["workspace-files"],
)


# Paths under these prefixes are stored in Postgres via DbWorkspaceClient
# (see PRD-130: graph artefacts persist to workspace_graphs instead of the
# workspace-worker filesystem because wizard-created workspaces do not have
# a worker container provisioned). Everything else still proxies to the
# worker over HTTP as before.
_DB_BACKED_PREFIXES = ("graph/", "graph")


def _select_client(workspace_id: str, path: str):
    """Route a (workspace, path) pair to the right storage backend."""
    normalised = (path or "").lstrip("/")
    if normalised == "graph" or normalised.startswith("graph/"):
        return DbWorkspaceClient(workspace_id)
    return WorkspaceClient(workspace_id)


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

    client = _select_client(workspace_id, path)
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

    client = _select_client(workspace_id, path)
    result = await client.read_file(path)

    if result.get("success") is False:
        status = result.get("status_code", 503)
        raise HTTPException(status_code=status, detail=result["error"])

    return result


# ---------------------------------------------------------------------------
# GET /api/workspaces/{workspace_id}/files/raw — raw bytes (proxied)
# ---------------------------------------------------------------------------
@router.get("/files/raw")
async def get_file_raw(
    workspace_id: str,
    path: str = Query(..., description="Relative file path inside workspace"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Return raw binary bytes of a file (for PDF/DOCX/XLSX/image preview)."""
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    client = WorkspaceClient(workspace_id)
    result = await client.download_file(path)

    if result.get("success") is False:
        status = result.get("status_code", 503)
        raise HTTPException(status_code=status, detail=result["error"])

    filename = result.get("filename") or path.rsplit("/", 1)[-1]
    return Response(
        content=result["content"],
        media_type=result.get("content_type", "application/octet-stream"),
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# PUT /api/workspaces/{workspace_id}/files/content — write file (proxied)
# ---------------------------------------------------------------------------
class WriteFileRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=1024, description="Relative file path inside workspace")
    content: str = Field(..., description="File content to write")


@router.put("/files/content")
async def write_file_content(
    workspace_id: str,
    body: WriteFileRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Write or create a file in the workspace."""
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    client = WorkspaceClient(workspace_id)
    result = await client.write_file(body.path, body.content)

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


# ---------------------------------------------------------------------------
# Canvas SDK session proxy (PRD-170 S1)
# ---------------------------------------------------------------------------
# One active headless Claude Agent SDK session per workspace, running in the
# worker container (services/workspace-worker/canvas_session_service.py).
# State + transcript persist on the worker volume; the session is confined to
# its workspace mount server-side. These routes only proxy + enforce tenancy.


@router.post("/canvas/sessions")
async def start_canvas_session(
    workspace_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Start (or resume from volume state) the workspace's canvas session."""
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    client = WorkspaceClient(workspace_id)
    result = await client.canvas_session_start()

    if result.get("success") is False:
        status = result.get("status_code", 503)
        raise HTTPException(status_code=status, detail=result["error"])

    return result


@router.get("/canvas/sessions")
async def get_canvas_session_status(
    workspace_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Return the status of the workspace's canvas session."""
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    client = WorkspaceClient(workspace_id)
    result = await client.canvas_session_status()

    if result.get("success") is False:
        status = result.get("status_code", 503)
        raise HTTPException(status_code=status, detail=result["error"])

    return result


@router.delete("/canvas/sessions")
async def stop_canvas_session(
    workspace_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Stop the workspace's canvas session."""
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    client = WorkspaceClient(workspace_id)
    result = await client.canvas_session_stop()

    if result.get("success") is False:
        status = result.get("status_code", 503)
        raise HTTPException(status_code=status, detail=result["error"])

    return result


# ---------------------------------------------------------------------------
# GET /api/workspaces/{workspace_id}/canvas/events — SSE stream (PRD-170 S3)
# ---------------------------------------------------------------------------
# Subscribes to the worker's per-workspace canvas Redis channel and re-emits
# each versioned canvas event to the browser as Server-Sent Events. The session
# panel renders streaming turns; the file tree live-refreshes on file.edit
# events. Tenancy is enforced here (RequestContext) exactly like the file/exec
# routes — the browser never touches Redis directly.
async def _canvas_event_stream(workspace_id: str, request: Request):
    from core.redis.client import get_redis_client

    redis_client = get_redis_client()
    if redis_client is None:
        # Redis is an optional service; without it there is no live stream.
        yield 'event: error\ndata: {"error": "event stream unavailable"}\n\n'
        return

    channel = _CANVAS_EVENTS_CHANNEL.format(workspace_id=workspace_id)
    redis_async, pubsub = await redis_client.get_async_pubsub(channel)
    try:
        # Opening comment flushes headers so the client's EventSource opens.
        yield ": canvas stream open\n\n"
        while True:
            if await request.is_disconnected():
                break
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=1.0
            )
            if message is None:
                # Heartbeat comment keeps the connection warm through proxies.
                yield ": ping\n\n"
                continue
            payload = message.get("data")
            if not isinstance(payload, str):
                continue
            # payload is already a JSON canvas-event envelope from the worker.
            yield f"data: {payload}\n\n"
    except asyncio.CancelledError:
        raise
    finally:
        try:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
            await redis_async.aclose()
        except Exception as exc:  # noqa: BLE001 — teardown must not raise
            logger.warning("Canvas SSE teardown error: %s", exc)


@router.get("/canvas/events")
async def stream_canvas_events(
    workspace_id: str,
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Server-Sent Events stream of the workspace's canvas session events."""
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    return StreamingResponse(
        _canvas_event_stream(workspace_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
        },
    )
