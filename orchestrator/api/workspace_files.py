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
  POST   /api/workspaces/{workspace_id}/canvas/decision  — approve/deny an edit (PRD-170 S4)
  POST   /api/workspaces/{workspace_id}/canvas/auto-accept — toggle auto-accept (PRD-170 S4)
  GET    /api/workspaces/{workspace_id}/canvas/commit-preview — editable message (PRD-170 S5)
  POST   /api/workspaces/{workspace_id}/canvas/commit    — commit + push branch (PRD-170 S5)
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
from core.auth.workspace_permission import require_workspace_permission
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


@router.put("/files/content", dependencies=[Depends(require_workspace_permission("documents:update"))])
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


@router.post("/exec", dependencies=[Depends(require_workspace_permission("agents:execute"))])
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


@router.post("/canvas/sessions", dependencies=[Depends(require_workspace_permission("agents:execute"))])
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


@router.delete("/canvas/sessions", dependencies=[Depends(require_workspace_permission("agents:execute"))])
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
# Canvas approval loop (PRD-170 S4) — nothing mutating applies without approval
# ---------------------------------------------------------------------------
# A mutating tool call (file edit / bash) pauses the SDK session and surfaces a
# permission.request event over the SSE stream; the UI renders an approval card
# and POSTs the decision here. Approve → the session applies the edit; deny →
# the session is told (PermissionResultDeny message) and reverts. Auto-accept is
# a session-scoped toggle for FILE EDITS only (never bash). Tenancy enforced.


class CanvasDecisionRequest(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=128)
    approved: bool


class CanvasAutoAcceptRequest(BaseModel):
    enabled: bool


@router.post("/canvas/decision", dependencies=[Depends(require_workspace_permission("agents:execute"))])
async def decide_canvas_permission(
    workspace_id: str,
    body: CanvasDecisionRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Approve or deny a pending canvas permission request (S4)."""
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    client = WorkspaceClient(workspace_id)
    result = await client.canvas_session_decide(body.request_id, body.approved)

    if result.get("success") is False:
        status = result.get("status_code", 503)
        raise HTTPException(status_code=status, detail=result["error"])

    return result


@router.post("/canvas/auto-accept", dependencies=[Depends(require_workspace_permission("agents:execute"))])
async def set_canvas_auto_accept(
    workspace_id: str,
    body: CanvasAutoAcceptRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Toggle session-scoped auto-accept for file edits (S4)."""
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    client = WorkspaceClient(workspace_id)
    result = await client.canvas_session_auto_accept(body.enabled)

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


# ---------------------------------------------------------------------------
# POST /api/workspaces/{workspace_id}/canvas/commit — commit + push (PRD-170 S5)
# ---------------------------------------------------------------------------
class CanvasCommitRequest(BaseModel):
    # The EDITABLE commit message the UI shows (pre-filled from the generator).
    message: str = Field(..., min_length=1, max_length=4096)
    # Repo path inside the workspace (e.g. "repos/my-app").
    cwd: str = Field(..., min_length=1, max_length=1024)
    remote: str = Field(default="origin", max_length=128)


def _parse_porcelain_paths(stdout: str) -> list[str]:
    """Extract changed paths from ``git status --porcelain`` output.

    Each line is ``XY <path>`` (or ``XY <old> -> <new>`` for a rename); we take
    the (new) path. Pure so the commit-preview is testable without a container.
    """
    paths: list[str] = []
    for line in (stdout or "").splitlines():
        rest = line[3:] if len(line) > 3 else line.strip()
        if " -> " in rest:
            rest = rest.split(" -> ", 1)[1]
        rest = rest.strip().strip('"')
        if rest:
            paths.append(rest)
    return paths


@router.get("/canvas/commit-preview")
async def preview_canvas_commit(
    workspace_id: str,
    cwd: str = Query(..., min_length=1, max_length=1024),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Return an EDITABLE generated commit message + the changed paths (S5).

    The UI pre-fills the message and the user may rewrite it before committing.
    Deterministic (no LLM) via the tested ``generate_commit_message``.
    """
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    from modules.tools.discovery.canvas_git import (
        CommitContext,
        canvas_branch_name,
        generate_commit_message,
    )

    client = WorkspaceClient(workspace_id)
    status = await client.canvas_session_status()
    session = (status or {}).get("session") or {}
    session_id = session.get("canvas_session_id") or "session"
    branch = canvas_branch_name(session_id)

    git_status = await client.git("status", cwd=cwd, args="--porcelain")
    changed = _parse_porcelain_paths(git_status.get("stdout", "")) if isinstance(git_status, dict) else []
    message = generate_commit_message(CommitContext(changed_paths=changed, branch=branch))

    return {
        "success": True,
        "branch": branch,
        "changed_paths": changed,
        "message": message,
        "has_changes": bool(changed),
    }


@router.post("/canvas/commit", dependencies=[Depends(require_workspace_permission("agents:execute"))])
async def commit_canvas_session(
    workspace_id: str,
    body: CanvasCommitRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Commit the session's work on ``canvas/<session-id>`` and push it.

    Reuses the existing ``workspace_git`` verbs (no new git surface). Push auth
    is a GitHub App installation token (PRD-165) injected server-side; NO token
    material is ever logged or returned — every git result is passed through
    ``redact_token`` first (PRD-154 S12 discipline).
    """
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    from modules.tools.discovery.canvas_git import (
        canvas_branch_name,
        plan_commit_push,
        redact_token,
    )
    from modules.codegraph.github_auth import resolve_github_token

    client = WorkspaceClient(workspace_id)

    # Resolve the session id (branch-per-session) from live/volume state.
    status = await client.canvas_session_status()
    session = (status or {}).get("session") or {}
    session_id = session.get("canvas_session_id")
    if not session_id:
        raise HTTPException(status_code=409, detail="No canvas session to commit")

    branch = canvas_branch_name(session_id)

    token = await resolve_github_token()  # may be None; push then relies on ambient auth

    try:
        steps = plan_commit_push(session_id, body.message, remote=body.remote)
    except ValueError as exc:
        # Invalid remote (allowlist rejects shell metacharacters) → 400, not 500.
        raise HTTPException(status_code=400, detail=str(exc))
    results = []
    for step in steps:
        result = await client.git(step.operation, cwd=body.cwd, args=step.args)
        # Redact the token from any echoed output BEFORE it leaves this process.
        if isinstance(result, dict):
            for key in ("error", "stdout", "stderr", "output"):
                if isinstance(result.get(key), str):
                    result[key] = redact_token(result[key], token)
        results.append({"operation": step.operation, "result": result})
        if isinstance(result, dict) and result.get("success") is False:
            # Stop on first failure; the (redacted) error is returned.
            return {
                "success": False,
                "failed_operation": step.operation,
                "steps": results,
            }

    return {"success": True, "branch": branch, "steps": results}
