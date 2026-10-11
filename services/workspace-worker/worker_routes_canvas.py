"""Canvas SDK session routes of the workspace worker (PRD-170 S1/S3).

One headless Claude Agent SDK session per workspace; state + transcript on the
volume (canvas_session_service.py). S3: the pump bridges SDK messages to a
per-workspace Redis channel that the platform proxy re-emits over SSE.
Moved unchanged out of ``WorkspaceWorker._health_server`` (PRD-256 O2b); see
``worker_http``.
"""
from __future__ import annotations

import json

from aiohttp import web

from worker_http import WORKER_HTTP

# The per-workspace Redis channel the platform proxy re-emits over SSE
# (mirrored by orchestrator/api/workspace_files.py _CANVAS_EVENTS_CHANNEL).
CANVAS_EVENTS_CHANNEL = "workspace:ws:{workspace_id}:canvas:events"


def make_canvas_event_sink(worker_self):
    """The sink the canvas manager publishes through: the worker's Redis, once it is up."""

    async def _canvas_event_sink(event: dict) -> None:
        """Publish one canvas event to its per-workspace Redis channel."""
        redis = getattr(worker_self, "_redis", None)
        if redis is None:
            return
        channel = CANVAS_EVENTS_CHANNEL.format(
            workspace_id=event.get("workspace_id", "")
        )
        await redis.publish(channel, json.dumps(event))

    return _canvas_event_sink


async def canvas_session_start_handler(request):
    """POST /workspaces/{workspace_id}/canvas/session — start/resume."""
    volume_path = request.app[WORKER_HTTP].volume_path
    _canvas_event_sink = request.app[WORKER_HTTP].canvas_event_sink
    from canvas_session_service import get_canvas_manager

    workspace_id = request.match_info["workspace_id"]
    manager = get_canvas_manager(volume_path, event_sink=_canvas_event_sink)
    result = await manager.start_session(workspace_id)
    if not result.get("success"):
        status = 409 if result.get("conflict") else 500
        return web.json_response(
            {"error": result.get("error", "Canvas session error")},
            status=status,
        )
    return web.json_response(result)


async def canvas_session_status_handler(request):
    """GET /workspaces/{workspace_id}/canvas/session — status."""
    volume_path = request.app[WORKER_HTTP].volume_path
    _canvas_event_sink = request.app[WORKER_HTTP].canvas_event_sink
    from canvas_session_service import get_canvas_manager

    workspace_id = request.match_info["workspace_id"]
    manager = get_canvas_manager(volume_path, event_sink=_canvas_event_sink)
    result = await manager.get_status(workspace_id)
    if not result.get("success"):
        status = 404 if result.get("not_found") else 500
        return web.json_response(
            {"error": result.get("error", "Canvas session error")},
            status=status,
        )
    return web.json_response(result)


async def canvas_session_stop_handler(request):
    """DELETE /workspaces/{workspace_id}/canvas/session — stop."""
    volume_path = request.app[WORKER_HTTP].volume_path
    _canvas_event_sink = request.app[WORKER_HTTP].canvas_event_sink
    from canvas_session_service import get_canvas_manager

    workspace_id = request.match_info["workspace_id"]
    manager = get_canvas_manager(volume_path, event_sink=_canvas_event_sink)
    result = await manager.stop_session(workspace_id)
    if not result.get("success"):
        status = 404 if result.get("not_found") else 500
        return web.json_response(
            {"error": result.get("error", "Canvas session error")},
            status=status,
        )
    return web.json_response(result)


async def canvas_decision_handler(request):
    """POST /workspaces/{workspace_id}/canvas/session/decision — S4.

    Resolve a pending approval so the awaiting can_use_tool callback
    proceeds. Body: {"request_id": "...", "approved": true|false}.
    """
    volume_path = request.app[WORKER_HTTP].volume_path
    _canvas_event_sink = request.app[WORKER_HTTP].canvas_event_sink
    from canvas_session_service import get_canvas_manager

    workspace_id = request.match_info["workspace_id"]
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — any unreadable body is a 400, as before the move (O2b)
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    request_id = (body.get("request_id") or "").strip()
    approved = body.get("approved")
    if not request_id:
        return web.json_response({"error": "request_id is required"}, status=400)
    if not isinstance(approved, bool):
        return web.json_response({"error": "approved must be a boolean"}, status=400)

    manager = get_canvas_manager(volume_path, event_sink=_canvas_event_sink)
    result = await manager.decide(workspace_id, request_id, approved)
    if not result.get("success"):
        status = 404 if result.get("not_found") else 500
        return web.json_response(
            {"error": result.get("error", "Canvas decision error")},
            status=status,
        )
    return web.json_response(result)


async def canvas_auto_accept_handler(request):
    """POST /workspaces/{workspace_id}/canvas/session/auto-accept — S4.

    Toggle session-scoped auto-accept for FILE EDITS (never bash).
    Body: {"enabled": true|false}.
    """
    volume_path = request.app[WORKER_HTTP].volume_path
    _canvas_event_sink = request.app[WORKER_HTTP].canvas_event_sink
    from canvas_session_service import get_canvas_manager

    workspace_id = request.match_info["workspace_id"]
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — any unreadable body is a 400, as before the move (O2b)
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    enabled = body.get("enabled")
    if not isinstance(enabled, bool):
        return web.json_response({"error": "enabled must be a boolean"}, status=400)

    manager = get_canvas_manager(volume_path, event_sink=_canvas_event_sink)
    result = await manager.set_auto_accept(workspace_id, enabled)
    if not result.get("success"):
        status = 404 if result.get("not_found") else 500
        return web.json_response(
            {"error": result.get("error", "Canvas auto-accept error")},
            status=status,
        )
    return web.json_response(result)


async def canvas_message_handler(request):
    """POST /workspaces/{workspace_id}/canvas/session/message — PRD-203 C·S7.

    Send a user prompt to the live session's SDK client (client.query) so
    the agent actually works. The pump streams the resulting turns back as
    canvas events. Body: {"prompt": "..."}.
    """
    volume_path = request.app[WORKER_HTTP].volume_path
    _canvas_event_sink = request.app[WORKER_HTTP].canvas_event_sink
    from canvas_session_service import get_canvas_manager

    workspace_id = request.match_info["workspace_id"]
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — any unreadable body is a 400, as before the move (O2b)
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return web.json_response({"error": "prompt is required"}, status=400)

    manager = get_canvas_manager(volume_path, event_sink=_canvas_event_sink)
    result = await manager.send_message(workspace_id, prompt)
    if not result.get("success"):
        status = 404 if result.get("not_found") else 500
        return web.json_response(
            {"error": result.get("error", "Canvas message error")},
            status=status,
        )
    return web.json_response(result)
