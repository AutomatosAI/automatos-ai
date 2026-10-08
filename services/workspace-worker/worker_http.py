"""The workspace worker's HTTP server: health, file browsing, tools, the canvas.

Moved out of ``WorkspaceWorker._health_server`` (PRD-256 O2b, #847), where every
route was a closure inside one 615-line method, so any change to the server,
tracing included, failed the 50-line code-shape rule. The behaviour is the
same; the closures' values now live on the app (:data:`WORKER_HTTP`), and the
handlers are plain functions in three modules:

* ``worker_routes_files``: list, read, write, grep and download files;
* ``worker_routes_tools``: exec, git and html-to-png;
* ``worker_routes_canvas``: the canvas SDK session endpoints (PRD-170).
"""
from __future__ import annotations

import asyncio
import hmac
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from aiohttp import web

from worker_config import worker_bind_host, worker_internal_token, workspace_root
from workspace_manager import WorkspaceManager, is_workspace_id

logger = logging.getLogger("workspace-worker")


@dataclass(frozen=True)
class WorkerHttp:
    """What every handler used to close over."""

    worker: Any
    volume_path: str
    internal_token: str
    canvas_event_sink: Callable[[dict], Awaitable[None]]


WORKER_HTTP = web.AppKey("worker_http", WorkerHttp)


def open_workspace(request):
    """F173: the request's workspace, provisioned on first use, or the
    response that refuses it. A workspace made by the signup wizard has
    no directory here until something opens it (the backend's PRD-130
    note: no worker container is provisioned for it), and only the file
    listing and the task runner used to provision one, so its first
    report or file write got 404 "Workspace not found". Only a
    canonical UUID may name a workspace directory."""
    workspace_id = request.match_info["workspace_id"]
    if not is_workspace_id(workspace_id):
        return None, web.json_response({"error": "Invalid workspace id"}, status=400)
    ws_manager = WorkspaceManager(workspace_id, request.app[WORKER_HTTP].volume_path)
    ws_manager.ensure_workspace_exists()
    return ws_manager, None


# Auth middleware — reject requests without valid internal token
@web.middleware
async def internal_auth_middleware(request, handler):
    # Health endpoint is always public
    if request.path == "/health":
        return await handler(request)
    # If token is configured, enforce it
    internal_token = request.app[WORKER_HTTP].internal_token
    if internal_token:
        req_token = request.headers.get("X-Internal-Token", "")
        # Constant time: a plain != stops at the first differing character (review on #1043).
        if not hmac.compare_digest(req_token.encode(), internal_token.encode()):
            return web.json_response({"error": "Unauthorized"}, status=401)
    return await handler(request)


async def health_handler(request):
    http = request.app[WORKER_HTTP]
    worker = http.worker
    return web.json_response({
        "status": "healthy",
        "worker_id": worker._worker_id,
        "active_tasks": len(worker._active_tasks),
        "concurrency": worker.concurrency,
        "volume_path": http.volume_path,
    })


def _routes():
    """``(method, path, handler)`` for every route, in the order they were added."""
    import worker_routes_canvas as canvas
    import worker_routes_files as files
    import worker_routes_tools as tools

    ws = "/workspaces/{workspace_id}"
    return (
        ("GET", "/health", health_handler),
        ("GET", f"{ws}/files", files.list_files_handler),
        ("GET", f"{ws}/files/content", files.file_content_handler),
        ("POST", f"{ws}/exec", tools.exec_handler),
        ("POST", f"{ws}/files/write", files.write_file_handler),
        ("GET", f"{ws}/files/grep", files.grep_handler),
        ("GET", f"{ws}/files/download", files.download_file_handler),
        ("POST", f"{ws}/git", tools.git_handler),
        ("POST", f"{ws}/html-to-png", tools.html_to_png_handler),
        ("POST", f"{ws}/canvas/session", canvas.canvas_session_start_handler),
        ("GET", f"{ws}/canvas/session", canvas.canvas_session_status_handler),
        ("DELETE", f"{ws}/canvas/session", canvas.canvas_session_stop_handler),
        ("POST", f"{ws}/canvas/session/decision", canvas.canvas_decision_handler),
        ("POST", f"{ws}/canvas/session/auto-accept", canvas.canvas_auto_accept_handler),
        ("POST", f"{ws}/canvas/session/message", canvas.canvas_message_handler),
    )


def build_app(worker) -> web.Application:
    """The worker's app: the internal-token check, Prometheus metrics, every route."""
    from automatos_metrics import add_aiohttp_metrics
    from worker_routes_canvas import make_canvas_event_sink

    app = web.Application(middlewares=[internal_auth_middleware])
    app[WORKER_HTTP] = WorkerHttp(worker=worker, volume_path=str(workspace_root()),
                                  internal_token=worker_internal_token(),
                                  canvas_event_sink=make_canvas_event_sink(worker))
    # Prometheus metrics endpoint + request tracking
    add_aiohttp_metrics(app, service="workspace-worker")
    for method, path, handler in _routes():
        # add_get/add_post/add_delete, as before: add_get also answers HEAD.
        getattr(app.router, f"add_{method.lower()}")(path, handler)
    return app


async def serve(worker) -> None:
    """Serve the app on the worker's health port until the worker stops."""
    runner = web.AppRunner(build_app(worker))
    await runner.setup()
    site = web.TCPSite(runner, worker_bind_host(), worker.health_port)

    try:
        await site.start()
        logger.info("Health server listening on port %d", worker.health_port)
        while worker._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


__all__ = ["WORKER_HTTP", "WorkerHttp", "build_app", "health_handler", "internal_auth_middleware",
           "open_workspace", "serve"]
