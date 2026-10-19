"""The workspace worker's HTTP server: health, file browsing, tools, the canvas.

Moved out of ``WorkspaceWorker._health_server`` (PRD-256 O2b, #847), where every
route was a closure inside one 615-line method, so any change to the server,
tracing included, failed the 50-line code-shape rule. The behaviour is the
same; the closures' values now live on the app (:func:`worker_http_of`), and the
handlers are plain functions in three modules:

* ``worker_routes_files``: list, read, write, grep and download files;
* ``worker_routes_tools``: exec, git and html-to-png;
* ``worker_routes_canvas``: the canvas SDK session endpoints (PRD-170).

PRD-262 (row 36 of the 19 Oct size breakdown): the server is Starlette on uvicorn,
which the image already carries for mcp, instead of aiohttp and its six helper
packages (12.2 MB). Every route keeps its path, its methods (GET also answers
HEAD), the X-Internal-Token check and its answers. What aiohttp did on its own is
spelt out here:

* a JSON body is read as aiohttp's ``client_max_size`` (1 MiB) allowed: a body of 1 MiB
  or more is the handler's 400 "Invalid JSON body", as it was (:func:`read_json`);
* a JSON answer is aiohttp's ``json_response`` byte for byte: ``json.dumps``, sent
  as ``application/json; charset=utf-8`` (:func:`json_response`);
* no redirect for a trailing slash, which aiohttp never sent;
* idle keep-alive connections are held 3630 s, aiohttp 3.13's default, never uvicorn's
  5 s, which would race the API's pooled connections (httpx keeps an idle one 5 s);
* uvicorn never takes SIGTERM or SIGINT: main.py's handlers stop the worker, and
  :func:`serve` then stops the server, as the aiohttp runner was stopped;
* a server that cannot start (its port taken) logs the error and leaves the worker
  running, as aiohttp's failed ``site.start()`` did; uvicorn would exit the process.
"""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterator

import uvicorn
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from worker_config import worker_bind_host, worker_internal_token, workspace_root
from workspace_manager import WorkspaceManager, is_workspace_id

logger = logging.getLogger("workspace-worker")

HEALTH_PATH = "/health"
TOKEN_HEADER = "x-internal-token"
JSON_CONTENT_TYPE = "application/json; charset=utf-8"
MAX_BODY_BYTES = 1024 ** 2       # aiohttp's client_max_size: a body this size or larger is refused
KEEP_ALIVE_SECONDS = 3630        # aiohttp 3.13's keepalive_timeout, the worker's until PRD-262
SHUTDOWN_GRACE_SECONDS = 60      # aiohttp's AppRunner shutdown_timeout
STOP_POLL_SECONDS = 1.0


@dataclass(frozen=True)
class WorkerHttp:
    """What every handler used to close over."""

    worker: Any
    volume_path: str
    internal_token: str
    canvas_event_sink: Callable[[dict], Awaitable[None]]


class BodyTooLarge(ValueError):
    """A request body of :data:`MAX_BODY_BYTES` or more."""


def worker_http_of(request: Request) -> WorkerHttp:
    """The values every handler shares, from the app serving ``request``."""
    return request.app.state.worker_http


def json_response(data: Any, status: int = 200) -> Response:
    """``data`` as aiohttp's ``json_response`` sent it: ``json.dumps``, UTF-8 JSON."""
    return Response(json.dumps(data), status_code=status, headers={"Content-Type": JSON_CONTENT_TYPE})


async def read_json(request: Request) -> Any:
    """The body as JSON, read while it stays under :data:`MAX_BODY_BYTES`.

    A body that reaches the limit raises :class:`BodyTooLarge` there (aiohttp's
    ``read()`` compared with ``>=``), and an
    unreadable one raises too: each handler answers both with its 400
    "Invalid JSON body", as it answered aiohttp's ``HTTPRequestEntityTooLarge``."""
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) >= MAX_BODY_BYTES:
            raise BodyTooLarge(f"request body of {MAX_BODY_BYTES} bytes or more")
    return json.loads(body.decode("utf-8"))


def open_workspace(request: Request):
    """F173: the request's workspace, provisioned on first use, or the
    response that refuses it. A workspace made by the signup wizard has
    no directory here until something opens it (the backend's PRD-130
    note: no worker container is provisioned for it), and only the file
    listing and the task runner used to provision one, so its first
    report or file write got 404 "Workspace not found". Only a
    canonical UUID may name a workspace directory."""
    workspace_id = request.path_params["workspace_id"]
    if not is_workspace_id(workspace_id):
        return None, json_response({"error": "Invalid workspace id"}, status=400)
    ws_manager = WorkspaceManager(workspace_id, worker_http_of(request).volume_path)
    ws_manager.ensure_workspace_exists()
    return ws_manager, None


def _token_accepted(scope: Scope) -> bool:
    """True when no token is configured, or the request carries it (constant time, #1043)."""
    internal_token = scope["app"].state.worker_http.internal_token
    if not internal_token:
        return True
    sent = Headers(scope=scope).get(TOKEN_HEADER, "")
    return hmac.compare_digest(sent.encode(), internal_token.encode())


class InternalAuthMiddleware:
    """Every route but /health needs X-Internal-Token when one is configured."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] == HEALTH_PATH or _token_accepted(scope):
            await self.app(scope, receive, send)
            return
        await json_response({"error": "Unauthorized"}, status=401)(scope, receive, send)


async def health_handler(request: Request) -> Response:
    http = worker_http_of(request)
    worker = http.worker
    return json_response({
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
        ("GET", HEALTH_PATH, health_handler),
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


def build_app(worker) -> Starlette:
    """The worker's app: the internal-token check, Prometheus metrics, every route, and
    (PRD-256 O2b, when OTEL_ENABLED) a server span per request that continues the caller's trace."""
    from automatos_metrics import add_starlette_metrics
    from worker_otel import attach as attach_tracing
    from worker_routes_canvas import make_canvas_event_sink

    # One route per method, as add_get/add_post/add_delete made them; a GET route also answers HEAD.
    app = Starlette(routes=[Route(path, handler, methods=[method]) for method, path, handler in _routes()],
                    middleware=[Middleware(InternalAuthMiddleware)])
    app.router.redirect_slashes = False
    app.state.worker_http = WorkerHttp(worker=worker, volume_path=str(workspace_root()),
                                       internal_token=worker_internal_token(),
                                       canvas_event_sink=make_canvas_event_sink(worker))
    # Prometheus metrics endpoint + request tracking, outside the token check
    add_starlette_metrics(app, service="workspace-worker")
    attach_tracing(app)   # outermost, so the span covers everything below it
    return app


class EmbeddedServer(uvicorn.Server):
    """uvicorn inside the worker's own event loop, without its signal handlers: main.py's
    SIGTERM and SIGINT handlers stop the worker, and the worker stops the server."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def make_server(worker) -> EmbeddedServer:
    """The server for ``worker``'s app on its health port, logging through the worker's
    own handlers (access lines included, as aiohttp's were)."""
    config = uvicorn.Config(build_app(worker), host=worker_bind_host(), port=worker.health_port,
                            log_config=None, ws="none", timeout_keep_alive=KEEP_ALIVE_SECONDS,
                            timeout_graceful_shutdown=SHUTDOWN_GRACE_SECONDS)
    return EmbeddedServer(config)


async def _serve_while_running(worker, server: EmbeddedServer, serving: asyncio.Task) -> None:
    """Wait while the worker runs and the server serves; say so once it listens."""
    announced = False
    while worker._running and not serving.done():
        if server.started and not announced:
            logger.info("Health server listening on port %d", worker.health_port)
            announced = True
        await asyncio.sleep(STOP_POLL_SECONDS)


async def _serve_or_log(server: EmbeddedServer, port: int) -> None:
    """``server.serve()``, with uvicorn's exit on a failed start (a taken port) logged
    instead: asyncio would re-raise that SystemExit out of the worker's event loop."""
    try:
        await server.serve()
    except SystemExit as exc:
        logger.error("Health server could not start on port %d (exit %s); the worker runs without it",
                     port, exc.code)


async def serve(worker) -> None:
    """Serve the app on the worker's health port until the worker stops."""
    server = make_server(worker)
    serving = asyncio.create_task(_serve_or_log(server, worker.health_port))
    try:
        await _serve_while_running(worker, server, serving)
    except asyncio.CancelledError:
        pass
    finally:
        server.should_exit = True
        await serving


__all__ = ["MAX_BODY_BYTES", "BodyTooLarge", "EmbeddedServer", "WorkerHttp", "build_app", "health_handler",
           "InternalAuthMiddleware", "json_response", "make_server", "open_workspace", "read_json", "serve",
           "worker_http_of"]
