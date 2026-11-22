"""PRD-282 US-301: the request's id, method, path and workspace, held for every line it writes.

``RequestContextMiddleware`` replaces main.py's ``add_request_id_middleware``, a
``BaseHTTPMiddleware`` that set the same context variables. main.py registers it last, so
it is the outermost of the app's own middleware, and it is plain ASGI, for one reason:
uvicorn writes a request's access line when the response starts, in the request's own task.
A ``BaseHTTPMiddleware`` runs the rest of the app in a task of its own, so what it sets is
never seen outside it: not by the middleware around it, not by uvicorn's access line. Set
here, in uvicorn's task and before any other middleware of the app runs, the request's id is
on every line the request writes, its access line included (``server_log_lines``).

What it sets is what the old middleware set: the request's ``X-Request-ID``, or a new
12-character id; the method and the path; the workspace from ``X-Workspace-ID`` (or
``X-Workspace``) unless it is ``__all__``; the two ids on the server span (PRD-256); and
``X-Request-ID`` on the response, the request's own when it sent one. Each value is put
back when the request ends. Anything but an HTTP request passes through untouched.
"""
from __future__ import annotations

import uuid
from contextvars import ContextVar, Token
from typing import Any, Awaitable, Callable, List, MutableMapping, Tuple

from starlette.datastructures import URL, Headers

from core.monitoring.automatos_logging import http_method_var, http_path_var, request_id_var, workspace_id_var
from core.observability.otel import annotate_request

REQUEST_ID_HEADER = "x-request-id"
WORKSPACE_HEADERS = ("x-workspace-id", "x-workspace")
ALL_WORKSPACES = "__all__"
NEW_ID_LENGTH = 12

Message = MutableMapping[str, Any]
Send = Callable[[Message], Awaitable[None]]


def _workspace(headers: Headers) -> str:
    """The workspace a request names (best effort, before sign-in), or ''."""
    named = next((headers[name] for name in WORKSPACE_HEADERS if headers.get(name)), "")
    return "" if named == ALL_WORKSPACES else named


def _hold(scope: Message, headers: Headers, request_id: str) -> List[Tuple[ContextVar, Token]]:
    """Set the request's context variables; the tokens put each back."""
    values = [(request_id_var, request_id), (http_method_var, scope["method"]),
              (http_path_var, URL(scope=scope).path)]
    workspace = _workspace(headers)
    if workspace:
        values.append((workspace_id_var, workspace))
    return [(var, var.set(value)) for var, value in values]


def _answering_with(send: Send, request_id: str) -> Send:
    """``send``, with ``X-Request-ID`` on the response's headers (any other value of it replaced)."""
    header = (REQUEST_ID_HEADER.encode("latin-1"), request_id.encode("latin-1"))

    async def send_with_request_id(message: Message) -> None:
        if message["type"] == "http.response.start":
            kept = [item for item in message.get("headers", []) if item[0].lower() != header[0]]
            message = {**message, "headers": [*kept, header]}
        await send(message)

    return send_with_request_id


class RequestContextMiddleware:
    """Plain ASGI: the request's context for its logs, its span and its response header."""

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Message, receive: Callable[[], Awaitable[Message]], send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        request_id = headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:NEW_ID_LENGTH]
        held = _hold(scope, headers, request_id)
        annotate_request(request_id_var.get(""), workspace_id_var.get(""))
        try:
            await self.app(scope, receive, _answering_with(send, request_id))
        finally:
            for var, token in reversed(held):
                var.reset(token)


__all__ = ["ALL_WORKSPACES", "NEW_ID_LENGTH", "REQUEST_ID_HEADER", "RequestContextMiddleware", "WORKSPACE_HEADERS"]
