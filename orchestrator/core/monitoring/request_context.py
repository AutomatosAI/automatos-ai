"""PRD-282 US-301: the request's id, method, path and workspace, held for every line it writes.

``RequestContextMiddleware`` replaces main.py's ``add_request_id_middleware``, a
``BaseHTTPMiddleware`` that set the same context variables. main.py registers it last, after
the metrics middleware too, so it is the outermost of the app's own middleware, and it is
plain ASGI, for one reason:
uvicorn writes a request's access line when the response starts, in the request's own task.
A ``BaseHTTPMiddleware`` runs the rest of the app in a task of its own, so what it sets is
never seen outside it: not by the middleware around it, not by uvicorn's access line. Set
here, in uvicorn's task and before any other middleware of the app runs, the request's id is
on every line the request writes, its access line included (``server_log_lines``).

What it sets is what the old middleware set: the request's ``X-Request-ID``, kept to the
characters ``safe_request_id`` keeps and at most 128 of them, or a new 12-character id when
it sends none or none is left; the method and the path, and the scope, whose route the log
lines name instead of the path (#1659); the workspace from ``X-Workspace-ID`` (or
``X-Workspace``) when it is a workspace id, a UUID of at most 64 characters, and none otherwise
(``__all__`` included): it is the caller's, read before sign-in (#1659); the two ids on the server span (PRD-256); and
``X-Request-ID`` on the response. That one id is on every line the request writes, on the
response, in a 500's body and on each call the API makes to the workspace worker. Each value
is put back when the request ends, whether it answered or raised. Anything but an HTTP request passes through untouched.
"""
from __future__ import annotations

import uuid
from contextvars import ContextVar, Token
from typing import Any, Awaitable, Callable, List, MutableMapping, Tuple

from starlette.datastructures import URL, Headers

from core.monitoring.automatos_logging import (
    http_method_var, http_path_var, http_scope_var, request_id_var, workspace_id_var,
)
from core.observability.error_response import safe_request_id
from core.observability.otel import annotate_request

REQUEST_ID_HEADER = "x-request-id"
WORKSPACE_HEADERS = ("x-workspace-id", "x-workspace")
NEW_ID_LENGTH = 12
# The longest header value read as a workspace id; a UUID in any form UUID() takes fits.
MAX_WORKSPACE_ID_LEN = 64

Message = MutableMapping[str, Any]
Send = Callable[[Message], Awaitable[None]]


def _workspace(headers: Headers) -> str:
    """The workspace a request names (best effort, before sign-in) as a canonical UUID, or ''
    when it names none or anything that is not a workspace id."""
    named = next((headers[name] for name in WORKSPACE_HEADERS if headers.get(name)), "")
    if len(named) > MAX_WORKSPACE_ID_LEN:
        return ""
    try:
        return str(uuid.UUID(named.strip()))
    except ValueError:
        return ""


def _hold(scope: Message, headers: Headers, request_id: str) -> List[Tuple[ContextVar, Token]]:
    """Set the request's context variables; the tokens put each back. Each is set every
    time: a request that names no workspace holds none, whatever the context around it held."""
    values = [(request_id_var, request_id), (http_method_var, scope["method"]),
              (http_path_var, URL(scope=scope).path), (http_scope_var, scope),
              (workspace_id_var, _workspace(headers))]
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
        request_id = safe_request_id(headers.get(REQUEST_ID_HEADER)) or uuid.uuid4().hex[:NEW_ID_LENGTH]
        held = _hold(scope, headers, request_id)
        annotate_request(request_id_var.get(""), workspace_id_var.get(""))
        try:
            await self.app(scope, receive, _answering_with(send, request_id))
        finally:
            for var, token in reversed(held):
                var.reset(token)


__all__ = ["MAX_WORKSPACE_ID_LEN", "NEW_ID_LENGTH", "REQUEST_ID_HEADER", "RequestContextMiddleware", "WORKSPACE_HEADERS"]
