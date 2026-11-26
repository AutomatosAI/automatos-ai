"""PRD-282 US-302: the request's fields on the worker's JSON log lines.

A JSON line (``json_log_format``, a copy of the platform's) carries the request's id,
workspace, method and path, and the trace and span ids when a trace is being recorded,
the same fields as the API's. ``RequestLogContextMiddleware`` holds the first four for
every line a request writes, uvicorn's access line included: it is plain ASGI, so what it
sets is seen in the request's own task, where uvicorn writes that line.

* **The request id** is the caller's ``X-Request-ID``. The platform sends its own request's
  id with every call (``orchestrator/core/workspace_client.py``), so one id finds a
  request's lines in the API's logs and the worker's. A call without one has none. Only
  the characters the platform's own check keeps are kept (``safe_request_id``), at most 128.
* **The workspace** is the one the path names (``/workspaces/<id>/...``).
* **The path** never carries the query.
* **The trace and span ids** are read from the current span once tracing has started
  (``worker_otel``); with tracing off nothing from OpenTelemetry is imported.

A text line (``TextLineFormatter``) carries the request's id as `` req=<id>`` after its level,
as the API's text line carries ``req=``; a line outside a request is as it was.
"""
from __future__ import annotations

import logging
import re
from contextvars import ContextVar, Token
from typing import Any, Awaitable, Callable, Dict, List, MutableMapping, Tuple

import worker_otel

REQUEST_ID_HEADER = b"x-request-id"
REQUEST_ID_UNSAFE = re.compile(r"[^A-Za-z0-9._:-]")
MAX_REQUEST_ID_LEN = 128
WORKSPACE_PATH = re.compile(r"^/workspaces/([^/]+)")

request_id_var: ContextVar[str] = ContextVar("worker_request_id", default="")
workspace_id_var: ContextVar[str] = ContextVar("worker_workspace_id", default="")
http_method_var: ContextVar[str] = ContextVar("worker_http_method", default="")
http_path_var: ContextVar[str] = ContextVar("worker_http_path", default="")

Message = MutableMapping[str, Any]

# PRD-256 O5's " trace=<id>" when a sampled span is current (worker_otel), and PRD-282's
# " req=<id>" in a request; each empty otherwise.
TEXT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s:%(_request_context)s%(_trace_context)s %(message)s"


def request_log_fields() -> Dict[str, str]:
    """The current request's fields for a JSON line; empty ones are left out by the formatter."""
    fields = {
        "request_id": request_id_var.get(),
        "workspace_id": workspace_id_var.get(),
        "method": http_method_var.get(),
        "path": http_path_var.get(),
    }
    if worker_otel.tracing_started():
        fields["trace_id"], fields["span_id"] = worker_otel.current_trace_ids()
    return fields


class TextLineFormatter(logging.Formatter):
    """The worker's text line, with the current request's id; the record is not changed."""

    def __init__(self) -> None:
        super().__init__(TEXT_FORMAT)

    def formatMessage(self, record: logging.LogRecord) -> str:
        request_id = request_id_var.get()
        values = {"_trace_context": "", **record.__dict__,
                  "_request_context": f" req={request_id}" if request_id else ""}
        return TEXT_FORMAT % values


def _request_id(scope: Message) -> str:
    """The caller's ``X-Request-ID``, its unsafe characters dropped; '' without one."""
    found = next((value for name, value in scope.get("headers", []) if name.lower() == REQUEST_ID_HEADER), b"")
    return REQUEST_ID_UNSAFE.sub("", found.decode("latin-1"))[:MAX_REQUEST_ID_LEN]


def _hold(scope: Message) -> List[Tuple[ContextVar, Token]]:
    """Set the request's context variables; the tokens put each back."""
    path = scope.get("path", "")
    workspace = WORKSPACE_PATH.match(path)
    values = [(request_id_var, _request_id(scope)), (http_method_var, scope.get("method", "")),
              (http_path_var, path), (workspace_id_var, workspace.group(1) if workspace else "")]
    return [(var, var.set(value)) for var, value in values]


class RequestLogContextMiddleware:
    """Plain ASGI: the request's id, workspace, method and path for every line it writes."""

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Message, receive: Callable[[], Awaitable[Message]],
                       send: Callable[[Message], Awaitable[None]]) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        held = _hold(scope)
        try:
            await self.app(scope, receive, send)
        finally:
            for var, token in reversed(held):
                var.reset(token)


__all__ = ["MAX_REQUEST_ID_LEN", "REQUEST_ID_HEADER", "REQUEST_ID_UNSAFE", "RequestLogContextMiddleware",
           "TEXT_FORMAT", "TextLineFormatter", "WORKSPACE_PATH", "http_method_var", "http_path_var", "request_id_var", "request_log_fields",
           "workspace_id_var"]
