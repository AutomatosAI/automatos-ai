"""PRD-282 US-301: the request's context is held where every line of the request can read it.

``RequestContextMiddleware`` replaced main.py's ``add_request_id_middleware`` (a
``BaseHTTPMiddleware``). Plain ASGI and registered last, it is the outermost of the app's
own middleware, so the request's id is set in the task the server writes its access line
from. These tests stand in for the server with a shim around the app that reads the id
when the response starts, as uvicorn's access line does.

* The id is the request's ``X-Request-ID`` cleaned by ``safe_request_id`` (the characters
  it keeps, at most 128), or a new 12-character one when none is sent or none is left; the
  response carries it once, and a 500's body and its log line carry the same id. The method, path and workspace are set as before, ``__all__`` is no
  workspace, and every value is put back when the request ends.
* A middleware inside it (``api_tracking_middleware`` is one) and the route see the id.
* With tracing on, the server span carries the ids (``annotate_request``), and a JSON line
  written in a sampled request carries its trace and span ids; an unsampled one, none.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.monitoring.automatos_logging import (  # noqa: E402
    http_method_var, http_path_var, request_id_var, request_log_fields, workspace_id_var,
)
from core.monitoring.json_log_format import JsonLineFormatter  # noqa: E402
from core.monitoring.request_context import RequestContextMiddleware  # noqa: E402
from core.observability import otel  # noqa: E402

WS = "00000000-0000-0000-0000-0000000000c2"


def _context() -> tuple:
    return (request_id_var.get(""), workspace_id_var.get(""), http_method_var.get(""), http_path_var.get(""))


def _app(seen: dict) -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def inner(request: Request, call_next):   # a BaseHTTPMiddleware inside it, as main.py has
        seen["inner"] = request_id_var.get("")
        return await call_next(request)

    @app.get("/api/things/{thing_id}")
    def thing(thing_id: int):
        seen["route"] = _context()
        seen["line"] = json.loads(JsonLineFormatter("s", "v", request_log_fields).format(
            logging.LogRecord("t", logging.INFO, __file__, 1, "in the route", (), None)))
        return JSONResponse({"id": thing_id}, headers={"X-Request-ID": "set-by-the-route"})

    app.add_middleware(RequestContextMiddleware)
    return app


def _as_the_server_sees_it(app, seen: dict):
    """The app behind a shim that reads the request id when the response starts, as uvicorn's access line does."""
    async def server(scope, receive, send):
        async def send_and_log(message):
            if message["type"] == "http.response.start":
                seen["access"] = request_id_var.get("")
            await send(message)

        await app(scope, receive, send_and_log)
        seen["after"] = _context()
    return server


def _get(headers: dict) -> tuple:
    seen: dict = {}
    response = TestClient(_as_the_server_sees_it(_app(seen), seen)).get("/api/things/7", headers=headers)
    assert response.status_code == 200
    return response, seen


def test_the_requests_id_is_on_every_line_its_access_line_included():
    response, seen = _get({"X-Request-ID": "req-abc", "X-Workspace-ID": WS})
    assert response.headers.get_list("x-request-id") == ["req-abc"]
    assert seen["access"] == seen["inner"] == "req-abc"
    assert seen["route"] == ("req-abc", WS, "GET", "/api/things/7")
    assert seen["after"] == ("", "", "", "")


def test_a_request_without_an_id_gets_a_new_one_and_the_response_says_it():
    response, seen = _get({})
    (request_id,) = response.headers.get_list("x-request-id")
    assert re.fullmatch(r"[0-9a-f]{12}", request_id)
    assert seen["access"] == seen["route"][0] == request_id


@pytest.mark.parametrize("headers, workspace", [
    ({"X-Workspace": WS}, WS), ({"X-Workspace-ID": "__all__"}, ""), ({}, ""),
])
def test_the_workspace_is_the_one_the_request_names(headers, workspace):
    _, seen = _get(headers)
    assert seen["route"][1] == workspace


@pytest.mark.parametrize("sent, kept", [
    ("req 282 [x]", "req282x"), ("a" * 300, "a" * 128), ("evil\tline <script>", "evillinescript"),
])
def test_the_id_is_cleaned_once_and_every_reader_sees_that_value(sent, kept):
    response, seen = _get({"X-Request-ID": sent})
    assert response.headers.get_list("x-request-id") == [kept]
    assert seen["access"] == seen["inner"] == seen["route"][0] == seen["line"]["request_id"] == kept


@pytest.mark.parametrize("sent", [" [ ] ", "<>", "\t"])
def test_an_id_that_cleans_to_nothing_is_replaced_by_a_new_one(sent):
    response, seen = _get({"X-Request-ID": sent})
    (request_id,) = response.headers.get_list("x-request-id")
    assert re.fullmatch(r"[0-9a-f]{12}", request_id) and seen["route"][0] == request_id


@pytest.mark.parametrize("sent", ["req 282 [x]", " [ ] "])
def test_a_500_says_the_same_id_as_the_response_and_its_log_line(sent, caplog):
    from core.observability.error_response import json_500_for_unhandled_errors

    app = FastAPI()
    app.middleware("http")(json_500_for_unhandled_errors)   # inside it, as main.py registers them

    @app.get("/boom")
    def boom():
        raise RuntimeError("it broke")

    app.add_middleware(RequestContextMiddleware)
    with caplog.at_level(logging.ERROR, logger="core.observability.error_response"):
        response = TestClient(app, raise_server_exceptions=False).get("/boom", headers={"X-Request-ID": sent})
    (request_id,) = response.headers.get_list("x-request-id")
    assert response.status_code == 500 and response.json()["request_id"] == request_id
    (line,) = [r.getMessage() for r in caplog.records if r.name == "core.observability.error_response"]
    assert line.endswith(f"(request_id={request_id})")


def test_the_context_is_put_back_after_the_app_raises():
    import asyncio

    async def app(scope, receive, send):
        raise RuntimeError("downstream")

    async def serve() -> tuple:
        scope = {"type": "http", "method": "GET", "path": "/x", "query_string": b"", "root_path": "",
                 "scheme": "http", "server": ("t", 80),
                 "headers": [(b"x-request-id", b"req-282"), (b"x-workspace-id", WS.encode())]}
        with pytest.raises(RuntimeError):
            await RequestContextMiddleware(app)(scope, None, None)
        return _context()

    assert asyncio.run(serve()) == ("", "", "", "")


def test_anything_but_an_http_request_passes_through_untouched():
    import asyncio

    calls = []

    async def app(scope, receive, send):
        calls.append((scope["type"], send, request_id_var.get("")))

    async def send(message):
        return None

    for kind in ("lifespan", "websocket"):
        asyncio.run(RequestContextMiddleware(app)({"type": kind}, None, send))
    assert calls == [("lifespan", send, ""), ("websocket", send, "")]


def test_main_registers_it_outermost_and_the_old_middleware_is_gone():
    from main import app

    assert app.user_middleware[0].cls is RequestContextMiddleware
    assert "add_request_id_middleware" not in (_ORCH / "main.py").read_text(encoding="utf-8")


@pytest.fixture
def traced(monkeypatch):
    """Tracing on, an in-memory exporter, and the app above under the server spans."""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setattr(otel.config, "OTEL_ENABLED", True, raising=False)

    def make(ratio: float):
        exporter, seen = InMemorySpanExporter(), {}
        provider = otel.build_provider(exporter, ratio)
        app = _app(seen)
        otel.instrument_app(app, tracer_provider=provider)
        return TestClient(app), provider, exporter, seen

    return make


def test_the_server_span_carries_the_ids_and_a_line_carries_the_trace(traced):
    from opentelemetry.trace import SpanKind

    client, provider, exporter, seen = traced(1.0)
    assert client.get("/api/things/7", headers={"X-Request-ID": "req-abc", "X-Workspace-ID": WS}).status_code == 200
    provider.force_flush()
    (span,) = [s for s in exporter.get_finished_spans() if s.kind == SpanKind.SERVER]
    assert span.attributes[otel.ATTR_REQUEST_ID] == "req-abc" and span.attributes[otel.ATTR_WORKSPACE_ID] == WS
    assert seen["line"]["trace_id"] == f"{span.context.trace_id:032x}"
    assert re.fullmatch(r"[0-9a-f]{16}", seen["line"]["span_id"])
    assert seen["line"]["request_id"] == "req-abc"


def test_a_line_in_a_trace_that_is_not_sampled_names_no_trace(traced):
    client, _, _, seen = traced(0.0)
    assert client.get("/api/things/7", headers={"X-Request-ID": "req-abc"}).status_code == 200
    assert "trace_id" not in seen["line"] and "span_id" not in seen["line"]
    assert seen["line"]["request_id"] == "req-abc"
