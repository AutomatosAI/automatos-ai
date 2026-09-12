"""PRD-242 S1 — an unhandled backend error must reach the browser as a 500 it
can READ, not as "TypeError: Failed to fetch".

Starlette's ``@app.exception_handler(Exception)`` runs in ServerErrorMiddleware
(outermost) — outside CORSMiddleware — so its 500 has no
Access-Control-Allow-Origin. The fix is a plain-JSON conversion registered
BEFORE CORS (later-added middleware wraps earlier ones). Two guards:

1. behaviour — a minimal Starlette app wired in the same order returns a 500
   WITH the CORS header;
2. placement — main.py registers the converter before it adds CORSMiddleware
   (an AST read; no boot, no DB).
"""

from __future__ import annotations

import ast
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from core.observability.error_response import (
    INTERNAL_ERROR_DETAIL,
    MAX_REQUEST_ID_LEN,
    json_500_for_unhandled_errors,
    safe_request_id,
)

_MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"
ORIGIN = "http://localhost:3000"


def _app(convert_inside_cors: bool) -> FastAPI:
    app = FastAPI()
    if convert_inside_cors:
        app.middleware("http")(json_500_for_unhandled_errors)
    app.add_middleware(CORSMiddleware, allow_origins=[ORIGIN], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.get("/boom")
    async def boom():
        raise RuntimeError("psycopg2 says no")

    @app.get("/fine")
    async def fine():
        return {"ok": True}

    return app


def test_unhandled_error_is_a_json_500_with_cors_headers():
    client = TestClient(_app(convert_inside_cors=True), raise_server_exceptions=False)
    resp = client.get("/boom", headers={"Origin": ORIGIN, "X-Request-ID": "req-1"})
    assert resp.status_code == 500
    assert resp.headers.get("access-control-allow-origin") == ORIGIN
    assert resp.json() == {"detail": INTERNAL_ERROR_DETAIL, "request_id": "req-1"}


def test_without_the_converter_the_500_has_no_cors_header():
    """Documents the failure the converter exists for."""
    client = TestClient(_app(convert_inside_cors=False), raise_server_exceptions=False)
    resp = client.get("/boom", headers={"Origin": ORIGIN})
    assert resp.status_code == 500
    assert "access-control-allow-origin" not in resp.headers


def test_request_id_is_sanitised_before_it_is_logged_or_echoed():
    # A client-supplied header must not forge log lines (CRLF) or bloat the body.
    assert safe_request_id("req-1.a:b_c") == "req-1.a:b_c"
    assert safe_request_id("evil\r\nINFO: forged line") == "evilINFOforgedline"
    assert safe_request_id(None) == ""
    assert len(safe_request_id("x" * 1000)) == MAX_REQUEST_ID_LEN
    client = TestClient(_app(convert_inside_cors=True), raise_server_exceptions=False)
    resp = client.get("/boom", headers={"Origin": ORIGIN, "X-Request-ID": "ok-1<script>"})
    assert resp.json() == {"detail": INTERNAL_ERROR_DETAIL, "request_id": "ok-1script"}


def test_happy_path_untouched():
    client = TestClient(_app(convert_inside_cors=True))
    resp = client.get("/fine", headers={"Origin": ORIGIN})
    assert resp.status_code == 200 and resp.json() == {"ok": True}
    assert resp.headers.get("access-control-allow-origin") == ORIGIN


def _top_level_statement_index(pred) -> int:
    tree = ast.parse(_MAIN_PY.read_text(encoding="utf-8"))
    for i, node in enumerate(tree.body):
        if pred(node):
            return i
    raise AssertionError("statement not found in main.py")


def _is_call_named(node, attr: str) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == attr
    )


def test_main_registers_the_converter_before_cors_middleware():
    converter = _top_level_statement_index(
        lambda n: isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and any(isinstance(a, ast.Name) and a.id == "json_500_for_unhandled_errors" for a in n.value.args)
    )
    cors = _top_level_statement_index(
        lambda n: _is_call_named(n, "add_middleware")
        and any(isinstance(a, ast.Name) and a.id == "CORSMiddleware" for a in n.value.args)
    )
    assert converter < cors, "the JSON-500 converter must be added BEFORE CORSMiddleware (it wraps inside it)"
