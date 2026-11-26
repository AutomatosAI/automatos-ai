"""#1642 and #1659: no credential from a request's path or query on a log line, and no
unchecked workspace id in the log context.

* #1642: a workspace webhook's key is the credential in its path
  (``/api/webhooks/ws/{workspace_key}``); its verification ``GET`` carries ``hub.verify_token``
  in the query, and the public invitation lookup carries the invitation's token. uvicorn's
  access line (``DigestPathFilter``) collapses the key and drops those queries, and every other
  copy of a path (``redact_path``: metrics, spans, the relay, the console field) collapses the key.
* #1659 item 1: a JSON line's ``path`` is the route template the request was routed to, so no
  path value is on it; before routing, or with no route, the redacted path. The worker's too.
* #1659 item 2: the workspace in the log context is the caller's, read before sign-in. Only a
  workspace id (a UUID, at most 64 characters) is kept, in the API and the worker.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from pathlib import Path

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.monitoring.automatos_logging import request_log_fields  # noqa: E402
from core.monitoring.json_log_format import JsonLineFormatter  # noqa: E402
from core.monitoring.path_digests import DigestPathFilter, redact_path  # noqa: E402
from core.monitoring.request_context import MAX_WORKSPACE_ID_LEN, RequestContextMiddleware  # noqa: E402

WORKER_DIR = _ORCH.parent / "services" / "workspace-worker"
KEY = "1642" * 8   # 32 hex characters, the shape of a workspace webhook key (uuid4().hex)
OTHER_KEY = "1659" * 8
VERIFY = "verify-1642-marker"
CHALLENGE = "challenge-1642-marker"
INVITE = "invite-1642-marker"
WEBHOOK = "/api/webhooks/ws"
TEMPLATE = f"{WEBHOOK}/{{workspace_key}}"
WS = "0000abcd-0000-0000-0000-000000001659"
ACCESS_FORMAT = '%s - "%s %s HTTP/%s" %d'


def _access(path: str, method: str = "GET") -> logging.LogRecord:
    return logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, ACCESS_FORMAT,
                             ("203.0.113.9:4000", method, path, "1.1", 200), None)


# ── #1642: the access line ───────────────────────────────────────────────────

def test_the_access_line_holds_neither_the_webhook_key_nor_its_verification_query():
    record = _access(f"{WEBHOOK}/{KEY}?hub.mode=subscribe&hub.verify_token={VERIFY}&hub.challenge={CHALLENGE}")

    assert DigestPathFilter().filter(record) is True
    line = record.getMessage()
    assert KEY not in line and VERIFY not in line and CHALLENGE not in line
    assert line == f'203.0.113.9:4000 - "GET {TEMPLATE}?{{redacted}} HTTP/1.1" 200'


def test_a_pre_formatted_access_line_is_redacted_too():
    record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
                               f'203.0.113.9:4000 - "POST {WEBHOOK}/{KEY}?hub.verify_token={VERIFY} HTTP/1.1" 200',
                               None, None)

    assert DigestPathFilter().filter(record) is True
    assert record.getMessage() == f'203.0.113.9:4000 - "POST {TEMPLATE}?{{redacted}} HTTP/1.1" 200'


def test_the_invitation_lookups_token_is_dropped_and_other_queries_kept():
    invite, other = _access(f"/api/team/invitation-info?token={INVITE}"), _access("/api/webhooks/list?page=2")

    assert DigestPathFilter().filter(invite) is True and DigestPathFilter().filter(other) is True
    assert INVITE not in invite.getMessage() and "/api/team/invitation-info?{redacted}" in invite.getMessage()
    assert "/api/webhooks/list?page=2" in other.getMessage()


def test_every_copy_of_a_path_collapses_the_webhook_key():
    from core.monitoring.automatos_metrics import _normalize_path

    assert redact_path(f"{WEBHOOK}/{KEY}") == TEMPLATE
    assert redact_path(f"{WEBHOOK}/{KEY}/") == f"{TEMPLATE}/"
    assert {_normalize_path(f"{WEBHOOK}/{key}") for key in (KEY, OTHER_KEY)} == {TEMPLATE}
    for kept in ("/api/webhooks/workspace/x", "/api/agents/42", f"{WEBHOOK}/", ""):
        assert redact_path(kept) == kept, kept


# ── #1659 item 1: the JSON line's path is the route template ─────────────────

def _app(seen: dict) -> FastAPI:
    router = APIRouter(prefix="/api/webhooks")   # included with a prefix, as main.py's routers are

    @router.get("/ws/{workspace_key}")
    def verify(workspace_key: str):
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "in the route", (), None)
        seen["line"] = json.loads(JsonLineFormatter("s", "v", request_log_fields).format(record))
        return {"ok": True}

    app = FastAPI()
    app.include_router(router)
    app.add_middleware(RequestContextMiddleware)
    return app


def _as_the_server_sees_it(app, seen: dict):
    """The app behind a shim that reads the fields when the response starts, as uvicorn's access line does."""
    async def server(scope, receive, send):
        async def send_and_log(message):
            if message["type"] == "http.response.start":
                seen["access"] = request_log_fields()
            await send(message)

        await app(scope, receive, send_and_log)
    return server


def test_a_json_lines_path_is_the_route_template_not_the_key():
    seen: dict = {}
    response = TestClient(_as_the_server_sees_it(_app(seen), seen)).get(
        f"{WEBHOOK}/{KEY}", params={"hub.verify_token": VERIFY})

    assert response.status_code == 200
    assert seen["line"]["path"] == seen["access"]["path"] == TEMPLATE
    assert KEY not in json.dumps(seen) and VERIFY not in json.dumps(seen)


def test_with_no_route_the_path_is_the_redacted_path():
    seen: dict = {}
    response = TestClient(_as_the_server_sees_it(_app(seen), seen)).get(f"{WEBHOOK}/{KEY}/extra")

    assert response.status_code == 404
    assert seen["access"]["path"] == f"{TEMPLATE}/extra"


# ── #1659 item 2: the workspace is a workspace id or none ────────────────────

def _workspace_seen(value: str) -> str:
    seen: dict = {}
    app = FastAPI()

    @app.get("/api/things")
    def things():
        seen.update(request_log_fields())
        return {}

    app.add_middleware(RequestContextMiddleware)
    TestClient(app).get("/api/things", headers={"X-Workspace-ID": value})
    return seen["workspace_id"]


@pytest.mark.parametrize("sent", [
    "not-a-workspace", "1659", "__all__", "x" * 5000, WS + " " * MAX_WORKSPACE_ID_LEN, f"{WS}\r\nforged=1",
], ids=["junk", "integer", "all", "oversized", "padded-past-the-cap", "line-break"])
def test_a_junk_or_oversized_workspace_header_is_no_workspace(sent):
    assert _workspace_seen(sent) == ""


@pytest.mark.parametrize("sent", [WS, WS.upper(), WS.replace("-", ""), f"{{{WS}}}"])
def test_a_workspace_id_is_kept_in_its_canonical_form(sent):
    assert _workspace_seen(sent) == WS


# ── the worker ───────────────────────────────────────────────────────────────

def _worker_fields(monkeypatch, path: str) -> dict:
    """The fields a line written inside the worker's route sees, routed by Starlette as the worker is."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    monkeypatch.syspath_prepend(str(WORKER_DIR))
    context = importlib.import_module("worker_log_context")
    seen: dict = {}

    async def files(request):
        seen.update(context.request_log_fields())
        return JSONResponse({})

    app = Starlette(routes=[Route("/workspaces/{workspace_id}/files", files)])
    app.add_middleware(context.RequestLogContextMiddleware)
    TestClient(app).get(path, params={"path": "a.txt"})
    return seen


def test_the_workers_path_is_its_route_template_and_its_workspace_a_workspace_id(monkeypatch):
    fields = _worker_fields(monkeypatch, f"/workspaces/{WS}/files")
    assert fields["path"] == "/workspaces/{workspace_id}/files" and fields["workspace_id"] == WS


@pytest.mark.parametrize("named", ["junk", "x" * 300, WS.upper(), WS.replace("-", "")],
                         ids=["junk", "oversized", "upper-case", "bare-hex"])
def test_the_worker_keeps_no_workspace_that_is_not_a_workspace_id(monkeypatch, named):
    assert _worker_fields(monkeypatch, f"/workspaces/{named}/files")["workspace_id"] == ""
