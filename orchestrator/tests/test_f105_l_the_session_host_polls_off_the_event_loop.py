"""F105 (4 Nov): the session host's polls run in the threadpool, never on the event loop.

A paired CLI host calls ``POST /api/v1/cli-hosts/{id}/heartbeat`` and ``…/claim``
continuously. Both routes and their host-token dependency (``require_cli_host``) were
``async def`` over a synchronous session, so every poll's token lookup and every
heartbeat or claim (a claim can wait on a field-memory read for seconds) ran on the event
loop. They are plain ``def``s now, so FastAPI runs them in its threadpool. The answers
are unchanged.

DB-free: the service is patched to record the thread it runs on.
"""
from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import api.cli_hosts as mod
from core.database.database import get_db

WS = uuid4()
HOST_ID = uuid4()
TOKEN = "f105-host-token"
POLLED = ("require_cli_host", "heartbeat", "claim")


def _client(monkeypatch, threads):
    monkeypatch.setattr(mod.config, "CLI_RUNTIME_ENABLED", True, raising=False)
    app = FastAPI()
    app.include_router(mod.router)
    app.dependency_overrides[get_db] = lambda: SimpleNamespace(name="request-session")

    @app.middleware("http")
    async def _on_the_loop(request: Request, call_next):
        threads["loop"] = threading.get_ident()
        return await call_next(request)

    def _resolve(db, token):
        threads["token lookup"] = threading.get_ident()
        return SimpleNamespace(id=HOST_ID, workspace_id=WS, name="laptop", status="paired") if token == TOKEN else None

    def _heartbeat(db, host, caps, running):
        threads["heartbeat"] = threading.get_ident()
        return {"reattached": [], "stale": [], "server_time": "t"}

    def _claim(db, host, limit):
        threads["claim"] = threading.get_ident()
        return {"tasks": [], "parked": []}

    monkeypatch.setattr(mod.svc, "resolve_host_by_token", _resolve, raising=True)
    monkeypatch.setattr(mod.svc, "record_heartbeat", _heartbeat, raising=True)
    monkeypatch.setattr(mod.svc, "claim_for_host", _claim, raising=True)
    return TestClient(app)


@pytest.mark.parametrize("name", POLLED)
def test_the_polled_route_and_its_dependency_are_plain_defs(name):
    assert not inspect.iscoroutinefunction(getattr(mod, name)), f"{name} must be a plain def (F105)"


@pytest.mark.parametrize("poll,body", [("heartbeat", {"running": []}), ("claim", {"limit": 1})])
def test_a_poll_runs_off_the_event_loop_and_answers_as_before(monkeypatch, poll, body):
    threads = {}
    client = _client(monkeypatch, threads)

    answer = client.post(f"/api/v1/cli-hosts/{HOST_ID}/{poll}", json=body, headers={mod.HOST_TOKEN_HEADER: TOKEN})

    assert answer.status_code == 200
    assert threads["token lookup"] != threads["loop"]
    assert threads[poll] != threads["loop"]


def test_a_bad_token_is_still_refused(monkeypatch):
    client = _client(monkeypatch, {})

    answer = client.post(f"/api/v1/cli-hosts/{HOST_ID}/claim", json={"limit": 1},
                         headers={mod.HOST_TOKEN_HEADER: "not-the-token"})

    assert answer.status_code == 401
    assert answer.json()["detail"] == "invalid or missing CLI host token"
