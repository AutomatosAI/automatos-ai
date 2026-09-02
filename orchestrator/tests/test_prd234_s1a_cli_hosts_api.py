"""PRD-234 S1a — the CLI host routes: gate, pairing, host-token auth, contract shapes.

Router mounted on a bare FastAPI app with dependency overrides — no DB, no real
auth — mirroring ``test_prd228_fleet_route.py``. The service layer is spied so
these tests prove the ROUTE contract: 404 everywhere while the flag is off,
401 on a bad or missing host token, a host cannot act as another host, and the
response shapes the CLI host (S1b) is written against.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.cli_hosts as mod  # noqa: E402
from core.database.database import get_db  # noqa: E402

WS = uuid4()
HOST_ID = uuid4()
OTHER_HOST_ID = uuid4()
TOKEN = "good-token"


def _host(host_id=HOST_ID):
    return SimpleNamespace(id=host_id, workspace_id=WS, name="laptop", status="paired")


def _client(monkeypatch, *, enabled=True):
    monkeypatch.setattr(mod.config, "CLI_RUNTIME_ENABLED", enabled, raising=False)
    app = FastAPI()
    app.include_router(mod.router)
    app.dependency_overrides[get_db] = lambda: SimpleNamespace(name="dummy-session")

    async def _operator_ctx():
        return SimpleNamespace(workspace_id=WS, user=SimpleNamespace(email="op@local"))

    app.dependency_overrides[mod._require_operator] = _operator_ctx
    # Host auth resolves through the (spied) service: only TOKEN maps to HOST_ID.
    monkeypatch.setattr(
        mod.svc, "resolve_host_by_token",
        lambda db, token: _host() if token == TOKEN else None, raising=True,
    )
    return TestClient(app)


def _h(token=TOKEN):
    return {mod.HOST_TOKEN_HEADER: token}


# ── the gate ─────────────────────────────────────────────────────────────────

def test_every_route_is_404_while_session_mode_is_off(monkeypatch):
    c = _client(monkeypatch, enabled=False)
    assert c.post("/api/v1/cli-hosts/pair", json={"code": "ABCD-EFGH"}).status_code == 404
    assert c.post(f"/api/v1/cli-hosts/{HOST_ID}/claim", json={"limit": 1}, headers=_h()).status_code == 404
    assert c.post(f"/api/v1/cli-hosts/{HOST_ID}/heartbeat", json={}, headers=_h()).status_code == 404


# ── pairing ──────────────────────────────────────────────────────────────────

def test_pairing_code_is_issued_once_with_the_command_to_run(monkeypatch):
    c = _client(monkeypatch)
    from datetime import datetime, timezone
    monkeypatch.setattr(
        mod.svc, "create_pairing_code",
        lambda db, ws, name: (_host(), "ABCD-2345", datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)),
        raising=True,
    )
    r = c.post("/api/v1/cli-hosts/pairing-codes", json={"name": "laptop"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["code"] == "ABCD-2345" and body["host_id"] == str(HOST_ID)
    assert body["pair_command"] == "make cli-host PAIR=ABCD-2345"


def test_pair_exchanges_a_code_for_a_token_exactly_once(monkeypatch):
    c = _client(monkeypatch)
    monkeypatch.setattr(
        mod.svc, "pair_host",
        lambda db, code, name, caps: (_host(), "minted-token") if code == "ABCD-2345" else None,
        raising=True,
    )
    ok = c.post("/api/v1/cli-hosts/pair", json={"code": "ABCD-2345", "name": "laptop"})
    assert ok.status_code == 200 and ok.json()["token"] == "minted-token"
    assert ok.json()["token_header"] == mod.HOST_TOKEN_HEADER
    assert ok.json()["workspace_id"] == str(WS)
    bad = c.post("/api/v1/cli-hosts/pair", json={"code": "WRONG-CODE"})
    assert bad.status_code == 401


# ── host-token auth ──────────────────────────────────────────────────────────

def test_host_routes_reject_a_missing_or_wrong_token(monkeypatch):
    c = _client(monkeypatch)
    assert c.post(f"/api/v1/cli-hosts/{HOST_ID}/claim", json={"limit": 1}).status_code == 401
    assert c.post(f"/api/v1/cli-hosts/{HOST_ID}/claim", json={"limit": 1}, headers=_h("nope")).status_code == 401


def test_a_valid_token_cannot_act_as_another_host(monkeypatch):
    c = _client(monkeypatch)
    r = c.post(f"/api/v1/cli-hosts/{OTHER_HOST_ID}/claim", json={"limit": 1}, headers=_h())
    assert r.status_code == 401


# ── the contract shapes ──────────────────────────────────────────────────────

def test_claim_returns_the_tickets_the_service_claimed(monkeypatch):
    c = _client(monkeypatch)
    seen = {}

    def _claim(db, host, limit):
        seen["host"], seen["limit"] = host.id, limit
        return {"tasks": [{"task_id": 5, "session_id": "s-1", "attempt": 1, "prompt": "do it"}],
                "parked": [{"task_id": 6, "title": "held", "reason": "Awaiting human approval (grant #9)"}]}

    monkeypatch.setattr(mod.svc, "claim_for_host", _claim, raising=True)
    r = c.post(f"/api/v1/cli-hosts/{HOST_ID}/claim", json={"limit": 3}, headers=_h())
    assert r.status_code == 200 and r.json()["tasks"][0]["session_id"] == "s-1"
    assert r.json()["parked"][0]["task_id"] == 6
    assert seen == {"host": HOST_ID, "limit": 3}


def test_claim_limit_is_bounded(monkeypatch):
    c = _client(monkeypatch)
    assert c.post(f"/api/v1/cli-hosts/{HOST_ID}/claim", json={"limit": 0}, headers=_h()).status_code == 422
    assert c.post(f"/api/v1/cli-hosts/{HOST_ID}/claim", json={"limit": 999}, headers=_h()).status_code == 422


def test_heartbeat_forwards_capabilities_and_running_sessions(monkeypatch):
    c = _client(monkeypatch)
    seen = {}

    def _hb(db, host, caps, running):
        seen["caps"], seen["running"] = caps, running
        return {"reattached": [], "stale": [9], "server_time": "t"}

    monkeypatch.setattr(mod.svc, "record_heartbeat", _hb, raising=True)
    r = c.post(
        f"/api/v1/cli-hosts/{HOST_ID}/heartbeat",
        json={"capabilities": {"claude": "2.1.236"}, "running": [{"task_id": 9, "session_id": "s-9"}]},
        headers=_h(),
    )
    assert r.status_code == 200 and r.json()["stale"] == [9]
    assert seen["caps"] == {"claude": "2.1.236"}
    assert seen["running"][0]["task_id"] == 9 and seen["running"][0]["session_id"] == "s-9"


def test_events_and_result_map_service_errors_to_http(monkeypatch):
    c = _client(monkeypatch)

    def _events(db, host, task_id, events):
        if task_id == 404:
            raise LookupError("no such task")
        if task_id == 403:
            raise PermissionError("not yours")
        return {"status": "in_progress", "lease_renewed": True, "control": ["cancel"]}

    async def _result(db, host, task_id, payload):
        return {"applied": True, "status": "done", "echo": payload["status"]}

    monkeypatch.setattr(mod.svc, "record_events", _events, raising=True)
    monkeypatch.setattr(mod.svc, "apply_result", _result, raising=True)
    base = f"/api/v1/cli-hosts/{HOST_ID}/tasks"
    assert c.post(f"{base}/404/events", json={"events": []}, headers=_h()).status_code == 404
    assert c.post(f"{base}/403/events", json={"events": []}, headers=_h()).status_code == 403
    ok = c.post(f"{base}/1/events", json={"events": [{"event": "PreToolUse", "tool_name": "Bash"}]}, headers=_h())
    assert ok.status_code == 200 and ok.json()["control"] == ["cancel"]
    res = c.post(f"{base}/1/result", json={"status": "success", "result_text": "done", "attempt": 1}, headers=_h())
    assert res.status_code == 200 and res.json() == {"applied": True, "status": "done", "echo": "success"}
    bad = c.post(f"{base}/1/result", json={"status": "exploded"}, headers=_h())
    assert bad.status_code == 422


def test_router_is_declared_in_the_mount_manifest():
    import router_manifest as rm

    assert 'RouterSpec("api.cli_hosts")' in Path(rm.__file__).read_text(encoding="utf-8")
