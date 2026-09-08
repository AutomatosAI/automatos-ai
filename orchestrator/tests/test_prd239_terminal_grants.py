"""PRD-239 S7 — terminal grants: minted for the operator against a paired host's
announced terminal port, scoped to a ticket's real directory or an allowed
folder, delivered once on the host's next heartbeat, and expired otherwise.
Pure units (process store, no Redis) + the route contract."""
from __future__ import annotations

import os
import sys
import time
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

from services import cli_host_service as svc  # noqa: E402

WS = uuid4()


class _Query:
    def __init__(self, result):
        self._result = result

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._result

    def all(self):
        return self._result if isinstance(self._result, list) else []


class _DB:
    """``query(BoardTask)`` → the task; ``query(CliHost)`` → the hosts list."""

    def __init__(self, task=None, hosts=None):
        self.task = task
        self.hosts = hosts or []

    def query(self, model):
        return _Query(self.hosts if model.__name__ == "CliHost" else self.task)


def _host(port=48123, allow_dirs=("/Users/me/Development",)):
    return SimpleNamespace(id=uuid4(), workspace_id=WS, status="paired",
                           capabilities={"terminal_port": port, "allow_dirs": list(allow_dirs)})


@pytest.fixture(autouse=True)
def _no_redis(monkeypatch):
    monkeypatch.setattr(svc, "_redis", lambda: None)
    svc._TERMINAL_GRANTS.clear()
    yield
    svc._TERMINAL_GRANTS.clear()


def test_a_grant_for_a_ticket_opens_in_the_sessions_real_directory():
    host = _host()
    task = SimpleNamespace(id=95, workspace_id=WS, runtime_ref={"cwd": "/w/.claude/worktrees/automatos-95"})
    out = svc.mint_terminal_grant(_DB(task=task, hosts=[host]), host, task_id=95)
    assert out["port"] == 48123 and out["cwd"] == "/w/.claude/worktrees/automatos-95" and out["task_id"] == "95"
    assert out["ws_url"] == f"ws://127.0.0.1:48123/terminal?token={out['token']}"
    delivered = svc.pop_terminal_grants(host.id)
    assert [g["token"] for g in delivered] == [out["token"]] and delivered[0]["cwd"] == out["cwd"]
    assert svc.pop_terminal_grants(host.id) == []  # delivered once


def test_a_grant_for_a_folder_must_be_inside_the_hosts_allowed_directories():
    host = _host()
    db = _DB(hosts=[host])
    ok = svc.mint_terminal_grant(db, host, cwd="/Users/me/Development/repo")
    assert ok["cwd"] == "/Users/me/Development/repo" and ok["task_id"] is None
    with pytest.raises(PermissionError):
        svc.mint_terminal_grant(db, host, cwd="/etc")
    with pytest.raises(ValueError):
        svc.mint_terminal_grant(db, host, cwd="relative/path")


def test_a_host_without_a_terminal_port_and_an_unknown_ticket_are_refused():
    with pytest.raises(LookupError):
        svc.mint_terminal_grant(_DB(), _host(port=None))
    with pytest.raises(LookupError):
        svc.mint_terminal_grant(_DB(task=None), _host(), task_id=404)


def test_expired_grants_are_not_delivered(monkeypatch):
    host = _host()
    svc.push_terminal_grant(host.id, {"token": "old", "cwd": None, "task_id": None, "expires_at": time.time() - 1})
    svc.push_terminal_grant(host.id, {"token": "new", "cwd": None, "task_id": None, "expires_at": time.time() + 60})
    assert [g["token"] for g in svc.pop_terminal_grants(host.id)] == ["new"]


# ── the route ────────────────────────────────────────────────────────────────

def _client(monkeypatch, host):
    import api.cli_hosts as mod
    from core.database.database import get_db

    monkeypatch.setattr(mod.config, "CLI_RUNTIME_ENABLED", True, raising=False)
    app = FastAPI()
    app.include_router(mod.router)
    app.dependency_overrides[get_db] = lambda: _DB(hosts=[host])

    async def _operator_ctx():
        return SimpleNamespace(workspace_id=WS, user=SimpleNamespace(email="op@local"))

    app.dependency_overrides[mod._require_operator] = _operator_ctx
    return TestClient(app), mod


def test_route_mints_a_grant_and_maps_the_refusals(monkeypatch):
    host = _host()
    client, mod = _client(monkeypatch, host)
    monkeypatch.setattr(mod.svc, "mint_terminal_grant", lambda db, h, cwd=None, task_id=None: {"token": "t", "ws_url": "ws://127.0.0.1:1/terminal?token=t", "cwd": cwd, "task_id": task_id})
    # the route's own host lookup uses the fake db's CliHost query (first() → the list) — patch it to return the host
    monkeypatch.setattr(_Query, "first", lambda self: (self._result[0] if isinstance(self._result, list) and self._result else self._result))
    r = client.post(f"/api/v1/cli-hosts/{host.id}/terminal", json={"task_id": 95})
    assert r.status_code == 200 and r.json()["ws_url"].startswith("ws://127.0.0.1:")

    def _refuse(db, h, cwd=None, task_id=None):
        raise PermissionError("outside")

    monkeypatch.setattr(mod.svc, "mint_terminal_grant", _refuse)
    assert client.post(f"/api/v1/cli-hosts/{host.id}/terminal", json={"cwd": "/etc"}).status_code == 403


def test_route_is_declared_in_the_mount_manifest():
    import json

    manifest = json.loads((_ORCH / "reports" / "route-manifest.json").read_text())
    routes = {(r["method"], r["path"]) for r in manifest["routes"]}
    assert ("POST", "/api/v1/cli-hosts/{host_id}/terminal") in routes
