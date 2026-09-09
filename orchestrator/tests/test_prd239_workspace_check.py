"""PRD-239 S6 — an agent's working directory is validated at save and explained
before it: valid or not, browsable in the Canvas as which root, inside the
paired host's allowed directories or not. Pure units + the route contract."""
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

from core.cli_runtime import validate_runtime_configuration, validate_working_directory  # noqa: E402
from services import cli_host_service as svc  # noqa: E402

WS = uuid4()


# ── validation ───────────────────────────────────────────────────────────────

def test_working_directory_rules():
    assert validate_working_directory(None) == [] and validate_working_directory("") == []
    assert validate_working_directory("/Users/me/repo") == []
    assert "absolute" in validate_working_directory("repo")[0]
    assert "'..'" in validate_working_directory("/Users/me/../etc")[0]
    assert "control character" in validate_working_directory("/tmp/x\n")[0]
    assert "string path" in validate_working_directory(42)[0]


def test_runtime_validation_covers_the_working_directory_for_cli_agents_only():
    bad = {"runtime": "cli", "provider": "claude", "model": None, "working_directory": "relative/path"}
    errors = validate_runtime_configuration(bad, cli_enabled=True)
    assert any("working_directory" in e for e in errors)
    api = {"runtime": "api", "working_directory": "relative/path"}
    assert validate_runtime_configuration(api, cli_enabled=True) == []


# ── the check ────────────────────────────────────────────────────────────────

class _HostQuery:
    def __init__(self, hosts):
        self._hosts = hosts

    def filter(self, *a, **k):
        return self

    def all(self):
        return self._hosts


class _DB:
    def __init__(self, hosts):
        self.hosts = hosts

    def query(self, model):
        return _HostQuery(self.hosts)


def _host(*allow_dirs):
    return SimpleNamespace(capabilities={"allow_dirs": list(allow_dirs)} if allow_dirs else {}, status="paired")


def test_check_names_the_canvas_root_and_the_hosts_verdict(monkeypatch):
    monkeypatch.setattr(svc.config, "LOCAL_PROJECTS_DIR", "/Users/me/Development", raising=False)
    db = _DB([_host("/Users/me/Development", "/Users/me/ws/workspaces")])
    out = svc.workspace_check(db, WS, "/Users/me/Development/automatos-ai")
    assert out["valid"] and out["errors"] == []
    assert out["explorer_root"] == "projects/automatos-ai" and out["browsable"] is True
    assert out["allowed"] is True and out["allowed_roots"] == ["/Users/me/Development", "/Users/me/ws/workspaces"]
    assert out["projects_dir"] == "/Users/me/Development"


def test_check_flags_a_folder_the_host_may_not_run_in(monkeypatch):
    monkeypatch.setattr(svc.config, "LOCAL_PROJECTS_DIR", "/Users/me/Development", raising=False)
    db = _DB([_host("/Users/me/ws/workspaces")])
    out = svc.workspace_check(db, WS, "/Users/me/Development/automatos-ai")
    assert out["browsable"] is True and out["allowed"] is False


def test_check_is_honest_when_nothing_is_known(monkeypatch):
    monkeypatch.setattr(svc.config, "LOCAL_PROJECTS_DIR", "", raising=False)
    out = svc.workspace_check(_DB([_host()]), WS, "/somewhere/else")
    assert out["valid"] and out["browsable"] is False and out["explorer_root"] is None
    assert out["allowed"] is None and out["allowed_roots"] == [] and out["projects_dir"] is None


def test_check_reports_an_invalid_path_without_mapping_it(monkeypatch):
    monkeypatch.setattr(svc.config, "LOCAL_PROJECTS_DIR", "/Users/me/Development", raising=False)
    out = svc.workspace_check(_DB([_host("/Users/me/Development")]), WS, "relative")
    assert out["valid"] is False and out["errors"] and out["browsable"] is False and out["allowed"] is None


def test_workspace_folder_paths_map_to_sessions_roots(monkeypatch):
    monkeypatch.setattr(svc.config, "LOCAL_PROJECTS_DIR", "", raising=False)
    out = svc.workspace_check(_DB([]), WS, f"/host/workspaces/{WS}/sessions/95")
    assert out["explorer_root"] == "sessions/95" and out["browsable"] is True


# ── the route ────────────────────────────────────────────────────────────────

def _client(monkeypatch, *, enabled=True):
    import api.cli_hosts as mod
    from core.database.database import get_db

    monkeypatch.setattr(mod.config, "CLI_RUNTIME_ENABLED", enabled, raising=False)
    app = FastAPI()
    app.include_router(mod.router)
    app.dependency_overrides[get_db] = lambda: SimpleNamespace(name="dummy-session")

    async def _operator_ctx():
        return SimpleNamespace(workspace_id=WS, user=SimpleNamespace(email="op@local"))

    app.dependency_overrides[mod._require_operator] = _operator_ctx
    return TestClient(app), mod


def test_route_returns_the_check_and_is_gated_by_session_mode(monkeypatch):
    client, mod = _client(monkeypatch)
    seen = {}
    monkeypatch.setattr(mod.svc, "workspace_check", lambda db, ws, path: seen.update({"ws": ws, "path": path}) or {"path": path, "valid": True})
    r = client.get("/api/v1/cli-hosts/workspace-check", params={"path": "/Users/me/repo"})
    assert r.status_code == 200 and r.json()["valid"] is True and seen == {"ws": WS, "path": "/Users/me/repo"}
    assert client.get("/api/v1/cli-hosts/workspace-check").status_code == 422  # path is required


def test_route_is_declared_in_the_mount_manifest():
    import json

    manifest = json.loads((_ORCH / "reports" / "route-manifest.json").read_text())
    routes = {(r["method"], r["path"]) for r in manifest["routes"]}
    assert ("GET", "/api/v1/cli-hosts/workspace-check") in routes


def test_the_projects_folder_itself_is_a_browsable_root_in_the_verdict_too(monkeypatch):
    """PRD-239 S6b: an agent rooted at LOCAL_PROJECTS_DIR browses all of it — the
    form's verdict and the Canvas (explorer_root_for) must say the same thing."""
    monkeypatch.setattr(svc.config, "LOCAL_PROJECTS_DIR", "/Users/me/Development", raising=False)
    db = _DB([_host("/Users/me/Development", "/Users/me/ws/workspaces")])
    out = svc.workspace_check(db, WS, "/Users/me/Development")
    assert out["valid"] and out["explorer_root"] == "projects" and out["browsable"] is True and out["allowed"] is True
    assert svc.explorer_root_for(7, "/Users/me/Development/", WS, "/Users/me/Development") == "projects"
    # a FILE at the root is still nothing — deliverables never live there
    assert svc.workspace_relative_path("/Users/me/Development", str(WS), "/Users/me/Development") is None
