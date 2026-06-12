"""PRD-165 S4 — codegraph platform tools + GitHub-App token seam.

Handlers are tested against a fake CodeGraphService (no DB, no repo): argument
validation, project resolution, the staleness stamp on every result, and the
semantic/fuzzy routing of the search tool. The GitHub token resolver is tested
for its PAT fallback (no App configured).
"""
from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio

import tests.conftest as _conftest  # noqa: E402
_conftest._restore_real_app_modules()

from modules.tools.discovery import handlers_codegraph  # noqa: E402
import modules.codegraph.github_auth as github_auth  # noqa: E402


class _FakeService:
    def list_projects(self, workspace_id=None):
        return [{
            "id": 1, "name": "repo", "source_type": "github",
            "last_indexed": "2026-06-12T00:00:00", "status": "ready",
            "auto_reindex": True,
        }]

    async def search_symbols(self, project, query, symbol_type=None, limit=10, workspace_id=None):
        if project != "repo":
            raise ValueError(f"Project '{project}' not found")
        return {"project": project, "query": query, "count": 1, "results": [
            {"name": query, "file_path": "a.py", "line_number": 5, "signature": "def x()"}
        ]}

    async def get_call_graph(self, project, symbol, depth=1, direction="outgoing", workspace_id=None):
        return {"project": project, "symbol": symbol, "nodes": [], "edges": []}

    async def find_dependencies(self, project_id, symbol_name, direction="both", workspace_id=None):
        return {"project_id": project_id, "symbol": symbol_name, "dependents": [], "dependencies": []}

    async def analyze_architecture(self, project_id, workspace_id=None, focus_path=None):
        return {"project_id": project_id, "modules": []}

    async def semantic_search(self, project, query, limit=10, workspace_id=None):
        return {"project": project, "query": query, "count": 0, "results": []}


def _patch(monkeypatch, service=None):
    monkeypatch.setattr(handlers_codegraph, "_service", lambda _db: service or _FakeService())
    monkeypatch.setattr(handlers_codegraph, "_result_limit", lambda _p: 10)


def test_list_projects(monkeypatch):
    _patch(monkeypatch)
    res = asyncio.run(handlers_codegraph.codegraph_list_projects(None, "ws", {}))
    assert res["success"] and res["project_count"] == 1


def test_get_symbol_success_carries_staleness(monkeypatch):
    _patch(monkeypatch)
    res = asyncio.run(handlers_codegraph.codegraph_get_symbol(None, "ws", {"project": "repo", "symbol": "x"}))
    assert res["success"]
    assert res["last_indexed"] == "2026-06-12T00:00:00"  # staleness stamp
    assert res["results"][0]["file_path"] == "a.py"


def test_get_symbol_requires_args():
    res = asyncio.run(handlers_codegraph.codegraph_get_symbol(None, "ws", {"project": "repo"}))
    assert not res["success"] and "required" in res["error"].lower()


def test_get_symbol_unknown_project(monkeypatch):
    _patch(monkeypatch)
    res = asyncio.run(handlers_codegraph.codegraph_get_symbol(None, "ws", {"project": "nope", "symbol": "x"}))
    assert not res["success"] and "not found" in res["error"].lower()


def test_dependencies_resolves_project_id(monkeypatch):
    _patch(monkeypatch)
    res = asyncio.run(handlers_codegraph.codegraph_dependencies(None, "ws", {"project": "repo", "symbol": "x"}))
    assert res["success"]
    assert res["last_indexed"]  # staleness stamp present


def test_architecture_unknown_project(monkeypatch):
    class _Empty(_FakeService):
        def list_projects(self, workspace_id=None):
            return []
    _patch(monkeypatch, _Empty())
    res = asyncio.run(handlers_codegraph.codegraph_architecture(None, "ws", {"project": "repo"}))
    assert not res["success"] and "not found" in res["error"].lower()


def test_search_routes_semantic_then_fuzzy(monkeypatch):
    _patch(monkeypatch)
    sem = asyncio.run(handlers_codegraph.codegraph_search(None, "ws", {"project": "repo", "query": "x"}))
    assert sem["mode"] == "semantic"
    fuzzy = asyncio.run(handlers_codegraph.codegraph_search(None, "ws", {"project": "repo", "query": "x", "mode": "fuzzy"}))
    assert fuzzy["mode"] == "fuzzy"


def test_call_graph_clamps_depth(monkeypatch):
    captured = {}

    class _DepthSvc(_FakeService):
        async def get_call_graph(self, project, symbol, depth=1, direction="outgoing", workspace_id=None):
            captured["depth"] = depth
            return {"nodes": [], "edges": []}

    _patch(monkeypatch, _DepthSvc())
    asyncio.run(handlers_codegraph.codegraph_call_graph(None, "ws", {"project": "repo", "symbol": "x", "depth": 99}))
    assert captured["depth"] == 5  # clamped to the 1..5 range


# ---------------------------------------------------------------------------
# GitHub-App token seam — PAT fallback when no App is configured
# ---------------------------------------------------------------------------

def test_resolve_github_token_falls_back_to_pat(monkeypatch):
    class _Cfg:
        GITHUB_PAT = "pat_abc"
        GITHUB_APP_ID = ""
        GITHUB_APP_PRIVATE_KEY = ""
        GITHUB_APP_INSTALLATION_ID = ""

    monkeypatch.setattr("config.Config", _Cfg)
    assert asyncio.run(github_auth.resolve_github_token()) == "pat_abc"


def test_resolve_github_token_none_when_nothing_configured(monkeypatch):
    class _Cfg:
        GITHUB_PAT = ""
        GITHUB_APP_ID = ""
        GITHUB_APP_PRIVATE_KEY = ""
        GITHUB_APP_INSTALLATION_ID = ""

    monkeypatch.setattr("config.Config", _Cfg)
    assert asyncio.run(github_auth.resolve_github_token()) is None
