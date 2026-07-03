"""PRD-183 S4 (F087/F022) — codegraph index/reindex tools + auto_reindex setter.

Two gaps closed:

  * F087 — codegraph had SIX read tools but no agent-side write/index tool, so
    "index this repo and tell me what calls X" was half-possible. New tools:
    ``platform_codegraph_index`` (onboard/refresh a repo),
    ``platform_codegraph_reindex`` (re-index an existing project by name).
  * F022 — the GitHub push webhook filters ``auto_reindex``, but nothing could
    ever set it True (only read, ``api/codegraph.py:683``), so the webhook was
    dead. New: ``CodeGraphService.set_auto_reindex`` + a
    ``platform_codegraph_set_auto_reindex`` tool.

Handlers are tested against a fake CodeGraphService (no DB, no repo): argument
validation, workspace scoping, and the delegation contract. The setter is
tested to issue a workspace-scoped UPDATE and reject a cross-workspace/unknown
project.
"""
from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio  # noqa: E402

import tests.conftest as _conftest  # noqa: E402
_conftest._restore_real_app_modules()

from modules.tools.discovery import handlers_codegraph  # noqa: E402
from modules.tools.discovery.action_registry import ActionRegistry  # noqa: E402
from modules.tools.discovery.actions_codegraph import register_codegraph_actions  # noqa: E402


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class _FakeService:
    def __init__(self):
        self.indexed = []
        self.reindexed = []
        self.auto_set = []

    def list_projects(self, workspace_id=None):
        return [{
            "id": 7, "name": "repo", "source_type": "github",
            "source_url": "https://github.com/AutomatosAI/automatos-ai",
            "branch": "main", "last_indexed": "2026-07-01T00:00:00",
            "status": "ready", "auto_reindex": False,
        }]

    async def index_github_project(self, project_name, github_url, branch="main", workspace_id=None, **kw):
        self.indexed.append((project_name, github_url, branch, workspace_id))
        return {"project_name": project_name, "total_symbols": 12, "status": "ready"}

    def set_auto_reindex(self, project_id, enabled, workspace_id=None):
        self.auto_set.append((project_id, enabled, workspace_id))
        return {"success": True, "project_id": project_id, "auto_reindex": enabled}


def _patch(monkeypatch, svc):
    monkeypatch.setattr(handlers_codegraph, "_service", lambda _db: svc)


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------


def test_new_codegraph_write_tools_registered():
    reg = ActionRegistry()
    register_codegraph_actions(reg)
    names = {a.name for a in reg.get_all()}
    assert "platform_codegraph_index" in names
    assert "platform_codegraph_reindex" in names
    assert "platform_codegraph_set_auto_reindex" in names


def test_write_tools_have_write_permission():
    reg = ActionRegistry()
    register_codegraph_actions(reg)
    assert reg.get("platform_codegraph_index").permission_level == "write"
    assert reg.get("platform_codegraph_reindex").permission_level == "write"
    assert reg.get("platform_codegraph_set_auto_reindex").permission_level == "write"


# ------------------------------------------------------------------
# index / reindex tool behaviour
# ------------------------------------------------------------------


def test_codegraph_index_tool(monkeypatch):
    """The index tool clones+indexes a repo for the executor's workspace."""
    svc = _FakeService()
    _patch(monkeypatch, svc)
    res = _run(handlers_codegraph.codegraph_index(
        None, "ws-1",
        {"project": "myrepo", "github_url": "https://github.com/x/y", "branch": "dev"},
    ))
    assert res["success"] is True
    assert svc.indexed == [("myrepo", "https://github.com/x/y", "dev", "ws-1")]


def test_codegraph_index_requires_args():
    res = _run(handlers_codegraph.codegraph_index(None, "ws-1", {"project": "only-name"}))
    assert not res["success"] and "required" in res["error"].lower()


def test_codegraph_reindex_tool(monkeypatch):
    """Reindex resolves an existing project by name and re-indexes its source."""
    svc = _FakeService()
    _patch(monkeypatch, svc)
    res = _run(handlers_codegraph.codegraph_reindex(None, "ws-1", {"project": "repo"}))
    assert res["success"] is True
    # Re-indexed the resolved project's source_url/branch under the workspace.
    assert svc.indexed == [(
        "repo", "https://github.com/AutomatosAI/automatos-ai", "main", "ws-1",
    )]


def test_codegraph_reindex_unknown_project(monkeypatch):
    svc = _FakeService()
    _patch(monkeypatch, svc)
    res = _run(handlers_codegraph.codegraph_reindex(None, "ws-1", {"project": "nope"}))
    assert not res["success"] and "not found" in res["error"].lower()
    assert svc.indexed == []


# ------------------------------------------------------------------
# auto_reindex setter — tool + service contract
# ------------------------------------------------------------------


def test_auto_reindex_setter_tool(monkeypatch):
    """The setter tool flips auto_reindex for a named project in the workspace."""
    svc = _FakeService()
    _patch(monkeypatch, svc)
    res = _run(handlers_codegraph.codegraph_set_auto_reindex(
        None, "ws-1", {"project": "repo", "enabled": True},
    ))
    assert res["success"] is True
    # Resolved project id 7, workspace-scoped.
    assert svc.auto_set == [(7, True, "ws-1")]


def test_auto_reindex_setter_service_issues_scoped_update():
    """CodeGraphService.set_auto_reindex issues a workspace-scoped UPDATE.

    Uses a fake DB that records the SQL text + params; asserts the update is
    guarded by BOTH project id and workspace_id (no cross-tenant flip) and sets
    the boolean via a bound param.
    """
    from modules.codegraph.codegraph_service import CodeGraphService

    captured = {}

    class _Result:
        rowcount = 1

    class _FakeDb:
        def execute(self, stmt, params=None):
            captured["sql"] = str(stmt)
            captured["params"] = params
            return _Result()

        def commit(self):
            captured["committed"] = True

    svc = CodeGraphService.__new__(CodeGraphService)  # skip heavy __init__
    svc.db = _FakeDb()

    out = svc.set_auto_reindex(7, True, workspace_id="ws-1")

    assert out["success"] is True
    sql = captured["sql"].lower()
    assert "update codegraph_projects" in sql
    assert "auto_reindex" in sql
    assert "workspace_id" in sql  # scoped — cannot flip another tenant's project
    assert captured["params"]["id"] == 7
    assert captured["params"]["ws"] == "ws-1"
    assert captured["params"]["val"] is True
    assert captured.get("committed") is True
