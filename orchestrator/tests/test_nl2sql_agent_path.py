"""PRD-160 S1 — In-process, workspace-scoped NL2SQL agent path.

S1 re-enables ``smart_query_database`` / ``query_database`` as first-class Auto
tools after PRD-156 S3 disabled the unsafe unscoped path. The contract:

* **In-process** — the executor calls ``DatabaseKnowledgeService`` directly; no
  HTTP self-call (pinned by ``tests/security/test_nl2sql_tenancy.py``).
* **Workspace-scoped** — ``resolve_source_id`` only ever returns a source that
  belongs to the caller's workspace; an agent cannot reach another workspace's
  source by guessing its name (the cross-workspace matrix below).
* **Fail-closed** — no workspace context ⇒ refused (pinned in the tenancy suite).

The cross-workspace matrix is DB-backed (Postgres via the root ``db_session``
fixture). The in-process wiring test is pure-mock so it runs in the lean suite.
"""
from __future__ import annotations

import asyncio

import pytest


# --- Cross-workspace matrix: resolve_source_id never crosses a workspace -------

def _bare_service():
    """A DatabaseKnowledgeService whose only exercised method is the pure-DB
    ``resolve_source_id`` — bypass __init__ so the test needs no LLM/RAG deps."""
    from modules.nl2sql.service import DatabaseKnowledgeService

    return DatabaseKnowledgeService.__new__(DatabaseKnowledgeService)


def _add_source(db_session, workspace_id, name):
    from core.models.database_knowledge import DatabaseKnowledgeSource

    src = DatabaseKnowledgeSource(
        workspace_id=workspace_id,
        tenant_id=1,
        name=name,
        credential_id=1,
        dialect="postgresql",
        is_active=True,
        schema_metadata={},
    )
    db_session.add(src)
    db_session.flush()
    return src


@pytest.mark.usefixtures("db_session")
def test_resolve_source_id_workspace_matrix(db_session, seed_workspace):
    ws_a = seed_workspace(name="ws-a")
    ws_b = seed_workspace(name="ws-b")
    a_sales = _add_source(db_session, ws_a, "Sales")
    b_ops = _add_source(db_session, ws_b, "Ops")
    svc = _bare_service()

    run = lambda **kw: asyncio.run(svc.resolve_source_id(db_session=db_session, **kw))

    # named match resolves within the owning workspace
    assert run(workspace_id=ws_a, database_name="Sales") == str(a_sales.id)
    # case-insensitive
    assert run(workspace_id=ws_a, database_name="sales") == str(a_sales.id)
    # CROSS-WORKSPACE: ws_b cannot reach ws_a's "Sales"
    assert run(workspace_id=ws_b, database_name="Sales") is None
    # single active source in a workspace auto-resolves with no name
    assert run(workspace_id=ws_b, database_name=None) == str(b_ops.id)
    # no workspace context ⇒ never resolves
    assert run(workspace_id="", database_name="Sales") is None

    # add a second source to ws_a → no-name pick is now ambiguous ⇒ None
    _add_source(db_session, ws_a, "Analytics")
    assert run(workspace_id=ws_a, database_name=None) is None
    # but an explicit name still resolves unambiguously
    assert run(workspace_id=ws_a, database_name="Analytics") is not None


@pytest.mark.usefixtures("db_session")
def test_resolve_source_id_ignores_inactive_sources(db_session, seed_workspace):
    from core.models.database_knowledge import DatabaseKnowledgeSource

    ws = seed_workspace(name="ws-inactive")
    inactive = DatabaseKnowledgeSource(
        workspace_id=ws, tenant_id=1, name="Archived", credential_id=1,
        dialect="postgresql", is_active=False, schema_metadata={},
    )
    db_session.add(inactive)
    db_session.flush()
    svc = _bare_service()
    assert asyncio.run(
        svc.resolve_source_id(ws, "Archived", db_session=db_session)
    ) is None


# --- In-process wiring: executor → service.smart_query, scope threaded ---------

def test_smart_database_tool_threads_scope_in_process(monkeypatch):
    """The executor resolves within the workspace and calls the service object
    in-process (no HTTP), threading workspace_id / user_id / agent_id through."""
    from modules.tools.execution import exec_research

    calls: dict = {}

    class FakeService:
        async def resolve_source_id(self, workspace_id, database_name=None, db_session=None):
            calls["resolve"] = {"workspace_id": workspace_id, "database_name": database_name}
            return "7"

        async def smart_query(self, source_id, text, user_id, agent_id=None, workspace_id=None):
            calls["smart_query"] = {
                "source_id": source_id, "text": text, "user_id": user_id,
                "agent_id": agent_id, "workspace_id": workspace_id,
            }
            return {"success": True, "sql": "SELECT 1", "data": [{"n": 1}],
                    "columns": ["n"], "row_count": 1}

    monkeypatch.setattr(
        "modules.nl2sql.get_database_knowledge_service", lambda: FakeService()
    )

    result = asyncio.run(
        exec_research.execute_smart_database_tool(
            executor=None,
            tool_name="smart_query_database",
            parameters={"query": "how many users?", "database_name": "Sales"},
            agent_id=42,
            workspace_id="ws-1",
            caller_context={"user_id": "u-9"},
        )
    )

    assert result["success"] is True and result["row_count"] == 1
    assert calls["resolve"] == {"workspace_id": "ws-1", "database_name": "Sales"}
    assert calls["smart_query"]["source_id"] == "7"
    assert calls["smart_query"]["workspace_id"] == "ws-1"
    assert calls["smart_query"]["user_id"] == "u-9"
    assert calls["smart_query"]["agent_id"] == "42"


def test_smart_database_tool_reports_missing_source(monkeypatch):
    """When no source resolves in the workspace, the tool returns a helpful,
    leak-free error rather than querying anything."""
    from modules.tools.execution import exec_research

    class FakeService:
        async def resolve_source_id(self, workspace_id, database_name=None, db_session=None):
            return None

        async def smart_query(self, *a, **k):  # pragma: no cover - must not be called
            raise AssertionError("smart_query must not run when no source resolves")

    monkeypatch.setattr(
        "modules.nl2sql.get_database_knowledge_service", lambda: FakeService()
    )

    result = asyncio.run(
        exec_research.execute_smart_database_tool(
            None, "smart_query_database",
            {"query": "x", "database_name": "Nope"}, 1,
            workspace_id="ws-1", caller_context={"user_id": "u"},
        )
    )
    assert result["success"] is False
    assert "Nope" in result["error"]
