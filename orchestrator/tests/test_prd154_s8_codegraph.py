"""PRD-154 S8 — CodeGraph semantic search routes to the working pgvector path;
query-log schema reconciled + made non-fatal; results carry path:line + signature.

Verified ground truth (main 2026-06-10):
  * modules/agents/services/agent_platform_tools.py ``search_codebase`` advertises
    a ``search_type`` enum ["fuzzy","semantic"] but the handler IGNORED it and
    always called ``search_symbols`` (fuzzy) — ``semantic_search`` was unreachable
    from agents, even though the pgvector path works.
  * the ``len(code_snippet.strip()) < 50`` filter dropped most short symbols.
  * ``codegraph_query_logs`` (20260218_fix_codegraph_schema_v2) has columns
    ``duration_ms`` (Float) and ``workspace_id`` (UUID NOT NULL), but every
    INSERT wrote a non-existent ``execution_time_ms`` and omitted ``workspace_id``
    -> every query-log write failed; an uncaught commit could fail the search.

No DB, no network: a fake db distinguishes SELECT vs the query-log INSERT, and
the service is built via ``__new__`` to skip the heavy embedding/vector-store
constructor. The tool handler is built with patched RAG/CodeGraph construction.
"""
from __future__ import annotations

import os

# Dummy POSTGRES_* satisfies the config import chain (blessed pattern); the port
# points at nothing so any stray connect fails fast. CI exports real values.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# CI collection-order safety net: this module imports the real modules.* chain
# at collection time; restore real app modules before that (no-op once conftest
# has run, which it always has under pytest).
import tests.conftest as _conftest  # noqa: E402
_conftest._restore_real_app_modules()

from modules.codegraph.codegraph_service import CodeGraphService  # noqa: E402

_SERVICE_SRC = (
    Path(__file__).resolve().parents[1]
    / "modules" / "codegraph" / "codegraph_service.py"
).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeResult:
    def __init__(self, one=None, many=None):
        self._one = one
        self._many = many or []

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._many


class _LogFailingDB:
    """Returns seeded rows for SELECTs and RAISES on the query-log INSERT.

    Proves the query-log write is best-effort: a failure must not fail the
    search (the row is rolled back, the search still returns its results)."""

    def __init__(self, project_row, symbol_rows):
        self.project_row = project_row
        self.symbol_rows = symbol_rows
        self.commits = 0
        self.rollbacks = 0
        self.log_attempts = 0

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "INSERT INTO codegraph_query_logs" in sql:
            self.log_attempts += 1
            raise RuntimeError("induced query-log failure")
        if "FROM codegraph_projects" in sql:
            return _FakeResult(one=self.project_row)
        # the symbol search SELECT
        return _FakeResult(many=self.symbol_rows)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _symbol_row(similarity=None):
    base = dict(
        id=1,
        symbol_type="function",
        name="authenticate_user",
        qualified_name="api.auth.authenticate_user",
        file_path="api/auth.py",
        line_number=42,
        signature="def authenticate_user(token: str) -> User",
        docstring="Validate a token and return the user.",
        code_snippet="def authenticate_user(token):\n    return verify(token)",
    )
    if similarity is not None:
        base["similarity"] = similarity
    return SimpleNamespace(**base)


def _make_service(db):
    """Build a CodeGraphService without its heavy embedding/vector constructor."""
    svc = CodeGraphService.__new__(CodeGraphService)
    svc.db = db
    svc.embedding_manager = MagicMock()
    svc.embedding_manager.generate_embedding = AsyncMock(
        return_value=[0.1, 0.2, 0.3, 0.4]
    )
    svc._vector_store = None  # force the SQL pgvector fallback (the working path)
    return svc


def _make_tools(code_graph):
    """Build AgentPlatformTools with patched construction, then inject a mock
    CodeGraphService so we can observe routing."""
    from modules.agents.services.agent_platform_tools import AgentPlatformTools

    with patch("modules.agents.services.agent_platform_tools.RAGService"), patch(
        "modules.agents.services.agent_platform_tools.CodeGraphService"
    ):
        tools = AgentPlatformTools(db_session=MagicMock())
    tools.code_graph = code_graph
    return tools


# ---------------------------------------------------------------------------
# Service: query-log schema reconciled + non-fatal, results carry path:line
# ---------------------------------------------------------------------------
def test_query_log_insert_uses_real_columns_not_execution_time_ms():
    inserts = re.findall(
        r"INSERT INTO codegraph_query_logs\s*\(([^)]*)\)", _SERVICE_SRC
    )
    assert inserts, "expected a codegraph_query_logs INSERT in the service"
    for cols in inserts:
        assert "duration_ms" in cols, f"INSERT must target duration_ms: {cols!r}"
        assert "workspace_id" in cols, f"INSERT must include workspace_id: {cols!r}"
        assert (
            "execution_time_ms" not in cols
        ), f"phantom execution_time_ms column still present: {cols!r}"


@pytest.mark.asyncio
async def test_search_symbols_survives_query_log_failure():
    db = _LogFailingDB(project_row=SimpleNamespace(id=7), symbol_rows=[_symbol_row()])
    svc = _make_service(db)

    out = await svc.search_symbols("proj", "auth", workspace_id="ws-1")

    assert out["count"] == 1
    assert out["results"][0]["file_path"] == "api/auth.py"
    assert db.log_attempts == 1, "the query-log write should have been attempted"
    assert db.rollbacks == 1, "a failed query-log write must be rolled back, not raised"


@pytest.mark.asyncio
async def test_semantic_search_returns_results_via_pgvector_fallback():
    db = _LogFailingDB(
        project_row=SimpleNamespace(id=7),
        symbol_rows=[_symbol_row(similarity=0.87)],
    )
    svc = _make_service(db)

    out = await svc.semantic_search("proj", "user authentication flow", workspace_id="ws-1")

    assert out["count"] == 1
    r = out["results"][0]
    assert r["file_path"] == "api/auth.py" and r["line_number"] == 42
    assert r["signature"].startswith("def authenticate_user")
    # path:line + signature reach the LLM prompt block
    assert "api/auth.py:42" in out["prompt_block"]
    assert "Signature:" in out["prompt_block"]
    # a query-log failure did not fail the search
    assert db.log_attempts == 1 and db.rollbacks == 1


# ---------------------------------------------------------------------------
# Tool handler: route search_type, forward params, stop dropping short hits
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_search_codebase_routes_semantic_to_semantic_search():
    cg = AsyncMock()
    cg.semantic_search = AsyncMock(
        return_value={
            "results": [
                {
                    "name": "authenticate_user",
                    "symbol_type": "function",
                    "file_path": "api/auth.py",
                    "line_number": 42,
                    "signature": "def authenticate_user(token)",
                    "code_snippet": "def authenticate_user(token):\n    return verify(token)",
                }
            ]
        }
    )
    cg.search_symbols = AsyncMock(return_value={"results": []})
    tools = _make_tools(cg)

    res = await tools.execute_tool(
        "search_codebase",
        {"query": "auth flow", "project_name": "proj", "search_type": "semantic", "limit": 5},
        agent_id=1,
    )

    cg.semantic_search.assert_awaited_once()
    cg.search_symbols.assert_not_awaited()
    assert cg.semantic_search.await_args.kwargs["limit"] >= 5  # requested limit forwarded
    assert res.get("success") is True
    assert res["results"], "semantic results should reach the agent"


@pytest.mark.asyncio
async def test_search_codebase_fuzzy_forwards_symbol_type_and_limit():
    cg = AsyncMock()
    cg.search_symbols = AsyncMock(return_value={"results": []})
    cg.semantic_search = AsyncMock(return_value={"results": []})
    tools = _make_tools(cg)

    await tools.execute_tool(
        "search_codebase",
        {
            "query": "auth",
            "project_name": "proj",
            "search_type": "fuzzy",
            "symbol_type": "function",
            "limit": 3,
        },
        agent_id=1,
    )

    cg.search_symbols.assert_awaited_once()
    kwargs = cg.search_symbols.await_args.kwargs
    assert kwargs["symbol_type"] == "function"  # previously dropped
    assert kwargs["limit"] >= 3
    cg.semantic_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_codebase_symbol_type_all_maps_to_none():
    cg = AsyncMock()
    cg.search_symbols = AsyncMock(return_value={"results": []})
    tools = _make_tools(cg)

    await tools.execute_tool(
        "search_codebase",
        {"query": "auth", "project_name": "proj", "symbol_type": "all"},
        agent_id=1,
    )

    assert cg.search_symbols.await_args.kwargs["symbol_type"] is None


@pytest.mark.asyncio
async def test_search_codebase_keeps_short_snippets():
    """The <50-char filter dropped most hits; short symbols must now survive."""
    cg = AsyncMock()
    cg.search_symbols = AsyncMock(
        return_value={
            "results": [
                {
                    "name": "ping",
                    "symbol_type": "function",
                    "file_path": "a.py",
                    "line_number": 1,
                    "signature": "def ping()",
                    "code_snippet": "def ping(): pass",  # 16 chars < 50
                }
            ]
        }
    )
    tools = _make_tools(cg)

    res = await tools.execute_tool(
        "search_codebase", {"query": "ping", "project_name": "proj"}, agent_id=1
    )

    names = [r.get("symbol_name") for r in res.get("results", [])]
    assert "ping" in names, "short-snippet symbol must not be filtered out"
