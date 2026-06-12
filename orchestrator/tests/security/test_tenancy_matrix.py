"""PRD-156 S1 — multimodal knowledge tools are workspace/tenant-scoped.

The four multimodal search tools (``search_tables/images/formulas/multimodal``,
``modules/rag/services/multimodal_knowledge_tools.py``) previously queried
``knowledge_items`` with NO workspace clause — a confirmed cross-tenant leak
exposed to every agent via the unified executor. These tests pin the fix:

  * every tool's executed SQL carries the mandatory ``ki.workspace_id`` filter
    and binds the workspace_id parameter (parametrized across all four);
  * a missing ``workspace_id`` FAILS CLOSED — no DB query is issued, never an
    unscoped scan;
  * when a team is supplied, the JSONB ``metadata->'team_access'`` clause is in
    the query and the normalized ``:team`` is bound;
  * ``_embed_query`` fails closed (returns ``None``) on any embedding error, so
    a novel query can never fall through to an unranked / unscoped scan.

Pure unit tests: the DB session is a mock that captures the SQL + params handed
to ``execute`` — no live Postgres needed, deterministic, and it proves the
*actual* query each tool sends is scoped (stronger than testing the clause
builder in isolation). The end-to-end DB matrix runs in CI.
"""
from __future__ import annotations

import importlib.util as _ilu
import sys as _sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Lean-venv shim: importing the multimodal tools pulls modules.rag's ingestion
# chain, whose PDF processor does ``import camelot`` at module top. camelot is a
# CI dep (requirements.txt) but often absent from a lean local venv, where it
# breaks collection. Stub only the missing *leaf* (never the modules.rag
# package) — the blessed pattern used across the security suite.
def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False  # spec-less camelot stub already present → chain satisfied


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    import types as _types

    _sys.modules.setdefault("camelot", _types.ModuleType("camelot"))

from modules.rag.services.multimodal_knowledge_tools import MultimodalKnowledgeTools

pytestmark = pytest.mark.asyncio

# All four multimodal search surfaces. Every one must scope by workspace.
TOOLS = ["search_tables", "search_images", "search_formulas", "search_multimodal"]


def _capturing_tools():
    """A MultimodalKnowledgeTools whose db.execute records (sql, params) and
    returns no rows; _embed_query is stubbed so the query path runs without a
    real embedding manager."""
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = []
    tools = MultimodalKnowledgeTools(db)
    tools._embed_query = AsyncMock(return_value="[0.1,0.2,0.3]")
    return tools, db


def _last_execute(db):
    assert db.execute.called, "tool issued no DB query"
    args = db.execute.call_args.args
    return str(args[0]), args[1]  # (sql_text, params)


@pytest.mark.parametrize("tool_name", TOOLS)
async def test_tool_query_is_workspace_scoped(tool_name):
    """Workspace A's query must carry the workspace_id filter + bound param."""
    tools, db = _capturing_tools()
    result = await getattr(tools, tool_name)("find anything", workspace_id="ws-A")
    sql, params = _last_execute(db)
    assert "ki.workspace_id = :workspace_id" in sql, (
        f"{tool_name} sends an UNSCOPED query — cross-tenant leak"
    )
    assert params.get("workspace_id") == "ws-A"
    assert result["success"] is True


@pytest.mark.parametrize("tool_name", TOOLS)
async def test_tool_fails_closed_without_workspace_id(tool_name):
    """No workspace_id → fail closed, and NEVER issue an (unscoped) query."""
    tools, db = _capturing_tools()
    result = await getattr(tools, tool_name)("find anything")  # no workspace_id
    assert result["success"] is False
    db.execute.assert_not_called()


@pytest.mark.parametrize("tool_name", TOOLS)
async def test_team_filter_applied_when_team_supplied(tool_name):
    """A team scope adds the JSONB metadata team clause + normalized :team."""
    tools, db = _capturing_tools()
    await getattr(tools, tool_name)("q", workspace_id="ws-A", team="  Support ")
    sql, params = _last_execute(db)
    assert "team_access" in sql and ":team" in sql
    assert params.get("team") == "support"  # normalized (stripped + lowercased)


async def test_embed_query_fails_closed_on_error():
    """Embedding failure returns None so callers never run an unranked scan."""
    tools = MultimodalKnowledgeTools(MagicMock())
    with patch(
        "core.llm.embedding_manager.create_embedding_manager",
        side_effect=RuntimeError("embedding backend down"),
    ):
        assert await tools._embed_query("novel query") is None


async def test_no_workspace_means_no_execute_even_with_team():
    """Team without workspace_id still fails closed (workspace is mandatory)."""
    tools, db = _capturing_tools()
    result = await tools.search_tables("q", team="Support")  # team but no workspace
    assert result["success"] is False
    db.execute.assert_not_called()


async def test_similarity_is_embedding_ranked_not_exact_match():
    """The broken ``WHERE content = :query`` exact-match is replaced by real
    vector ranking — the query orders by embedding distance and binds the
    embedded query vector, so a novel query returns ranked (not NULL-ordered)
    results."""
    tools, db = _capturing_tools()
    await tools.search_tables("a query never seen before", workspace_id="ws-A")
    sql, params = _last_execute(db)
    assert "ki.embedding <=> :query_embedding" in sql, "not embedding-ranked"
    assert "as similarity" in sql
    assert params.get("query_embedding") == "[0.1,0.2,0.3]"
    assert "content = :query" not in sql, "old exact-match subquery still present"


async def test_upload_team_access_parse_and_normalize():
    """Upload persistence logic: the comma-separated team_access form field is
    parsed and normalized (stripped + lowercased, blanks dropped) before being
    written into each knowledge_item's metadata (end-to-end store/retrieve is
    the CI integration test; this pins the parse/normalize the endpoint uses)."""
    from core.team_access import normalize_teams

    assert normalize_teams("Support, sales,  ,Eng ".split(",")) == [
        "support",
        "sales",
        "eng",
    ]
    assert normalize_teams("".split(",")) == []  # empty field → visible to all
