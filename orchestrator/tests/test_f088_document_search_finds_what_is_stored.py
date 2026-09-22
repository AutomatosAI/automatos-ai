"""F088 (night 3) — the document search finds what is stored.

Two ways it did not:
* Auto called ``search_documents`` four times and got "Unknown tool" each time
  (the registry check ran before the alias could), then told the owner the
  product's document search was broken. The name now runs
  ``platform_search_documents``; any other unknown name says which real tools
  are nearest, so the model retries instead of giving up.
* ``POST /api/documents/search`` ran its own vector-only pass with a 0.70
  cosine floor: exact words stored in three documents each ("The Salt Loft",
  "moreish", "Declan Frost") returned nothing. It now takes its hits from the
  one retrieval funnel the chat and the agents use.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from modules.tools.execution import unified_executor as ue


@pytest.fixture
def executor(monkeypatch):
    for name in ("fire_telemetry", "fire_tool_gap", "capture_tool_outcome", "fire_tool_trace", "rollback_if_aborted"):
        monkeypatch.setattr(ue, name, lambda *a, **k: None)
    ex = ue.UnifiedToolExecutor.__new__(ue.UnifiedToolExecutor)
    ex.composio_actions, ex.tool_routes = {}, {"search_knowledge": None, "search_codebase": None}
    ex._policy_gate_check = lambda *a, **k: None
    ex._tool_registry = NS(get_tool=lambda _name: None,
                          get_all_tools=lambda: [NS(name="search_knowledge"), NS(name="search_codebase")])
    calls = []

    async def platform_action(name, params, **kw):
        calls.append((name, params))
        return {"success": True, "results": []}

    ex._execute_platform_action = platform_action
    ex.calls = calls
    return ex


def test_the_name_auto_reached_for_runs_the_document_search(executor):
    out = asyncio.run(executor.execute_tool("search_documents", {"query": "Callum"}, agent_id=1))
    assert out["success"] is True
    assert executor.calls == [("platform_search_documents", {"query": "Callum"})]


def test_an_unknown_name_says_which_real_tools_are_nearest(executor):
    out = asyncio.run(executor.execute_tool("serch_knowledge", {"query": "x"}, agent_id=1))
    assert out["success"] is False and out["error"].startswith("Unknown tool: serch_knowledge. Nearest real tools:")
    assert "search_knowledge" in out["error"]
    far = asyncio.run(executor.execute_tool("zzqq", {}, agent_id=1))
    assert far["error"] == "Unknown tool: zzqq — there is no tool by that name; use one from your tool list."


# ── the search route ────────────────────────────────────────────────────────

def _funnel(monkeypatch, chunks, sources_map):
    seen = {}

    async def retrieve(self, **kw):
        seen.update(kw)
        return NS(chunks=chunks, sources_map=sources_map)

    monkeypatch.setattr("modules.rag.service.RAGService.__init__", lambda self, *a, **k: None)
    monkeypatch.setattr("modules.rag.service.RAGService.retrieve", retrieve)
    return seen


def test_the_route_takes_its_hits_from_the_retrieval_funnel(monkeypatch):
    from api.documents import _retrieval_hits

    chunks = [
        {"document_id": 724, "source_file": "q2-2026-numbers.csv", "content": "| The Salt Loft | 12 kg |", "chunk_index": 3},
        {"metadata": {"document_id": 885, "file_name": "wholesale-accounts.md", "chunk_index": 0},
         "content": "The Salt Loft pays on 30 days.", "similarity": 0.41},
    ]
    seen = _funnel(monkeypatch, chunks, [{"document_id": 724, "score": 0.52}])
    hits = asyncio.run(_retrieval_hits("The Salt Loft", 30, "ws-1", None))
    assert [(h["metadata"]["document_id"], h["file_name"], h["score"]) for h in hits] == [
        (724, "q2-2026-numbers.csv", 0.52), (885, "wholesale-accounts.md", 0.41)]
    assert seen["workspace_id"] == "ws-1" and seen["max_chunks"] == 30
    # a floor only when the caller asks for one — the old 0.70 default returned nothing here
    assert [h["metadata"]["document_id"] for h in asyncio.run(
        _retrieval_hits("The Salt Loft", 30, "ws-1", None, floor=0.5))] == [724]


def test_the_route_no_longer_carries_its_own_vector_search():
    import inspect

    import api.documents as documents

    source = inspect.getsource(documents.semantic_search)
    assert "_retrieval_hits(" in source
    assert "get_vector_store" not in source and "create_embedding_manager" not in source
