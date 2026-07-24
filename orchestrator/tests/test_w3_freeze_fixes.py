"""Freeze-findings fixes — the two measured actions from the 2026-07 live
retrieval baseline (evals/baseline/kg_retrieval_2026-07.json):

1. Query enhancement (HyDE + decomposition + expansion) measured −26.9
   recall@5 points versus plain dense retrieval on pilot-a while paying ~4
   extra LLM calls per query. It now defaults OFF via the canonical config
   accessor: admin/env control kept, an explicit caller value (the eval
   lever grid) still wins, and the documents.py search endpoint no longer
   hardcodes it on.

2. The rerank manager cached one httpx.AsyncClient for the process's
   lifetime; httpx pools bind connections to the loop they open on, so any
   caller on a later loop (thread bridges, per-run asyncio.run) hit "Event
   loop is closed" and the stage silently degraded to identity order — 30
   such failures in the baseline run, on the one lever that measured
   positive (+7.7 recall@5). The client now rebinds per running loop.

All pure — no network, no DB.
"""
import asyncio
import sys
from pathlib import Path

import httpx
import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

import core.llm.manager as llm_manager_mod  # noqa: E402
import core.llm.rerank_manager as rerank_manager_mod  # noqa: E402
import modules.rag.service as rag_service_mod  # noqa: E402
from modules.rag.service import RAGConfig  # noqa: E402


@pytest.fixture
def hermetic_settings(monkeypatch):
    """No DB, no env: the flat rag-settings blob is empty and the canonical
    get_system_setting returns its passed default — so tests read the shipped
    defaults, not this machine's state."""
    monkeypatch.setattr(rag_service_mod, "_load_rag_settings", lambda force=False: {})
    monkeypatch.setattr(
        llm_manager_mod, "get_system_setting", lambda group, key, default=None: default
    )
    monkeypatch.delenv("RAG_QUERY_ENHANCEMENT_ENABLED", raising=False)
    monkeypatch.delenv("RAG_RERANK_ENABLED", raising=False)


# ---------------------------------------------------------------------------
# The enhancement default flip (ON → OFF)
# ---------------------------------------------------------------------------

def test_enhancement_off_by_default(hermetic_settings):
    assert RAGConfig().enable_query_enhancement is False


def test_explicit_caller_value_wins(hermetic_settings):
    assert RAGConfig(enable_query_enhancement=True).enable_query_enhancement is True
    assert RAGConfig(enable_query_enhancement=False).enable_query_enhancement is False


def test_admin_setting_on_is_respected(hermetic_settings, monkeypatch):
    def _setting(group, key, default=None):
        if (group, key) == ("rag", "query_enhancement_enabled"):
            return "true"
        return default

    monkeypatch.setattr(llm_manager_mod, "get_system_setting", _setting)
    assert RAGConfig().enable_query_enhancement is True


def test_search_endpoint_does_not_force_enhancement_on():
    documents_src = (Path(_orchestrator_root) / "api" / "documents.py").read_text()
    assert "enable_query_enhancement=True" not in documents_src


def test_eval_lever_grid_pins_enhancement_explicitly():
    """The freeze instrument sets every variant's enhancement lever explicitly
    (never the default), so the default flip cannot change what it measures."""
    src = (Path(_orchestrator_root) / "evals" / "retrieval_recall.py").read_text()
    assert "enable_query_enhancement=True" in src
    assert "enable_query_enhancement=False" in src


# ---------------------------------------------------------------------------
# The rerank client loop binding
# ---------------------------------------------------------------------------

def test_rerank_client_stable_within_a_loop():
    mgr = rerank_manager_mod.RerankManager()

    async def grab_twice():
        return mgr._get_client(), mgr._get_client()

    a, b = asyncio.run(grab_twice())
    assert a is b


def test_rerank_client_rebinds_across_loops():
    """A client cached on a finished loop must never be handed to the next
    loop — that is exactly the 'Event loop is closed' degradation."""
    mgr = rerank_manager_mod.RerankManager()

    async def grab():
        return mgr._get_client()

    first = asyncio.run(grab())
    second = asyncio.run(grab())
    assert first is not second


def test_rerank_completes_after_prior_loop_closed(monkeypatch):
    """End-to-end rerank on two successive loops returns real Cohere scores
    both times — not the zero-score identity fallback of the swallowed path."""
    mgr = rerank_manager_mod.RerankManager()
    mgr._api_key = "test-key"
    mgr._loaded = True

    def handler(request):
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            },
        )

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        rerank_manager_mod.httpx,
        "AsyncClient",
        lambda **kw: real_async_client(transport=httpx.MockTransport(handler), **kw),
    )

    out1 = asyncio.run(mgr.rerank("q", ["a", "b"], top_n=2))
    out2 = asyncio.run(mgr.rerank("q", ["a", "b"], top_n=2))
    for out in (out1, out2):
        assert [r.index for r in out] == [1, 0]
        assert out[0].relevance_score == 0.9
