"""Multi-query RAG retrieval runs its query variations CONCURRENTLY
(``asyncio.gather``) instead of a serial ``await`` loop.

The serial loop made retrieval latency the SUM of all variations — the dominant
cost in the ~80-113s context-prep times observed in prod (2026-07-09). RRF
fusion must stay byte-identical: a sum of ``1/(k+rank)`` per doc, order-
independent, results consumed in query order.

Pure unit test — ``_get_candidates`` is mocked; no DB / network / embeddings.
"""
import asyncio
from types import SimpleNamespace

import pytest


def _svc():
    try:
        from modules.rag.service import RAGService
    except Exception as e:  # env without the heavy service deps
        pytest.skip(f"rag.service not importable in this env: {e}")
    svc = RAGService.__new__(RAGService)  # bypass heavy __init__
    svc.config = SimpleNamespace(rrf_k=60)
    return svc


def test_all_variations_queried_and_fused():
    svc = _svc()
    seen = []

    async def fake_candidates(q, **kw):
        seen.append(q)
        # one doc unique to this query + one shared across all queries
        return [{"id": f"only-{q}", "content": q}, {"id": "shared", "content": "s"}]

    svc._get_candidates = fake_candidates

    queries = ["q1", "q2", "q3"]
    out = asyncio.run(svc._multi_query_retrieval_with_rrf(queries, workspace_id="ws"))

    # every variation was queried (not just the first)
    assert set(seen) == set(queries)
    assert {d["id"] for d in out} == {"only-q1", "only-q2", "only-q3", "shared"}
    # the doc present in all 3 variations fuses to the top with query_count=3
    shared = next(d for d in out if d["id"] == "shared")
    assert shared["query_count"] == 3
    assert out[0]["id"] == "shared"


def test_one_failing_variation_does_not_sink_the_rest():
    svc = _svc()

    async def fake_candidates(q, **kw):
        if q == "boom":
            raise RuntimeError("provider down")
        return [{"id": f"doc-{q}", "content": q}]

    svc._get_candidates = fake_candidates

    out = asyncio.run(
        svc._multi_query_retrieval_with_rrf(["good1", "boom", "good2"], workspace_id="ws")
    )
    # the two healthy variations still fuse; the failing one is skipped, not fatal
    assert {d["id"] for d in out} == {"doc-good1", "doc-good2"}
