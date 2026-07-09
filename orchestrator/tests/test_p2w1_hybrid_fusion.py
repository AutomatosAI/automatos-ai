"""PRD-188 S3: the real BM25 sparse leg + the one shared RRF fuser.

"Hybrid search" was three dataclass fields with no sparse leg behind them —
RRF fused vector variants against themselves while the tsvector index paid
maintenance on every insert with zero readers. These tests pin the fix:

* the fusion math is a pure function (hand-computed table), extracted intact
  from the old in-method multi-query RRF (regression-pinned);
* the hybrid weights are REAL — flipping them flips the fused order;
* the same chunk arriving from both legs (different id schemes) FUSES via
  (document_id, chunk_index) instead of duplicating;
* the BM25 leg fails closed without a workspace and returns candidates in the
  dense leg's exact shape (DB boundary mocked — no Postgres in these tests);
* a sparse-leg failure degrades to dense-only (loud, never fatal), and a
  dense-leg miss can be rescued by a lexical hit — the audit's exact-term
  failure mode;
* the eval harness's self-contained RRF restatement matches this fuser on the
  same inputs (the equivalence pin the eval's docstring promises).
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

import core.llm.manager as llm_manager_mod  # noqa: E402
import modules.rag.bm25_leg as bm25_leg_mod  # noqa: E402
import modules.rag.service as rag_service_mod  # noqa: E402
from modules.rag.bm25_leg import bm25_search  # noqa: E402
from modules.rag.fusion import fusion_key, reciprocal_rank_fuse  # noqa: E402
from modules.rag.retrieval_filters import RetrievalFilters  # noqa: E402
from modules.rag.service import RAGConfig, RAGService  # noqa: E402


@pytest.fixture
def hermetic_settings(monkeypatch):
    monkeypatch.setattr(rag_service_mod, "_load_rag_settings", lambda force=False: {})
    monkeypatch.setattr(
        llm_manager_mod, "get_system_setting", lambda group, key, default=None: default
    )
    monkeypatch.delenv("RAG_RERANK_ENABLED", raising=False)
    monkeypatch.delenv("RAG_HYBRID_ENABLED", raising=False)


def _c(id_, content="", doc_id=None, chunk_index=None, **extra):
    d = {"id": id_, "content": content or id_}
    if doc_id is not None:
        d["document_id"] = doc_id
    if chunk_index is not None:
        d["metadata"] = {"chunk_index": chunk_index}
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
# The pure fuser
# ---------------------------------------------------------------------------

def test_reciprocal_rank_fuse_hand_computed():
    """Equal weights = the exact math extracted from the old in-method RRF:
    a: 1/60; b: 1/61 + 1/60; c: 1/61 (k=60)."""
    fused = reciprocal_rank_fuse([[_c("a"), _c("b")], [_c("b"), _c("c")]], k=60)
    assert [d["id"] for d in fused] == ["b", "a", "c"]
    by_id = {d["id"]: d for d in fused}
    assert by_id["b"]["rrf_score"] == pytest.approx(1 / 61 + 1 / 60)
    assert by_id["a"]["rrf_score"] == pytest.approx(1 / 60)
    assert by_id["c"]["rrf_score"] == pytest.approx(1 / 61)
    assert by_id["b"]["query_count"] == 2
    assert by_id["a"]["query_count"] == 1


def test_fuse_weights_are_real_not_decorative():
    lists = [[_c("x"), _c("y"), _c("z")], [_c("y"), _c("x")]]
    heavier_first_leg = reciprocal_rank_fuse(lists, k=60, weights=[0.7, 0.3])
    assert [d["id"] for d in heavier_first_leg][:2] == ["x", "y"]
    heavier_second_leg = reciprocal_rank_fuse(lists, k=60, weights=[0.3, 0.7])
    assert [d["id"] for d in heavier_second_leg][0] == "y"


def test_fuse_weights_length_mismatch_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fuse([[_c("a")]], weights=[0.5, 0.5])


def test_fusion_key_dedups_same_chunk_across_legs():
    """Dense ids by S3 key, sparse by pg pk — same (document_id, chunk_index)
    must fuse into ONE candidate with the summed score."""
    dense = _c("s3-vec-abc123", content="the chunk", doc_id=7, chunk_index=2, similarity=0.81)
    sparse = _c("pg-chunk-9911", content="the chunk", doc_id=7, chunk_index=2, similarity=1.0)
    assert fusion_key(dense) == fusion_key(sparse)

    fused = reciprocal_rank_fuse([[dense], [sparse]], k=60, weights=[0.7, 0.3])
    assert len(fused) == 1
    assert fused[0]["rrf_score"] == pytest.approx(0.7 / 60 + 0.3 / 60)
    assert fused[0]["query_count"] == 2
    # First-seen payload wins: the dense leg's real cosine similarity survives.
    assert fused[0]["similarity"] == 0.81


def test_fusion_key_falls_back_to_id_then_content():
    assert fusion_key({"id": "abc", "content": "text"}) == "abc"
    assert fusion_key({"id": "", "content": "text body"}) == "text body"


def test_fuse_never_mutates_inputs():
    a = _c("a", doc_id=1, chunk_index=0)
    before = dict(a)
    reciprocal_rank_fuse([[a]], k=60)
    assert a == before  # rrf_score/query_count land on copies only


# ---------------------------------------------------------------------------
# Multi-query fusion behaviour unchanged by the extraction
# ---------------------------------------------------------------------------

def test_multi_query_rrf_behaviour_unchanged(hermetic_settings, monkeypatch):
    service = RAGService(RAGConfig(hybrid_search_enabled=False, enable_reranking=False))
    per_query = {
        "q1": [_c("a"), _c("b")],
        "q2": [_c("b"), _c("c")],
    }

    async def fake_get_candidates(q, limit=20, min_similarity=0.5, workspace_id=None):
        return per_query[q]

    monkeypatch.setattr(service, "_get_candidates", fake_get_candidates)
    fused = asyncio.run(
        service._multi_query_retrieval_with_rrf(["q1", "q2"], workspace_id="ws")
    )
    assert [d["id"] for d in fused] == ["b", "a", "c"]
    assert fused[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 60)
    assert fused[0]["query_count"] == 2


# ---------------------------------------------------------------------------
# The hybrid fusion seam in RAGService
# ---------------------------------------------------------------------------

def _service(**cfg):
    return RAGService(RAGConfig(enable_reranking=False, **cfg))


def test_hybrid_fuses_dense_and_sparse(hermetic_settings, monkeypatch):
    dense = [_c("s3-1", doc_id=1, chunk_index=0), _c("s3-2", doc_id=2, chunk_index=0)]
    sparse = [_c("pg-chunk-9", doc_id=3, chunk_index=1), _c("pg-chunk-4", doc_id=1, chunk_index=0)]
    monkeypatch.setattr(bm25_leg_mod, "bm25_search", AsyncMock(return_value=sparse))

    service = _service(hybrid_search_enabled=True)
    fused = asyncio.run(
        service._fuse_with_sparse_leg(
            "q", dense, RetrievalFilters(workspace_id="ws-1"), limit=10
        )
    )
    keys = {fusion_key(d) for d in fused}
    # doc1:chunk0 fused across legs; doc2 + doc3 present once each.
    assert len(fused) == 3
    assert keys == {"doc:1:chunk:0", "doc:2:chunk:0", "doc:3:chunk:1"}
    by_key = {fusion_key(d): d for d in fused}
    assert by_key["doc:1:chunk:0"]["query_count"] == 2


def test_hybrid_dense_miss_rescued_by_sparse(hermetic_settings, monkeypatch):
    """The audit's exact-term case: embeddings miss, the lexical leg must
    still ground the turn (fusion runs BEFORE the empty-check)."""
    sparse = [_c("pg-chunk-7", doc_id=9, chunk_index=0)]
    monkeypatch.setattr(bm25_leg_mod, "bm25_search", AsyncMock(return_value=sparse))
    fused = asyncio.run(
        _service()._fuse_with_sparse_leg("ERR-SYNC-409", [], RetrievalFilters(workspace_id="w"), limit=5)
    )
    assert len(fused) == 1 and fusion_key(fused[0]) == "doc:9:chunk:0"


def test_hybrid_sparse_failure_degrades_to_dense(hermetic_settings, monkeypatch):
    dense = [_c("s3-1", doc_id=1, chunk_index=0)]
    monkeypatch.setattr(
        bm25_leg_mod, "bm25_search", AsyncMock(side_effect=RuntimeError("pg down"))
    )
    out = asyncio.run(
        _service()._fuse_with_sparse_leg("q", dense, RetrievalFilters(workspace_id="w"), limit=5)
    )
    assert out == dense  # loud warning, never fatal


def test_hybrid_no_sparse_hits_is_dense_passthrough(hermetic_settings, monkeypatch):
    dense = [_c("s3-1", doc_id=1, chunk_index=0)]
    monkeypatch.setattr(bm25_leg_mod, "bm25_search", AsyncMock(return_value=[]))
    out = asyncio.run(
        _service()._fuse_with_sparse_leg("q", dense, RetrievalFilters(workspace_id="w"), limit=5)
    )
    assert out == dense  # no rrf_score decoration when there is nothing to fuse


def test_hybrid_flag_resolution(hermetic_settings):
    assert RAGConfig(hybrid_search_enabled=False).hybrid_search_enabled is False
    # None resolves from the canonical accessor default (ON).
    assert RAGConfig().hybrid_search_enabled is True


# ---------------------------------------------------------------------------
# The BM25 leg (DB boundary mocked)
# ---------------------------------------------------------------------------

def _rows(*rows):
    return list(rows)


def test_bm25_leg_fails_closed_without_workspace(monkeypatch):
    fetch = AsyncMock()
    monkeypatch.setattr(bm25_leg_mod, "_fetch_rows", fetch)
    out = asyncio.run(bm25_search("query", RetrievalFilters(workspace_id=""), limit=5))
    assert out == []
    fetch.assert_not_called()  # fail-closed BEFORE any database work

    out_none = asyncio.run(bm25_search("query", None, limit=5))
    assert out_none == []
    fetch.assert_not_called()


def test_bm25_leg_ignores_empty_query(monkeypatch):
    fetch = AsyncMock()
    monkeypatch.setattr(bm25_leg_mod, "_fetch_rows", fetch)
    assert asyncio.run(bm25_search("   ", RetrievalFilters(workspace_id="w"), limit=5)) == []
    fetch.assert_not_called()


def test_bm25_maps_rows_to_dense_candidate_shape(monkeypatch):
    rows = _rows(
        {
            "id": 11,
            "document_id": 7,
            "chunk_index": 2,
            "content": "ERR-SYNC-409 means a stale item version.",
            "metadata": '{"page_start": 3}',
            "parent_content": None,
            "headers": "{}",
            "source_file": "sync-errors.pdf",
            "score": 0.62,
        },
        {
            "id": 12,
            "document_id": 8,
            "chunk_index": 0,
            "content": "Retry policy for sync failures.",
            "metadata": None,
            "parent_content": "parent text",
            "headers": None,
            "source_file": "runbook.md",
            "score": 0.31,
        },
    )
    monkeypatch.setattr(bm25_leg_mod, "_fetch_rows", AsyncMock(return_value=rows))

    out = asyncio.run(bm25_search("ERR-SYNC-409", RetrievalFilters(workspace_id="w"), limit=5))
    assert len(out) == 2
    top = out[0]
    # The dense leg's exact shape — the fuser and every downstream stage
    # (team filter, rerank, feedback penalty, parent expansion) are leg-agnostic.
    assert set(top) == {
        "id", "content", "source_file", "document_id", "file_type",
        "similarity", "metadata", "parent_content", "headers",
    }
    assert top["id"] == "pg-chunk-11"
    assert top["document_id"] == 7
    assert top["file_type"] == "pdf"
    assert top["metadata"]["chunk_index"] == 2  # parent expansion needs this
    assert top["metadata"]["retrieval_leg"] == "bm25"
    assert top["metadata"]["page_start"] == 3  # jsonb-as-string parsed
    # Max-normalised ts_rank: top hit 1.0, second relative.
    assert top["similarity"] == pytest.approx(1.0)
    assert out[1]["similarity"] == pytest.approx(0.31 / 0.62)


# ---------------------------------------------------------------------------
# Equivalence pin: the eval's self-contained RRF matches the shipped fuser
# ---------------------------------------------------------------------------

def test_eval_rrf_restatement_matches_shipped_fuser():
    """evals/retrieval_recall re-states weighted RRF locally (its CI lane
    installs no deps and modules.rag imports tiktoken at package import).
    This pin keeps the two implementations the same math — if one changes,
    change both, on purpose."""
    from evals.retrieval_recall import _weighted_rrf

    id_lists = [["a", "b", "c"], ["b", "a", "d"]]
    weights = [0.7, 0.3]
    eval_order = _weighted_rrf(id_lists, weights, k=60)

    dict_lists = [[_c(i) for i in lst] for lst in id_lists]
    fused = reciprocal_rank_fuse(dict_lists, k=60, weights=weights)
    assert [d["id"] for d in fused] == eval_order
