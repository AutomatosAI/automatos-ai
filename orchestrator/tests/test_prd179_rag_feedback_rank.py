"""PRD-179 S4 (F070) — rag_feedback as a live ranking feature.

`rag_feedback` used to be a write-only bucket: the endpoint INSERTed thumbs /
ratings that fed nothing back into retrieval (F070 CONFIRMED,
`api/rag_feedback.py:50-70`). This wires the negative signal into
`RAGService.retrieve()` ranking on the live hot path so a document marked
unhelpful de-ranks (lower score, and drops out of the top-K when the field is
tight) on the next retrieval of the same query.

Uses the RANKING path in `modules/rag/service.py` — NOT tool-affinity edges
(`edge_builder.py` is Wave 7's). Pure: the feedback DB read is mocked at the
boundary, so no Postgres.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config as _config  # noqa: E402

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB"):
    if not getattr(_config, _k, None):
        setattr(_config, _k, os.environ[_k])

from modules.rag.service import RAGService  # noqa: E402

WS = "22222222-2222-2222-2222-222222222222"


def _candidate(doc_id: int, score: float, text: str = "chunk") -> Dict[str, Any]:
    return {
        "document_id": doc_id,
        "content": f"{text}-{doc_id}",
        "score": score,
        "similarity": score,
        "metadata": {"document_id": doc_id},
    }


@pytest.fixture
def svc() -> RAGService:
    # __new__ avoids the heavy __init__ (embedders / optimizers); we only test
    # the pure ranking method, which needs no initialised collaborators.
    return RAGService.__new__(RAGService)


def test_marked_unhelpful_doc_is_deranked(svc):
    """Doc 7 is top by similarity but was marked unhelpful → after the feedback
    penalty it must rank BELOW the un-penalised doc 3."""
    candidates = [_candidate(7, 0.90), _candidate(3, 0.70), _candidate(5, 0.65)]

    with patch.object(
        RAGService, "_negative_feedback_doc_ids", return_value={"7"}
    ):
        ranked = svc._apply_feedback_penalty(candidates, workspace_id=WS)

    order = [c["document_id"] for c in ranked]
    assert order.index(3) < order.index(7), (
        f"unhelpful doc 7 was not de-ranked below doc 3 (order={order})"
    )
    # The penalty lowers the score, it doesn't just reorder ties.
    penalised = next(c for c in ranked if c["document_id"] == 7)
    assert penalised["score"] < 0.90


def test_no_feedback_leaves_order_untouched(svc):
    """No negative feedback → identical ranking (feature is inert until used)."""
    candidates = [_candidate(7, 0.90), _candidate(3, 0.70)]
    with patch.object(RAGService, "_negative_feedback_doc_ids", return_value=set()):
        ranked = svc._apply_feedback_penalty(candidates, workspace_id=WS)
    assert [c["document_id"] for c in ranked] == [7, 3]
    assert ranked[0]["score"] == 0.90


def test_penalty_can_drop_doc_from_top_k(svc):
    """With a strong penalty an unhelpful doc that was #1 falls out of top-1 —
    the 'absent from top-K' half of the acceptance criterion."""
    candidates = [_candidate(7, 0.88), _candidate(3, 0.80)]
    with patch.object(RAGService, "_negative_feedback_doc_ids", return_value={"7"}):
        ranked = svc._apply_feedback_penalty(candidates, workspace_id=WS)
    top_1 = [c["document_id"] for c in ranked][:1]
    assert 7 not in top_1, f"unhelpful doc still occupies top-1 (top={top_1})"


def test_feedback_penalty_never_raises(svc):
    """A feedback-read explosion must not break retrieval — degrade to the
    original candidate order."""
    candidates = [_candidate(7, 0.90), _candidate(3, 0.70)]
    with patch.object(
        RAGService, "_negative_feedback_doc_ids", side_effect=RuntimeError("db down")
    ):
        ranked = svc._apply_feedback_penalty(candidates, workspace_id=WS)
    assert [c["document_id"] for c in ranked] == [7, 3]


def test_retrieve_calls_feedback_penalty_on_hot_path():
    """The live retrieve() path must invoke the feedback penalty — proves the
    signal feeds ranking, not just that a helper exists.

    PRD-185 S9 wrapped retrieve() in a thin tracing shim that delegates to
    _retrieve_impl(), so the penalty now lives in the impl. Introspect both so
    the guard proves the penalty is on the live path regardless of the split."""
    import inspect

    src = inspect.getsource(RAGService.retrieve) + inspect.getsource(RAGService._retrieve_impl)
    assert "_apply_feedback_penalty" in src, (
        "retrieve() does not apply rag_feedback to ranking on the live path"
    )


def test_negative_feedback_query_is_workspace_scoped(svc):
    """The feedback read must be workspace-scoped and use CAST(:p AS type) (the
    SQLAlchemy-2.0 bind idiom), never :p::type."""
    import inspect

    src = inspect.getsource(RAGService._negative_feedback_doc_ids)
    assert "workspace_id" in src, "feedback read is not workspace-scoped"
    assert "rag_feedback" in src, "feedback read does not query rag_feedback"
    assert "::" not in src, "raw ::cast used — must be CAST(:p AS type) under SQLAlchemy 2.0"


def test_rag_feedback_to_ranking(svc):
    """W9 acceptance (F070): the signal the POST /rag/feedback endpoint WRITES is
    exactly the signal the ranking READS, and a doc so-marked de-ranks on the
    follow-up retrieval.

    The write side (endpoint INSERT) and the read side (ranking query) must
    agree on the contract: a ``thumbs_down`` carrying ``document_ids``. This
    asserts that contract from both ends, then drives a follow-up retrieval whose
    negative-feedback read returns the marked doc and confirms it de-ranks —
    closing the write-only-bucket gap end to end (DB mocked at the boundary).
    """
    import inspect

    # Write side: the shared writer persists feedback_type + document_ids to
    # rag_feedback, and the endpoint delegates to it. PRD-185 S7 moved the INSERT
    # into modules.rag.feedback_writer so the chat-vote path writes through the
    # same seam — the write contract is unchanged, only its home.
    from api import rag_feedback as rag_feedback_api
    from modules.rag import feedback_writer

    # The INSERT lives in a module-level statement (_INSERT) that the writer
    # references, so inspect the whole module source, not just the function body.
    write_src = inspect.getsource(feedback_writer)
    assert "INSERT INTO rag_feedback" in write_src
    assert "document_ids" in write_src and "feedback_type" in write_src
    assert "write_rag_feedback" in inspect.getsource(rag_feedback_api.submit_feedback), (
        "feedback endpoint no longer delegates to the shared rag_feedback writer"
    )

    # Read side consumes those same columns (thumbs_down / document_ids).
    read_src = inspect.getsource(RAGService._negative_feedback_doc_ids)
    assert "thumbs_down" in read_src and "document_ids" in read_src

    # End-to-end behaviour: doc 9 was thumbed-down last turn; on the follow-up
    # retrieval it de-ranks below the un-flagged doc 4 (was ranked above it).
    candidates = [_candidate(9, 0.92), _candidate(4, 0.71)]
    with patch.object(RAGService, "_negative_feedback_doc_ids", return_value={"9"}):
        ranked = svc._apply_feedback_penalty(candidates, workspace_id=WS)
    order = [c["document_id"] for c in ranked]
    assert order.index(4) < order.index(9), (
        f"doc marked unhelpful via POST /rag/feedback did not de-rank (order={order})"
    )
