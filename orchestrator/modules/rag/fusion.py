"""Reciprocal-rank fusion — the ONE fusion path for retrieval (PRD-188 S3).

Extracted from ``RAGService._multi_query_retrieval_with_rrf`` so the
multi-query variant fusion and the dense+sparse hybrid fusion are the same
math with the same dedup identity — not two drifting copies. Pure and
dependency-free (stdlib only) so the eval harness's equivalence test can pin
its own restatement to this one.
"""
from typing import Dict, List, Optional, Sequence

DEFAULT_RRF_K = 60


def fusion_key(doc: Dict) -> str:
    """Leg-agnostic identity for a candidate chunk.

    The dense leg ids chunks by S3 vector key; the sparse BM25 leg by
    ``document_chunks`` pk — the same chunk arrives with two different
    ``id``s and must FUSE, not duplicate. ``(document_id, chunk_index)``
    identifies a chunk in both legs; the fallback is the id / content-prefix
    idiom the old in-method fusion used.
    """
    doc_id = doc.get("document_id")
    meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    chunk_index = meta.get("chunk_index")
    if doc_id not in (None, "", 0) and chunk_index is not None:
        return f"doc:{doc_id}:chunk:{chunk_index}"
    return str(doc.get("id") or "") or doc.get("content", "")[:100]


def reciprocal_rank_fuse(
    ranked_lists: Sequence[List[Dict]],
    k: int = DEFAULT_RRF_K,
    weights: Optional[Sequence[float]] = None,
) -> List[Dict]:
    """Fuse N best-first candidate lists: score(c) = Σ_lists weight_i / (k + rank_i).

    Returns NEW dicts (inputs are never mutated): the payload of the first
    list that surfaced the chunk, plus ``rrf_score`` and ``query_count`` (how
    many lists surfaced it), sorted best-first. Equal weights (``None``) is
    exactly the multi-query RRF this was extracted from; the hybrid caller
    passes the real ``hybrid_vector_weight`` / ``hybrid_keyword_weight``.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights ({len(weights)}) must match ranked_lists ({len(ranked_lists)})"
        )

    fused: Dict[str, Dict] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(ranked):
            key = fusion_key(doc)
            entry = fused.get(key)
            if entry is None:
                entry = {"doc": doc, "score": 0.0, "appearances": 0}
                fused[key] = entry
            entry["score"] += weight / (k + rank)
            entry["appearances"] += 1

    scored = []
    for entry in fused.values():
        doc = entry["doc"].copy()
        doc["rrf_score"] = entry["score"]
        doc["query_count"] = entry["appearances"]
        scored.append(doc)
    # Stable sort: ties keep first-seen order, deterministic run to run.
    scored.sort(key=lambda d: d["rrf_score"], reverse=True)
    return scored
