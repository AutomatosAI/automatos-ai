"""
RAG Retrieval Quality Scorer
==============================

Scores the quality of retrieved chunks after each RAG query.
Used for trending analysis and feedback loops.
"""

import logging
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


def score_retrieval_quality(
    query: str,
    chunks: List[Dict],
    response: str = None,
) -> Dict:
    """
    Score the quality of retrieved chunks using heuristics.

    Returns:
        {
            "avg_similarity": float,
            "source_diversity": float,   # 0-1, higher = more diverse sources
            "coverage_score": float,      # Do chunks cover the query's concepts?
            "freshness_score": float,     # How recent are the source documents?
            "chunk_count": int,
            "total_content_length": int,
        }
    """
    if not chunks:
        return {
            "avg_similarity": 0.0,
            "source_diversity": 0.0,
            "coverage_score": 0.0,
            "freshness_score": 0.0,
            "chunk_count": 0,
            "total_content_length": 0,
        }

    # Average similarity
    similarities = [c.get("similarity", 0.0) for c in chunks]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

    # Source diversity: ratio of unique documents to total chunks
    unique_docs = set()
    for c in chunks:
        doc_id = c.get("document_id") or c.get("source_file", "")
        if doc_id:
            unique_docs.add(doc_id)
    source_diversity = len(unique_docs) / len(chunks) if chunks else 0.0

    # Coverage: check if query terms appear in retrieved content
    query_terms = set(query.lower().split())
    # Remove stop words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                  "to", "for", "of", "with", "and", "or", "not", "it", "this", "that",
                  "how", "what", "when", "where", "why", "who", "which", "do", "does"}
    query_terms -= stop_words

    if query_terms:
        all_content = " ".join(c.get("content", "") for c in chunks).lower()
        covered = sum(1 for t in query_terms if t in all_content)
        coverage_score = covered / len(query_terms)
    else:
        coverage_score = 1.0

    # Freshness: based on metadata timestamps (if available)
    freshness_score = 0.5  # Default neutral
    now = datetime.utcnow()
    dates = []
    for c in chunks:
        meta = c.get("metadata", {})
        if isinstance(meta, dict):
            upload_date = meta.get("upload_date") or meta.get("created_at")
            if upload_date:
                try:
                    if isinstance(upload_date, str):
                        dt = datetime.fromisoformat(upload_date.replace("Z", "+00:00"))
                    else:
                        dt = upload_date
                    days_old = (now - dt.replace(tzinfo=None)).days
                    dates.append(days_old)
                except (ValueError, TypeError):
                    pass

    if dates:
        avg_days = sum(dates) / len(dates)
        # Score: 1.0 for today, 0.5 for 30 days, 0.1 for 365+ days
        freshness_score = max(0.1, 1.0 - (avg_days / 365.0))

    total_content = sum(len(c.get("content", "")) for c in chunks)

    return {
        "avg_similarity": round(avg_similarity, 4),
        "source_diversity": round(source_diversity, 4),
        "coverage_score": round(coverage_score, 4),
        "freshness_score": round(freshness_score, 4),
        "chunk_count": len(chunks),
        "total_content_length": total_content,
    }
