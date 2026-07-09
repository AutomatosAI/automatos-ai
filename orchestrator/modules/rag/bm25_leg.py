"""The real BM25 sparse leg (PRD-188 S3) — the reader the index never had.

``document_chunks.search_vector`` (tsvector + GIN + maintenance trigger on
every insert/update, migration 20260218) has been paid for on every chunk
write since RAG v3 — and until this module, nothing live read it: "hybrid
search" fused vector variants against themselves. This leg turns that dead
cost into the lexical half of real dense+sparse hybrid: exact-term queries
(SKUs, error codes, names — where pure-vector fails hardest) rank by Postgres
``ts_rank`` and fuse into the same RRF as the dense candidates.

Scope: workspace-scoped through the resolved :class:`RetrievalFilters` from
``build_retrieval_filters`` — the PRD-157 fail-closed choke point. No
workspace ⇒ ``[]`` without touching the database. Team restrictions are NOT
re-implemented here: sparse candidates carry ``document_id`` and flow through
the same downstream ``_filter_by_team`` the dense candidates do — one policy,
one place.

Do NOT resurrect ``EnhancedVectorStore._hybrid_search`` — that dead reader is
the vector-substrate kill-list's to delete; this is the clean, small one.
"""
import json
import logging
from typing import Any, Dict, List

from modules.rag.retrieval_filters import RetrievalFilters

logger = logging.getLogger(__name__)

_BM25_SQL = """
    SELECT dc.id,
           dc.document_id,
           dc.chunk_index,
           dc.content,
           dc.metadata,
           dc.parent_content,
           dc.headers,
           d.filename AS source_file,
           ts_rank(dc.search_vector, plainto_tsquery('english', $1)) AS score
    FROM document_chunks dc
    JOIN documents d ON d.id = dc.document_id
    WHERE d.workspace_id = $2::uuid
      AND dc.search_vector @@ plainto_tsquery('english', $1)
    ORDER BY score DESC, dc.id
    LIMIT $3
"""


async def _fetch_rows(query: str, workspace_id: str, limit: int) -> List[Any]:
    """One asyncpg round-trip (same idiom as the parent-context hydration in
    ``RAGService._expand_to_parent_context``)."""
    import asyncpg
    from config import config as app_config

    conn = await asyncpg.connect(app_config.DATABASE_URL)
    try:
        return await conn.fetch(_BM25_SQL, query, workspace_id, limit)
    finally:
        await conn.close()


def _as_dict(value: Any, default: Dict) -> Dict:
    """asyncpg returns JSONB as str without a codec — parse defensively."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else default
        except (ValueError, TypeError):
            return default
    return default


async def bm25_search(query: str, filters: RetrievalFilters, limit: int = 20) -> List[Dict]:
    """Workspace-scoped lexical candidates in the dense leg's exact shape.

    Fail-closed: an empty query or a scope without a workspace returns ``[]``
    before any database work. ``similarity`` is the max-normalised ``ts_rank``
    of the result set (relative lexical strength on the [0,1] scale downstream
    scoring expects); the RRF fusion is rank-based, so the normalisation never
    changes fusion order — it only keeps the metadata honest.
    """
    if not query or not query.strip():
        return []
    if filters is None or not filters.workspace_id:
        logger.warning("bm25_search without workspace scope — failing closed to []")
        return []

    rows = await _fetch_rows(query, str(filters.workspace_id), limit)
    if not rows:
        return []

    top_score = max(float(r["score"]) for r in rows) or 1.0
    candidates: List[Dict] = []
    for r in rows:
        source_file = r["source_file"] or "unknown"
        metadata = _as_dict(r["metadata"], {})
        candidates.append(
            {
                "id": f"pg-chunk-{r['id']}",
                "content": r["content"] or "",
                "source_file": source_file,
                "document_id": r["document_id"],
                "file_type": source_file.rsplit(".", 1)[-1] if "." in source_file else "",
                "similarity": float(r["score"]) / top_score,
                "metadata": {**metadata, "chunk_index": r["chunk_index"], "retrieval_leg": "bm25"},
                "parent_content": r["parent_content"],
                "headers": _as_dict(r["headers"], {}),
            }
        )
    logger.info(
        f"BM25 sparse leg: {len(candidates)} candidates (workspace={filters.workspace_id})"
    )
    return candidates
