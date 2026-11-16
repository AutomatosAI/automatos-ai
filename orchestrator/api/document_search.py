"""What ``POST /api/documents/search`` does around the retrieval funnel (``api/documents.py``
``semantic_search``, which runs the funnel through ``_retrieval_hits``).

Moved from ``api/documents.py`` as it was (code shape: the route was one function of over 200
lines, in a file well over the size limit), but for one rule, PRD-274 §5.2's, marked below. In
the order the route takes them:

* **The parameters,** from the JSON body or the query string, the body first. Night 1
  (2026-09-18): every agent that reached the route sent a JSON body and got a 422, because the
  parameters were query-string only. A missing query is a 422 that says so.
* **Each hit's document** in the caller's workspace: by the hit's own document id (the pgvector
  backend carries it, and it survives two documents sharing a file name), else by its file name,
  which is all the S3 payload allows. **A server key's hit is never resolved by file name**: the
  route keeps only the hits naming by id a document the key reaches
  (``services.document_key_access.reachable_hits``), so a document sharing a name with one the
  key can't read never answers with that one's text.
* **Grouped per document,** the best similarity and its excerpt kept, then held to the team
  scope (PRD-157 S1, the fail-closed choke point; a no-op with no team).
* **Previews and chunk counts** for the documents shown, in batched queries.
* **The search tracked** in ``document_usage``, attributed to its workspace (PRD-156 S5).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException
from sqlalchemy import text

logger = logging.getLogger(__name__)

SEARCH_DEFAULT_LIMIT = 10           # Results returned when the caller names no limit
SEARCH_MAX_LIMIT = 50               # Hard ceiling, mirrored by the query-param validator
PREVIEW_CHUNK_LIMIT = 6  # Max number of chunks included in preview window
PREVIEW_CONTEXT_RADIUS = 2  # Number of chunks to include before the best match
PREVIEW_CHAR_LIMIT = 2000  # Safety limit for preview text length
QUERY_REQUIRED = "'query' is required — send it in the JSON body or as a query parameter."

_DOCUMENT_BY_ID = text("""
    SELECT id, filename, file_type, file_size, upload_date
    FROM documents
    WHERE workspace_id = :workspace_id AND id = :document_id
    LIMIT 1
""")
_DOCUMENT_BY_FILE_NAME = text("""
    SELECT id, filename, file_type, file_size, upload_date
    FROM documents
    WHERE workspace_id = :workspace_id AND filename = :filename
    LIMIT 1
""")
_CHUNK_COUNTS = text("""
    SELECT document_id, COUNT(*) AS chunk_count
    FROM document_chunks
    WHERE document_id = ANY(:document_ids)
    GROUP BY document_id
""")
_TRACK_SEARCH = text("""
    INSERT INTO document_usage (event_type, query, results_count, execution_time_ms, metadata, timestamp)
    VALUES ('document_searched', :query, :results_count, :execution_time_ms, :metadata, :timestamp)
""")


@dataclass(frozen=True)
class SearchAsk:
    """What a search asks for, its parameters settled."""

    query: str
    limit: int
    min_similarity: Optional[float]
    document_ids: Optional[List[int]]
    team: Optional[str]


def _body_first(body: Any, name: str, value: Any) -> Any:
    """The JSON body's ``name`` where it carries one, else ``value`` (the query string's)."""
    carried = getattr(body, name, None) if body is not None else None
    return carried if carried is not None else value


def search_ask(body: Any, query: Optional[str], limit: Optional[int], min_similarity: Optional[float],
               document_ids: Optional[List[int]], team: Optional[str]) -> SearchAsk:
    """The search's parameters, the body's first; a 422 that says so when there is no query."""
    query = _body_first(body, "query", query)
    if not query or not str(query).strip():
        raise HTTPException(status_code=422, detail=QUERY_REQUIRED)
    limit = _body_first(body, "limit", limit)
    min_similarity = _body_first(body, "min_similarity", min_similarity)
    return SearchAsk(
        query=str(query).strip(),
        limit=SEARCH_DEFAULT_LIMIT if limit is None else max(1, min(int(limit), SEARCH_MAX_LIMIT)),
        min_similarity=None if min_similarity is None else float(min_similarity),
        document_ids=_body_first(body, "document_ids", document_ids),
        team=_body_first(body, "team", team),
    )


def _hit_document(db: Any, workspace_id: Any, hit: Dict[str, Any], by_file_name: bool) -> Any:
    """The workspace's document a hit is from: by the hit's own document id, else, when
    ``by_file_name``, by its file name; None when neither finds one."""
    hit_doc_id = (hit.get("metadata") or {}).get("document_id")
    doc_row = None
    if hit_doc_id is not None:
        try:
            doc_row = db.execute(_DOCUMENT_BY_ID, {"workspace_id": workspace_id,
                                                   "document_id": int(hit_doc_id)}).fetchone()
        except (TypeError, ValueError):
            doc_row = None
    file_name = hit.get("file_name", "")
    if not doc_row and file_name and by_file_name:
        doc_row = db.execute(_DOCUMENT_BY_FILE_NAME, {"workspace_id": workspace_id, "filename": file_name}).fetchone()
    return doc_row


def _first_entry(doc_row: Any, hit: Dict[str, Any]) -> Dict[str, Any]:
    """A document's entry from its first hit: the hit's score, text and chunk, the document's source."""
    return {
        "document_id": doc_row.id,
        "best_similarity": hit.get("score", 0.0),
        "best_excerpt": hit.get("content", ""),
        "best_chunk_index": hit.get("chunk_index", 0),
        "metadata": {},
        "source": {
            "filename": doc_row.filename,
            "file_type": doc_row.file_type,
            "file_size": doc_row.file_size,
            "upload_date": doc_row.upload_date.isoformat() if doc_row.upload_date else None,
        },
    }


def grouped_hits(db: Any, workspace_id: Any, hits: List[Dict[str, Any]], document_ids: Optional[List[int]],
                 *, by_file_name: bool) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    """Each hit mapped to its document and grouped per document, the best similarity and its
    excerpt kept; and the documents in the order first hit. ``document_ids``, when given, keeps
    only those documents. ``by_file_name`` False: a hit is mapped by its own document id only."""
    grouped: Dict[int, Dict[str, Any]] = {}
    order: List[int] = []
    for hit in hits:
        doc_row = _hit_document(db, workspace_id, hit, by_file_name)
        if not doc_row or (document_ids and doc_row.id not in document_ids):
            continue
        entry = grouped.get(doc_row.id)
        if entry is None:
            grouped[doc_row.id] = _first_entry(doc_row, hit)
            order.append(doc_row.id)
        elif hit.get("score", 0.0) > entry["best_similarity"]:
            grouped[doc_row.id] = {**entry, "best_similarity": hit.get("score", 0.0),
                                   "best_excerpt": hit.get("content", ""), "best_chunk_index": hit.get("chunk_index", 0)}
    return grouped, order


def in_team_scope(db: Any, workspace_id: Any, team: Optional[str], order: List[int]) -> List[int]:
    """``order`` held to the team scope (PRD-157 S1: the centralized fail-closed choke point).
    The workspace is already held per hit; with no team this leaves ``order`` as it is."""
    from modules.rag.retrieval_filters import allowed_document_ids, build_retrieval_filters

    scope = build_retrieval_filters(workspace_id=workspace_id, team=team)
    if not (scope.has_team_restriction and order):
        return order
    allowed = allowed_document_ids(db, order, scope)
    return [doc_id for doc_id in order if str(doc_id) in allowed]


def _preview_range(entry: Dict[str, Any]) -> Tuple[int, int]:
    """The chunks a document's preview shows: from just before its best match, at most
    ``PREVIEW_CHUNK_LIMIT`` of them."""
    start = max((entry.get("best_chunk_index") or 0) - PREVIEW_CONTEXT_RADIUS, 0)
    return start, start + PREVIEW_CHUNK_LIMIT - 1


def _chunk_counts(db: Any, doc_ids: List[int]) -> Dict[int, int]:
    """Each document's chunk count, in one query."""
    rows = db.execute(_CHUNK_COUNTS, {"document_ids": doc_ids}).fetchall()
    return {int(row.document_id): int(row.chunk_count) for row in rows}


def _previews(db: Any, ranges: Dict[int, Tuple[int, int]]) -> Dict[int, str]:
    """Each document's preview text, its chunks in the range joined, in one UNION ALL query with
    one SELECT per document."""
    selects: List[str] = []
    params: Dict[str, Any] = {}
    for idx, (doc_id, (start, end)) in enumerate(ranges.items()):
        params.update({f"doc_{idx}": doc_id, f"start_{idx}": start, f"end_{idx}": end})
        # Only the bound parameters' own names are formatted in; every value is bound.
        selects.append(
            f"""
            SELECT
                :doc_{idx} AS document_id,
                string_agg(content, '\n\n' ORDER BY chunk_index) AS preview_content
            FROM document_chunks
            WHERE document_id = :doc_{idx}
              AND chunk_index BETWEEN :start_{idx} AND :end_{idx}
            """  # noqa: S608
        )
    if not selects:
        return {}
    rows = db.execute(text(" UNION ALL ".join(selects)), params).fetchall()
    return {int(row.document_id): (row.preview_content if row.preview_content else "") for row in rows}


def _result(entry: Dict[str, Any], preview_range: Tuple[int, int], chunk_count: int, preview: str) -> Dict[str, Any]:
    """One document's search result: its best excerpt and a preview window around it."""
    start, end = preview_range
    truncated = False
    if preview and len(preview) > PREVIEW_CHAR_LIMIT:
        preview = preview[:PREVIEW_CHAR_LIMIT] + "…"
        truncated = True
    if end < (chunk_count - 1):
        truncated = True
    return {
        "document_id": entry["document_id"],
        "excerpt": entry["best_excerpt"],
        "similarity": entry["best_similarity"],
        "chunk_index": entry["best_chunk_index"],
        "chunk_count": chunk_count,
        "metadata": entry["metadata"],
        "source": entry["source"],
        "preview": preview,
        "preview_truncated": truncated,
        "full_content_available": chunk_count > 0,
        "preview_chunk_start": start,
        "preview_chunk_end": min(end, max(chunk_count - 1, 0)),
    }


def aggregated_results(db: Any, grouped: Dict[int, Dict[str, Any]], doc_ids: List[int]) -> List[Dict[str, Any]]:
    """The results for ``doc_ids``, in that order, with their previews and chunk counts."""
    if not doc_ids:
        return []
    ranges = {doc_id: _preview_range(grouped[doc_id]) for doc_id in doc_ids}
    counts = _chunk_counts(db, doc_ids)
    previews = _previews(db, ranges)
    return [_result(grouped[doc_id], ranges[doc_id], counts.get(doc_id, 0), previews.get(doc_id, ""))
            for doc_id in doc_ids]


def track_search(db: Any, query: str, results_count: int, execution_time_ms: int,
                 min_similarity: Optional[float], workspace_id: Any) -> None:
    """Track the search for analytics. Best effort: an untracked search still answers."""
    try:
        db.execute(_TRACK_SEARCH, {
            "query": query,
            "results_count": results_count,
            "execution_time_ms": execution_time_ms,
            # PRD-156 S5: attribute each usage row to its workspace (the
            # table has no workspace_id column; metadata JSONB carries it so
            # analytics can filter by metadata->>'workspace_id' without a migration).
            "metadata": json.dumps({"min_similarity": min_similarity, "workspace_id": str(workspace_id)}),
            "timestamp": datetime.now(),
        })
        db.commit()
        logger.info(f"Search tracked: '{query[:50]}' ({results_count} results)")
    except Exception:  # noqa: BLE001 -- best effort, as before: an untracked search still answers
        logger.warning("Could not track search event", exc_info=True)
        db.rollback()


__all__ = [
    "SEARCH_DEFAULT_LIMIT", "SEARCH_MAX_LIMIT", "SearchAsk", "aggregated_results", "grouped_hits", "in_team_scope",
    "search_ask", "track_search",
]
