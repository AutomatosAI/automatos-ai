"""Document access tracking: a retrieval stamps ``last_accessed`` and counts the query.

Moved out of ``RAGService._track_document_access`` (modules/rag/service.py is over 800
lines). ``RAGService._schedule_access_tracking`` runs it off the event loop, on a session
of its own. Analytics only, so a failed write never fails the retrieval.

PRD-268 P268W1-RVW-1 (K17): ``platform_search_documents`` is labelled read, so a check-in's
search stamps nothing (``core.read_only_call``).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

from sqlalchemy import text

from core.read_only_call import only_reading

logger = logging.getLogger(__name__)

_STAMP = text("""
    UPDATE documents
    SET last_accessed = NOW(),
        rag_query_count = COALESCE(rag_query_count, 0) + 1
    WHERE id = ANY(CAST(:ids AS int[])) AND workspace_id = CAST(:ws AS uuid)
""")


def retrieved_document_ids(chunks: List[Dict[str, Any]]) -> Set[str]:
    """The ids of the documents ``chunks`` came from, as strings."""
    ids = set()
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        doc_id = meta.get("document_id") or meta.get("doc_id")
        if doc_id:
            ids.add(str(doc_id))
    return ids


def track_document_access(chunks: List[Dict[str, Any]], workspace_id: str) -> None:
    """Stamp the retrieved documents of ``workspace_id``; nothing in a check-in. Blocking."""
    if only_reading():
        return
    try:
        doc_ids = retrieved_document_ids(chunks)
        if doc_ids:
            _stamp(doc_ids, workspace_id)
    except Exception:  # noqa: BLE001 — logged; analytics never fail the retrieval
        logger.exception("Document access tracking failed")


def _stamp(doc_ids: Set[str], workspace_id: str) -> None:
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        db.execute(_STAMP, {"ids": list(doc_ids), "ws": str(workspace_id)})
        db.commit()
    finally:
        db.close()


__all__ = ["retrieved_document_ids", "track_document_access"]
