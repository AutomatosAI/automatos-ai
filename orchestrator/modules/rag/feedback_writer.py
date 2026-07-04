"""PRD-185 S7: the single writer for the ``rag_feedback`` table.

Both the explicit feedback endpoint (``POST /api/rag/feedback``) and the chat
thumbs vote (``PATCH /api/chat/vote``) land rows here, so the INSERT lives in
exactly one place. The PRD-179 live ranker reads these rows via
``UNNEST(document_ids)`` to shape retrieval — feeding it real votes is the whole
point of S7.

Pure DB helper — no framework — so callers (HTTP + vote path) and tests share it.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_INSERT = text("""
    INSERT INTO rag_feedback
        (query, response_text, chunk_ids, document_ids, rating,
         feedback_type, correction_text, rag_config_id,
         execution_time_ms, workspace_id, user_id, created_at)
    VALUES
        (:query, :response_text, :chunk_ids, :document_ids, :rating,
         :feedback_type, :correction_text, :rag_config_id,
         :execution_time_ms, :workspace_id, :user_id, NOW())
    RETURNING id
""")


def write_rag_feedback(
    db: Session,
    *,
    query: str,
    workspace_id: Any,
    user_id: Optional[int] = None,
    document_ids: Optional[List[int]] = None,
    chunk_ids: Optional[List[int]] = None,
    rating: Optional[int] = None,
    feedback_type: str = "thumbs_up",
    correction_text: Optional[str] = None,
    rag_config_id: Optional[int] = None,
    execution_time_ms: Optional[float] = None,
    response_text: Optional[str] = None,
    commit: bool = True,
) -> Optional[int]:
    """Insert one ``rag_feedback`` row and return its id.

    ``query`` is stored TEXT NOT NULL — an empty string is legal, ``None`` is not.
    ``document_ids`` / ``chunk_ids`` are ``INTEGER[]``; pass ``None`` (not ``[]``)
    to store SQL NULL. Set ``commit=False`` to write within a caller-owned
    transaction.
    """
    result = db.execute(_INSERT, {
        "query": query or "",
        "response_text": response_text,
        "chunk_ids": chunk_ids or None,
        "document_ids": document_ids or None,
        "rating": rating,
        "feedback_type": feedback_type,
        "correction_text": correction_text,
        "rag_config_id": rag_config_id,
        "execution_time_ms": execution_time_ms,
        "workspace_id": str(workspace_id) if workspace_id is not None else None,
        "user_id": user_id,
    })
    row = result.fetchone()
    if commit:
        db.commit()
    return row.id if row else None


def feedback_from_retrieval_context(
    db: Session,
    *,
    retrieval_context: Optional[dict],
    is_upvoted: bool,
    workspace_id: Any,
    user_id: Optional[int] = None,
    response_text: Optional[str] = None,
    commit: bool = True,
) -> Optional[int]:
    """Write a ``rag_feedback`` row from a chat vote's stored retrieval provenance.

    Returns the feedback id, or ``None`` when the voted message carried no
    retrieved documents (nothing to learn from — e.g. a pure-chat turn). A thumbs
    up/down maps to ``thumbs_up`` / ``thumbs_down`` so the PRD-179 ranker only
    penalises documents behind a down-voted answer.
    """
    if not isinstance(retrieval_context, dict):
        return None
    document_ids = [i for i in (retrieval_context.get("document_ids") or []) if isinstance(i, int)]
    chunk_ids = [i for i in (retrieval_context.get("chunk_ids") or []) if isinstance(i, int)]
    if not document_ids and not chunk_ids:
        return None
    return write_rag_feedback(
        db,
        query=retrieval_context.get("query") or "",
        workspace_id=workspace_id,
        user_id=user_id,
        document_ids=document_ids or None,
        chunk_ids=chunk_ids or None,
        feedback_type="thumbs_up" if is_upvoted else "thumbs_down",
        response_text=response_text,
        commit=commit,
    )
