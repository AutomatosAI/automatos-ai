"""
Widget Documents API
====================

REST endpoints for document search and retrieval consumed by embedded SDK
widgets.  All queries are workspace-scoped (and team-scoped when the widget key
is team-locked) using the authenticated context from :mod:`api.widgets.auth`.

PRD-158 S5 — schema truth: this surface previously queried ``documents.title`` /
``documents.content`` / ``documents.created_at`` and typed ids as UUID. None of
those exist — ``documents`` has ``original_filename`` / ``upload_date`` /
``doc_metadata`` and an INTEGER id, and chunk text lives in ``document_chunks``.
The search was therefore impossible to satisfy. It is rebuilt here on the real
schema and the PRD-157 retrieval path (vector search via ``RAGService``), not
ILIKE over a non-existent column.

Prefix: /api/widget/documents
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.widgets.auth import WidgetAuthContext, require_permission, widget_auth
from core.database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/widget/documents", tags=["Widget Documents"])

# ---------------------------------------------------------------------------
# Pydantic models  (ids are INTEGER — documents.id is a serial, not a UUID)
# ---------------------------------------------------------------------------


class DocumentSearchRequest(BaseModel):
    """Body for POST /documents/search."""

    query: str = Field(..., min_length=1, description="Search query string")
    limit: int = Field(default=10, ge=1, le=20, description="Max results (1-20)")


class DocumentSearchItem(BaseModel):
    """Single search result (one per surfaced document)."""

    id: int
    title: str
    snippet: str
    score: Optional[float] = None


class DocumentSearchResponse(BaseModel):
    """Response for POST /documents/search."""

    query: str
    results: List[DocumentSearchItem]
    total: int


class DocumentDetail(BaseModel):
    """Full document returned by GET /documents/{document_id}."""

    id: int
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(
    body: DocumentSearchRequest,
    auth: WidgetAuthContext = Depends(widget_auth),
    _perm: WidgetAuthContext = Depends(require_permission("documents:read")),
    db: Session = Depends(get_db),
):
    """Semantic (vector) search over the workspace knowledge base.

    Scoped to the widget's workspace and — when the key is team-locked — its team,
    via the PRD-157 retrieval path. Results are grouped by document (best chunk
    per document), so one row per surfaced document.
    """
    from modules.rag.service import RAGService
    from modules.rag.retrieval_filters import build_retrieval_filters

    filters = build_retrieval_filters(workspace_id=auth.workspace_id, team=auth.team)

    try:
        result = await RAGService().retrieve(
            query=body.query,
            max_chunks=body.limit,
            workspace_id=filters.workspace_id,
            team=filters.team,
        )
        chunks = result.chunks if result else []
    except Exception:
        logger.warning("[widget-docs] semantic search failed", exc_info=True)
        chunks = []

    # Group by document, keeping the best (highest-scored, first) chunk per doc.
    by_doc: Dict[int, Dict[str, Any]] = {}
    for chunk in chunks:
        meta = chunk.get("metadata", {}) or {}
        raw_id = chunk.get("document_id") or meta.get("document_id") or meta.get("external_file_id")
        if raw_id is None:
            continue
        try:
            doc_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if doc_id not in by_doc:
            by_doc[doc_id] = chunk

    # Resolve real display names from PostgreSQL (chunk source_file may be a temp name).
    names: Dict[int, str] = {}
    if by_doc:
        rows = db.execute(
            text(
                "SELECT id, COALESCE(original_filename, filename) AS name "
                "FROM documents WHERE id = ANY(CAST(:ids AS int[]))"
            ),
            {"ids": list(by_doc.keys())},
        ).fetchall()
        names = {r.id: r.name for r in rows}

    results = [
        DocumentSearchItem(
            id=doc_id,
            title=names.get(doc_id) or chunk.get("source_file") or "Document",
            snippet=(chunk.get("content") or "")[:200],
            score=float(chunk["similarity"]) if chunk.get("similarity") is not None else None,
        )
        for doc_id, chunk in list(by_doc.items())[: body.limit]
    ]

    return DocumentSearchResponse(query=body.query, results=results, total=len(results))


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: int,
    auth: WidgetAuthContext = Depends(widget_auth),
    _perm: WidgetAuthContext = Depends(require_permission("documents:read")),
    db: Session = Depends(get_db),
):
    """Retrieve a single document by ID (workspace-scoped).

    Content is reassembled from ``document_chunks`` (in ``chunk_index`` order),
    since ``documents`` has no content column.
    """
    doc = db.execute(
        text(
            "SELECT id, COALESCE(original_filename, filename) AS title, upload_date, doc_metadata "
            "FROM documents WHERE id = :id AND workspace_id = CAST(:ws_id AS uuid)"
        ),
        {"id": document_id, "ws_id": str(auth.workspace_id)},
    ).fetchone()

    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    chunk_rows = db.execute(
        text(
            "SELECT content FROM document_chunks "
            "WHERE document_id = :id ORDER BY chunk_index"
        ),
        {"id": document_id},
    ).fetchall()
    content = "\n\n".join(r.content for r in chunk_rows if r.content)

    return DocumentDetail(
        id=doc.id,
        title=doc.title,
        content=content,
        metadata=doc.doc_metadata,
        created_at=doc.upload_date,
    )
