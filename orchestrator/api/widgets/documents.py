"""
Widget Documents API
====================

REST endpoints for document search and retrieval consumed by embedded SDK
widgets.  All queries are workspace-scoped using the authenticated context
from :mod:`api.widgets.auth`.

Prefix: /api/widget/documents
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.widgets.auth import WidgetAuthContext, require_permission, widget_auth
from core.database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/widget/documents", tags=["Widget Documents"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class DocumentSearchRequest(BaseModel):
    """Body for POST /documents/search."""

    query: str = Field(..., min_length=1, description="Search query string")
    limit: int = Field(default=10, ge=1, le=20, description="Max results (1-20)")


class DocumentSearchItem(BaseModel):
    """Single search result."""

    id: UUID
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

    id: UUID
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
    """Search the workspace knowledge base by title or content."""

    like_pattern = f"%{body.query}%"

    rows = db.execute(
        text(
            "SELECT id, title, content, similarity "
            "FROM documents "
            "WHERE workspace_id = :ws_id "
            "  AND (title ILIKE :q OR content ILIKE :q) "
            "LIMIT :limit"
        ),
        {"ws_id": str(auth.workspace_id), "q": like_pattern, "limit": body.limit},
    ).fetchall()

    results = [
        DocumentSearchItem(
            id=row.id,
            title=row.title,
            snippet=(row.content or "")[:200],
            score=getattr(row, "similarity", None),
        )
        for row in rows
    ]

    return DocumentSearchResponse(
        query=body.query,
        results=results,
        total=len(results),
    )


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: UUID,
    auth: WidgetAuthContext = Depends(widget_auth),
    _perm: WidgetAuthContext = Depends(require_permission("documents:read")),
    db: Session = Depends(get_db),
):
    """Retrieve a single document by ID (workspace-scoped)."""

    row = db.execute(
        text(
            "SELECT id, title, content, metadata, created_at "
            "FROM documents "
            "WHERE id = :id AND workspace_id = :ws_id"
        ),
        {"id": str(document_id), "ws_id": str(auth.workspace_id)},
    ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return DocumentDetail(
        id=row.id,
        title=row.title,
        content=row.content,
        metadata=row.metadata,
        created_at=row.created_at,
    )
