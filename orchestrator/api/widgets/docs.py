"""
Widget Docs API — Team-Scoped Document Search
==============================================

REST endpoints for document search and retrieval with PRD-124 team-based
access control.  All queries honour the ``team_access`` column so agents
and widgets only see documents tagged for their team.

Prefix: ``/docs`` (mounted under ``/api/widgets``)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.widgets.auth import WidgetAuthContext, require_permission, widget_auth
from core.database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/docs", tags=["Widget Docs"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class DocsSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Max results")
    team: Optional[str] = Field(None, description="Team filter override")


class DocsSearchItem(BaseModel):
    id: str
    title: str
    snippet: str
    tags: Optional[List[str]] = None


class DocsSearchResponse(BaseModel):
    query: str
    results: List[DocsSearchItem]
    total: int


class DocsDetailResponse(BaseModel):
    id: str
    title: str
    content: str
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None


class DocsCategoryItem(BaseModel):
    tag: str
    count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _effective_team(auth: WidgetAuthContext, request_team: Optional[str]) -> Optional[str]:
    """API key team overrides request-level team parameter."""
    return auth.team or request_team or None


_TEAM_FILTER_CLAUSE = (
    "AND (team_access = '{}' OR :team = ANY(team_access))"
)


# ---------------------------------------------------------------------------
# POST /docs/search
# ---------------------------------------------------------------------------

@router.post("/search", response_model=DocsSearchResponse)
async def docs_search(
    body: DocsSearchRequest,
    auth: WidgetAuthContext = Depends(require_permission("documents:read")),
    db: Session = Depends(get_db),
):
    """Search documents filtered by team_access."""
    workspace_id = str(auth.workspace_id)
    team = _effective_team(auth, body.team)

    params: Dict[str, Any] = {
        "ws": workspace_id,
        "q": f"%{body.query}%",
        "limit": body.limit,
    }

    team_clause = ""
    if team:
        team_clause = _TEAM_FILTER_CLAUSE
        params["team"] = team

    sql = (
        "SELECT id, title, COALESCE(content, '') AS content, team_access "
        "FROM documents "
        f"WHERE workspace_id = :ws AND (title ILIKE :q OR content ILIKE :q) {team_clause} "
        "ORDER BY updated_at DESC NULLS LAST "
        "LIMIT :limit"
    )

    rows = db.execute(text(sql), params).fetchall()

    results = [
        DocsSearchItem(
            id=str(row.id),
            title=row.title or "Untitled",
            snippet=(row.content or "")[:200],
            tags=row.team_access if row.team_access else [],
        )
        for row in rows
    ]

    return DocsSearchResponse(query=body.query, results=results, total=len(results))


# ---------------------------------------------------------------------------
# GET /docs/{document_id}
# ---------------------------------------------------------------------------

@router.get("/{document_id}", response_model=DocsDetailResponse)
async def docs_detail(
    document_id: int,
    team: Optional[str] = Query(None, description="Team filter"),
    auth: WidgetAuthContext = Depends(require_permission("documents:read")),
    db: Session = Depends(get_db),
):
    """Retrieve a single document, 404 if team-blocked."""
    workspace_id = str(auth.workspace_id)
    effective = _effective_team(auth, team)

    params: Dict[str, Any] = {"ws": workspace_id, "doc_id": document_id}
    team_clause = ""
    if effective:
        team_clause = _TEAM_FILTER_CLAUSE
        params["team"] = effective

    sql = (
        "SELECT id, title, content, team_access, metadata, created_at "
        "FROM documents "
        f"WHERE id = :doc_id AND workspace_id = :ws {team_clause}"
    )

    row = db.execute(text(sql), params).fetchone()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    return DocsDetailResponse(
        id=str(row.id),
        title=row.title or "Untitled",
        content=row.content or "",
        tags=row.team_access if row.team_access else [],
        metadata=row.metadata if hasattr(row, "metadata") else None,
        created_at=row.created_at.isoformat() if row.created_at else None,
    )


# ---------------------------------------------------------------------------
# GET /docs/categories
# ---------------------------------------------------------------------------

@router.get("/categories", response_model=List[DocsCategoryItem])
async def docs_categories(
    team: Optional[str] = Query(None, description="Team filter"),
    auth: WidgetAuthContext = Depends(require_permission("documents:read")),
    db: Session = Depends(get_db),
):
    """Return distinct tags from team-scoped documents."""
    workspace_id = str(auth.workspace_id)
    effective = _effective_team(auth, team)

    params: Dict[str, Any] = {"ws": workspace_id}
    team_clause = ""
    if effective:
        team_clause = _TEAM_FILTER_CLAUSE
        params["team"] = effective

    sql = (
        "SELECT tag, COUNT(*) AS cnt "
        "FROM documents, UNNEST(team_access) AS tag "
        f"WHERE workspace_id = :ws {team_clause} "
        "GROUP BY tag ORDER BY cnt DESC"
    )

    rows = db.execute(text(sql), params).fetchall()

    return [DocsCategoryItem(tag=row.tag, count=row.cnt) for row in rows]
