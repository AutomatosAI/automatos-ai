"""
Widget Data Query API
======================

Provides a natural-language-to-SQL query for embedded SDK widgets, scoped
to the key's workspace by the NL2SQL service.

Endpoints:

    POST /data/query    — NL question  -> SQL + results

F155: the raw-SQL endpoint (POST /data/execute) is gone — caller SQL on the
shared database cannot be scoped to one workspace by string checks.
"""

from __future__ import annotations

import logging
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.widgets.auth import WidgetAuthContext, require_permission, widget_auth
from core.database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Widget Data"])

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class NLQueryRequest(BaseModel):
    """Natural language question to be translated to SQL."""

    question: str = Field(..., min_length=1, description="Natural language question")


class NLQueryResponse(BaseModel):
    """Result of a natural language query."""

    sql: str
    columns: List[str]
    rows: List[List[Any]]
    summary: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/data/query", response_model=NLQueryResponse)
async def nl_query(
    body: NLQueryRequest,
    auth: WidgetAuthContext = Depends(widget_auth),
    _perm: WidgetAuthContext = Depends(require_permission("data:query")),
    db: Session = Depends(get_db),
) -> NLQueryResponse:
    """Translate a natural language question into SQL, execute it, and return
    the results along with a human-readable summary."""

    try:
        from consumers.data import DataQueryService  # type: ignore[import-untyped]
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data query service not available",
        )

    try:
        service = DataQueryService()
        result = await service.query(
            workspace_id=auth.workspace_id,
            question=body.question,
            db=db,
        )

        return NLQueryResponse(
            sql=result.get("sql", ""),
            columns=result.get("columns", []),
            rows=result.get("rows", []),
            summary=result.get("summary", ""),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("NL query failed for workspace %s", auth.workspace_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {exc}",
        )
