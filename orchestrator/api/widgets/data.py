"""
Widget Data Query API
======================

Provides natural-language-to-SQL query and read-only SQL execution for
embedded SDK widgets.

Endpoints:

    POST /data/query    — NL question  -> SQL + results
    POST /data/execute  — Raw SQL (SELECT only) -> results
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.widgets.auth import WidgetAuthContext, require_permission, widget_auth
from core.database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Widget Data"])

# ---------------------------------------------------------------------------
# Blocked SQL keywords (anything that mutates data or schema)
# ---------------------------------------------------------------------------

_BLOCKED_KEYWORDS: re.Pattern = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE|GRANT|REVOKE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)


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


class SQLExecuteRequest(BaseModel):
    """Raw SQL to execute (SELECT only)."""

    sql: str = Field(..., min_length=1, description="SQL SELECT statement to execute")


class SQLExecuteResponse(BaseModel):
    """Result of a raw SQL execution."""

    columns: List[str]
    rows: List[List[Any]]
    row_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_readonly(sql: str) -> None:
    """Raise 400 if *sql* is not a read-only SELECT statement."""
    stripped = sql.strip()

    if not stripped.upper().startswith("SELECT"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only SELECT statements are allowed",
        )

    if _BLOCKED_KEYWORDS.search(stripped):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Statement contains disallowed keywords. Only read-only SELECT queries are permitted.",
        )


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


@router.post("/data/execute", response_model=SQLExecuteResponse)
async def execute_sql(
    body: SQLExecuteRequest,
    auth: WidgetAuthContext = Depends(widget_auth),
    _perm: WidgetAuthContext = Depends(require_permission("data:execute")),
    db: Session = Depends(get_db),
) -> SQLExecuteResponse:
    """Execute a read-only SQL SELECT statement against the workspace database
    and return the result set."""

    _validate_readonly(body.sql)

    try:
        result = db.execute(text(body.sql))
        columns = list(result.keys())
        rows = [list(row) for row in result.fetchall()]

        return SQLExecuteResponse(
            columns=columns,
            rows=rows,
            row_count=len(rows),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("SQL execute failed for workspace %s", auth.workspace_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SQL execution failed: {exc}",
        )
