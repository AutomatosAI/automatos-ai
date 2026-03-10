"""
Agent Reports API (PRD-76)
===========================

CRUD endpoints for agent reports: list, get, download, grade, stats.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.workspace_client import WorkspaceClient
from services.report_service import ReportService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reports", tags=["reports"])


# ── List Reports ─────────────────────────────────────────────────────
@router.get("")
async def list_reports(
    agent_id: Optional[int] = Query(None, description="Filter by agent ID"),
    report_type: Optional[str] = Query(None, description="Filter by type: standup, research, incident, summary, delivery, audit"),
    status: Optional[str] = Query(None, description="Filter by status: ok, warning, critical, info"),
    graded: Optional[bool] = Query(None, description="Filter by graded (true) or ungraded (false)"),
    period: str = Query("30d", description="Time period: 1d, 7d, 30d, 90d"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List reports for the current workspace with optional filters."""
    svc = ReportService(db, ctx.workspace_id)
    return await svc.list_reports(
        agent_id=agent_id,
        report_type=report_type,
        status=status,
        graded=graded,
        period=period,
        limit=limit,
        offset=offset,
    )


# ── Get Single Report ────────────────────────────────────────────────
@router.get("/stats")
async def report_stats(
    period: str = Query("7d", description="Time period: 1d, 7d, 30d, 90d"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Aggregate report stats for the workspace."""
    svc = ReportService(db, ctx.workspace_id)
    return await svc.get_stats(period=period)


@router.get("/{report_id}")
async def get_report(
    report_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get a single report with its content."""
    svc = ReportService(db, ctx.workspace_id)
    result = await svc.get_report(report_id, include_content=True)

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Report not found"))

    return result


# ── Download Report File ─────────────────────────────────────────────
@router.get("/{report_id}/download")
async def download_report(
    report_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Download the report file."""
    from fastapi.responses import Response

    svc = ReportService(db, ctx.workspace_id)
    result = await svc.get_report(report_id, include_content=True)

    if not result.get("success"):
        raise HTTPException(status_code=404, detail="Report not found")

    report = result["report"]
    content = report.get("content")
    if content is None:
        raise HTTPException(
            status_code=502,
            detail=report.get("content_error", "Report file could not be read from storage"),
        )
    filename = report["file_path"].split("/")[-1]

    return Response(
        content=content.encode("utf-8"),
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Grade Report ─────────────────────────────────────────────────────
class GradeRequest(BaseModel):
    grade: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    grade_notes: Optional[str] = Field(None, max_length=1000)


@router.patch("/{report_id}/grade")
async def grade_report(
    report_id: str,
    body: GradeRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Submit a grade for a report."""
    svc = ReportService(db, ctx.workspace_id)
    result = await svc.grade_report(
        report_id=report_id,
        grade=body.grade,
        grade_notes=body.grade_notes,
        graded_by=ctx.user_id,
    )

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Report not found"))

    return result


# ── Agent Reports ────────────────────────────────────────────────────
@router.get("/agent/{agent_id}")
async def agent_reports(
    agent_id: int,
    report_type: Optional[str] = Query(None),
    period: str = Query("30d"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get reports for a specific agent."""
    svc = ReportService(db, ctx.workspace_id)
    return await svc.list_reports(
        agent_id=agent_id,
        report_type=report_type,
        period=period,
        limit=limit,
        offset=offset,
    )
