"""
Activity Command Centre API (PRD-72 US-005)
=============================================

Unified activity feed and stats endpoints for the Activity page.
Delegates to ActivityService for data merging from chats, heartbeats,
and recipe executions.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from services.activity_service import ActivityService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/activity", tags=["Activity"])


@router.get("/feed")
async def get_activity_feed(
    type: Optional[str] = Query(
        None,
        description="Comma-separated activity types to include: chat,routine,recipe",
    ),
    status: Optional[str] = Query(
        None,
        description="Status filter: working, done, attention, upcoming",
    ),
    period: str = Query(
        "7d",
        description="Time window: 1d, 7d, 30d, 90d",
    ),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Return a unified activity feed merging chats, routines, and recipes."""
    try:
        types = [t.strip() for t in type.split(",") if t.strip()] if type else None

        svc = ActivityService(db, ctx.workspace_id)
        result = svc.get_feed(
            types=types,
            status=status,
            period=period,
            limit=limit,
            offset=offset,
        )
        return result
    except Exception as e:
        logger.error("Activity feed error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch activity feed")


@router.get("/schedule")
async def get_activity_schedule(
    range: str = Query(
        "7d",
        description="Schedule range: 7d, 14d, 30d",
    ),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Return upcoming scheduled routines and recipes for the calendar widget."""
    try:
        days_map = {"7d": 7, "14d": 14, "30d": 30}
        range_days = days_map.get(range, 7)

        svc = ActivityService(db, ctx.workspace_id)
        return svc.get_schedule(range_days=range_days)
    except Exception as e:
        logger.error("Activity schedule error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch schedule")


@router.get("/agent-reports")
async def get_agent_reports(
    agent_ids: str = Query(
        ...,
        description="Comma-separated agent IDs to fetch reports for",
    ),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Return latest execution summaries for pinned agents."""
    try:
        ids = [int(x.strip()) for x in agent_ids.split(",") if x.strip().isdigit()]
        if not ids:
            return {"reports": []}

        svc = ActivityService(db, ctx.workspace_id)
        return svc.get_agent_reports(agent_ids=ids)
    except Exception as e:
        logger.error("Agent reports error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch agent reports")


@router.get("/stats")
async def get_activity_stats(
    period: str = Query(
        "1d",
        description="Time window for stats: 1d, 7d, 30d, 90d",
    ),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Return hero-card stats: working_now, channels_live, completed_today, needs_attention."""
    try:
        svc = ActivityService(db, ctx.workspace_id)
        return svc.get_stats(period=period)
    except Exception as e:
        logger.error("Activity stats error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch activity stats")
