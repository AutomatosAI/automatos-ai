"""
Activity Command Centre API (PRD-72 US-005)
=============================================

Unified activity feed and stats endpoints for the Activity page.
Delegates to ActivityService for data merging from chats, heartbeats,
and recipe executions.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from pydantic import BaseModel

from core.models.core import Agent, BoardTask, DigestFeedback
from services.activity_service import ActivityService
from services.digest_service import generate_digest

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


@router.get("/digest")
async def get_workspace_digest(
    period: str = Query("1d", description="Time window: 1d, 7d, 30d, 90d"),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Auto's Read — a cached plain-English summary of the workspace (PRD-221 S9).

    Cached per (workspace, state_hash): the digest LLM fires at most once per
    real state change. Never 500s — a degraded model returns a deterministic
    fallback. Response: {text, generated_at, state_hash, needs_attention_count}.
    """
    try:
        return await generate_digest(db, ctx.workspace_id, period=period)
    except Exception as e:
        # Defence in depth — generate_digest already falls back internally.
        logger.error("Digest error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to build digest")


class DigestFeedbackRequest(BaseModel):
    state_hash: str
    rating: int  # -1 (down) or 1 (up)


@router.post("/digest/feedback")
async def submit_digest_feedback(
    body: DigestFeedbackRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Record a thumbs up/down on an Auto's Read digest (PRD-221 S10).

    Keyed by the digest's state_hash — feedback attaches to the workspace state
    the read described. rating must be -1 or 1; anything else is a 422.
    """
    if body.rating not in (-1, 1):
        raise HTTPException(status_code=422, detail="rating must be -1 or 1")
    row = DigestFeedback(
        workspace_id=ctx.workspace_id,
        user_id=ctx.clerk_user_id,
        state_hash=body.state_hash,
        rating=body.rating,
    )
    db.add(row)
    db.commit()
    return {"ok": True}


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


@router.get("/scheduler-health")
async def get_scheduler_health(
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Advisory scheduler health for the calendar banner (PRD-162 S2).

    Separate from /schedule and NON-BLOCKING: the calendar renders configured
    schedules regardless (Q49). Returns ``healthy: null`` when it can't tell.
    """
    try:
        svc = ActivityService(db, ctx.workspace_id)
        return svc.get_scheduler_health()
    except Exception as e:
        logger.error("Scheduler health error: %s", e, exc_info=True)
        return {"healthy": None, "last_fired_at": None}


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


@router.get("/board/stats")
async def get_board_stats(
    period: str = Query(
        "1d",
        description="Time window: 1d, 7d, 30d, 90d",
    ),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Return board-level stats for dashboard widgets (status, priority, types, workload)."""
    try:
        days_map = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}
        days = days_map.get(period, 7)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        base_q = db.query(BoardTask).filter(
            BoardTask.workspace_id == ctx.workspace_id,
            BoardTask.created_at >= cutoff,
        )

        # Status columns
        status_rows = (
            base_q
            .with_entities(BoardTask.status, func.count(BoardTask.id))
            .group_by(BoardTask.status)
            .all()
        )
        all_statuses = ["inbox", "assigned", "in_progress", "review", "blocked", "done"]
        status_map = dict(status_rows)
        columns = [
            {"status": s, "count": status_map.get(s, 0)}
            for s in all_statuses
        ]
        total_tasks = sum(c["count"] for c in columns)

        # Priority breakdown
        priority_rows = (
            base_q
            .with_entities(BoardTask.priority, func.count(BoardTask.id))
            .group_by(BoardTask.priority)
            .all()
        )
        priorities = [
            {"priority": p, "count": c}
            for p, c in priority_rows
        ]

        # Types of work (source_type: user, mission, recipe, routine, heartbeat)
        type_rows = (
            base_q
            .with_entities(BoardTask.source_type, func.count(BoardTask.id))
            .group_by(BoardTask.source_type)
            .all()
        )
        # Map source_type to widget categories
        type_map = {
            "user": "recipe",
            "recipe": "recipe",
            "mission": "mission",
            "heartbeat": "routine",
            "routine": "routine",
        }
        type_counts: dict[str, int] = {}
        for source_type, count in type_rows:
            category = type_map.get(source_type, "recipe")
            type_counts[category] = type_counts.get(category, 0) + count
        types = [
            {
                "type": t,
                "count": c,
                "percentage": round((c / total_tasks) * 100, 1) if total_tasks > 0 else 0,
            }
            for t, c in type_counts.items()
        ]

        # Agent workload
        workload_rows = (
            base_q
            .filter(BoardTask.assigned_agent_id.isnot(None))
            .with_entities(
                BoardTask.assigned_agent_id,
                func.count(BoardTask.id),
            )
            .group_by(BoardTask.assigned_agent_id)
            .order_by(func.count(BoardTask.id).desc())
            .limit(10)
            .all()
        )
        agent_ids = [r[0] for r in workload_rows]
        agents_by_id = {}
        if agent_ids:
            agents = db.query(Agent).filter(Agent.id.in_(agent_ids)).all()
            agents_by_id = {a.id: a for a in agents}

        workload = []
        for agent_id, task_count in workload_rows:
            agent = agents_by_id.get(agent_id)
            workload.append({
                "agent_id": agent_id,
                "agent_name": agent.name if agent else f"Agent {agent_id}",
                "agent_icon": agent.marketplace_icon if agent else None,
                "task_count": task_count,
            })

        return {
            "columns": columns,
            "total_tasks": total_tasks,
            "priorities": priorities,
            "types": types,
            "workload": workload,
        }
    except Exception as e:
        logger.error("Board stats error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch board stats")


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
