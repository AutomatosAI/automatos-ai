"""
Missions REST API — PRD-82A/82B Sequential Mission Coordinator
===============================================================

CRUD + lifecycle + telemetry endpoints for missions. API uses "mission"
terminology; DB/backend uses "orchestration" (PRD-82A Section 10).

13 endpoints:
  POST   /api/missions           — create mission
  GET    /api/missions           — list missions (paginated, filterable)
  GET    /api/missions/stats     — aggregate mission stats (PRD-82B US-004)
  GET    /api/missions/{id}      — get mission detail
  GET    /api/missions/{id}/events  — paginated events (PRD-82B US-004)
  GET    /api/missions/{id}/cost    — token/cost breakdown (PRD-82B US-004)
  POST   /api/missions/{id}/approve  — approve plan
  POST   /api/missions/{id}/reject   — reject plan
  POST   /api/missions/{id}/review   — submit human review
  POST   /api/missions/{id}/pause    — pause mission
  POST   /api/missions/{id}/resume   — resume mission
  POST   /api/missions/{id}/cancel   — cancel mission

Agent telemetry:
  GET    /api/agents/{agent_id}/mission-history  — agent mission perf (PRD-82B US-004)

Source: PRD-82A Section 12, PRD-82B US-004
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, validator
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import Agent
from core.models.orchestration import (
    OrchestrationEvent,
    OrchestrationRun,
    OrchestrationTask,
)
from core.models.orchestration_enums import (
    EventType,
    RunState,
    TaskState,
    TERMINAL_RUN_STATES,
)
from modules.coordination.planner import PlanValidationError
from services.coordinator_service import get_coordinator_service
from services.orchestration_state import (
    ConflictError,
    InvalidTransitionError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/missions", tags=["missions"])


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class MissionCreateRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=5000, description="Natural-language goal")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional mission config overrides")


ALLOWED_MODIFICATION_KEYS = {"task_overrides", "notes", "agent_overrides"}


class MissionApproveRequest(BaseModel):
    modifications: Optional[Dict[str, Any]] = Field(None, description="Optional plan modifications")

    @validator("modifications")
    def validate_modifications(cls, v):
        if v is None:
            return v
        unknown = set(v.keys()) - ALLOWED_MODIFICATION_KEYS
        if unknown:
            raise ValueError(f"Unknown modification keys: {unknown}")
        # Cap total serialised size to prevent memory abuse
        import json
        if len(json.dumps(v)) > 10_000:
            raise ValueError("Modifications payload too large (max 10KB)")
        return v


class MissionRejectRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=2000, description="Rejection reason")


class MissionReviewRequest(BaseModel):
    verdict: str = Field(..., pattern="^(accept|reject)$", description="'accept' or 'reject'")
    task_feedback: Optional[Dict[str, str]] = Field(
        None,
        description="Map of task_id → feedback string. On reject, tasks with feedback get re-queued.",
    )

    @validator("task_feedback")
    def validate_task_feedback(cls, v):
        if v is None:
            return v
        if len(v) > 50:
            raise ValueError("Too many task feedback entries (max 50)")
        from uuid import UUID as UUIDType
        validated = {}
        for k, val in v.items():
            try:
                UUIDType(k)
            except ValueError:
                raise ValueError(f"Invalid task ID (not a UUID): {k}")
            validated[k] = val[:2000]  # cap feedback length
        return validated


class TaskResponse(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    task_type: Optional[str] = None
    sequence_number: int
    agent_role: Optional[str] = None
    state: str
    state_type: str
    assigned_agent_id: Optional[int] = None
    attempt_number: int = 0
    tokens_used: int = 0
    failure_reason_code: Optional[str] = None
    failure_detail: Optional[str] = None
    output_excerpt: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class EventResponse(BaseModel):
    id: str
    event_type: str
    actor_type: str
    actor_id: Optional[str] = None
    old_state: Optional[str] = None
    new_state: Optional[str] = None
    task_id: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MissionResponse(BaseModel):
    id: str
    workspace_id: str
    goal: str
    state: str
    state_type: str
    plan: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    output_summary: Optional[Dict[str, Any]] = None
    token_budget_estimate: Optional[int] = None
    tokens_used: int = 0
    max_retries: int = 3
    created_by: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MissionDetailResponse(MissionResponse):
    tasks: List[TaskResponse] = []
    recent_events: List[EventResponse] = []


class MissionListResponse(BaseModel):
    missions: List[MissionResponse]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Telemetry response models (PRD-82B US-004)
# ---------------------------------------------------------------------------


class PaginatedEventsResponse(BaseModel):
    events: List[EventResponse]
    total: int
    limit: int
    offset: int


class TaskCostBreakdown(BaseModel):
    task_id: str
    title: str
    tokens_used: int


class MissionCostResponse(BaseModel):
    mission_id: str
    total_tokens: int
    estimated_cost: float
    cost_per_1k_tokens: float
    tasks: List[TaskCostBreakdown]


class TopAgentStats(BaseModel):
    agent_id: int
    agent_name: str
    tasks_completed: int
    avg_tokens_per_task: float


class MissionStatsResponse(BaseModel):
    total_missions: int
    success_rate: float
    avg_duration_minutes: Optional[float] = None
    avg_tokens_used: float
    avg_tasks_per_mission: float
    top_agents: List[TopAgentStats]
    common_failure_reasons: Dict[str, int]
    period: str


class AgentMissionHistoryResponse(BaseModel):
    agent_id: int
    agent_name: str
    tasks_completed: int
    avg_tokens_per_task: float
    failure_rate: float
    recent_missions: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_to_response(run: OrchestrationRun) -> dict:
    """Convert an OrchestrationRun ORM object to a dict matching MissionResponse."""
    return {
        "id": str(run.id),
        "workspace_id": str(run.workspace_id),
        "goal": run.goal,
        "state": run.state,
        "state_type": run.state_type,
        "plan": run.plan,
        "config": run.config,
        "output_summary": run.output_summary,
        "token_budget_estimate": run.token_budget_estimate,
        "tokens_used": run.tokens_used or 0,
        "max_retries": run.max_retries or 3,
        "created_by": run.created_by,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


def _task_to_response(task: OrchestrationTask) -> dict:
    """Convert an OrchestrationTask ORM object to a TaskResponse dict."""
    return {
        "id": str(task.id),
        "title": task.title,
        "description": task.description,
        "task_type": task.task_type,
        "sequence_number": task.sequence_number,
        "agent_role": task.agent_role,
        "state": task.state,
        "state_type": task.state_type,
        "assigned_agent_id": task.assigned_agent_id,
        "attempt_number": task.attempt_number or 0,
        "tokens_used": task.tokens_used or 0,
        "failure_reason_code": task.failure_reason_code,
        "failure_detail": task.failure_detail,
        "output_excerpt": (task.output[:500] if task.output else None),
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "created_at": task.created_at,
    }


def _event_to_response(event: OrchestrationEvent) -> dict:
    """Convert an OrchestrationEvent ORM object to an EventResponse dict."""
    return {
        "id": str(event.id),
        "event_type": event.event_type,
        "actor_type": event.actor_type,
        "actor_id": event.actor_id,
        "old_state": event.old_state,
        "new_state": event.new_state,
        "task_id": str(event.task_id) if event.task_id else None,
        "created_at": event.created_at,
    }


def _get_run_for_workspace(
    db: Session,
    run_id: UUID,
    workspace_id: UUID,
) -> OrchestrationRun:
    """Load an OrchestrationRun by ID scoped to workspace. Raises 404 if not found."""
    run = (
        db.query(OrchestrationRun)
        .filter(
            and_(
                OrchestrationRun.id == run_id,
                OrchestrationRun.workspace_id == workspace_id,
            )
        )
        .first()
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Mission not found")
    return run


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", status_code=201)
async def create_mission(
    body: MissionCreateRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create a new mission from a natural-language goal."""
    coordinator = get_coordinator_service()
    try:
        run = await coordinator.create_mission(
            db=db,
            workspace_id=ctx.workspace_id,
            goal=body.goal,
            created_by=ctx.user.id or "unknown",
            config=body.config,
        )
        db.commit()
        return _run_to_response(run)

    except PlanValidationError as exc:
        db.rollback()
        raise HTTPException(
            status_code=422,
            detail=f"Plan validation failed: {exc}",
        )
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        logger.error("Failed to create mission: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("")
async def list_missions(
    state: Optional[str] = Query(None, description="Filter by state (e.g., 'running', 'completed')"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List missions for the current workspace, paginated and optionally filtered by state."""
    try:
        query = db.query(OrchestrationRun).filter(
            OrchestrationRun.workspace_id == ctx.workspace_id,
        )

        if state:
            # Validate state value
            try:
                RunState(state)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid state: '{state}'. Valid states: {[s.value for s in RunState]}",
                )
            query = query.filter(OrchestrationRun.state == state)

        total = query.count()

        runs = (
            query
            .order_by(OrchestrationRun.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return {
            "missions": [_run_to_response(r) for r in runs],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list missions: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats")
async def get_mission_stats(
    period: str = Query("30d", pattern="^(7d|30d|90d|all)$", description="Stats period"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Aggregate mission statistics for the current workspace."""
    try:
        # Determine date cutoff
        period_days = {"7d": 7, "30d": 30, "90d": 90, "all": None}
        days = period_days.get(period)

        base_filter = [OrchestrationRun.workspace_id == ctx.workspace_id]
        if days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            base_filter.append(OrchestrationRun.created_at >= cutoff)

        # Total missions
        total_missions = (
            db.query(func.count(OrchestrationRun.id))
            .filter(*base_filter)
            .scalar()
        ) or 0

        # Success rate
        completed_count = (
            db.query(func.count(OrchestrationRun.id))
            .filter(*base_filter, OrchestrationRun.state == RunState.COMPLETED.value)
            .scalar()
        ) or 0

        terminal_count = (
            db.query(func.count(OrchestrationRun.id))
            .filter(
                *base_filter,
                OrchestrationRun.state.in_(
                    [s.value for s in TERMINAL_RUN_STATES]
                ),
            )
            .scalar()
        ) or 0

        success_rate = round(completed_count / terminal_count, 4) if terminal_count > 0 else 0.0

        # Average duration (completed missions only)
        avg_duration_row = (
            db.query(
                func.avg(
                    func.extract("epoch", OrchestrationRun.completed_at)
                    - func.extract("epoch", OrchestrationRun.started_at)
                )
            )
            .filter(
                *base_filter,
                OrchestrationRun.state == RunState.COMPLETED.value,
                OrchestrationRun.started_at.isnot(None),
                OrchestrationRun.completed_at.isnot(None),
            )
            .scalar()
        )
        avg_duration_minutes = round(avg_duration_row / 60.0, 2) if avg_duration_row else None

        # Average tokens used
        avg_tokens = (
            db.query(func.avg(OrchestrationRun.tokens_used))
            .filter(*base_filter)
            .scalar()
        )
        avg_tokens_used = round(float(avg_tokens), 2) if avg_tokens else 0.0

        # Average tasks per mission
        task_counts = (
            db.query(func.count(OrchestrationTask.id))
            .join(OrchestrationRun, OrchestrationTask.run_id == OrchestrationRun.id)
            .filter(*base_filter)
            .scalar()
        ) or 0
        avg_tasks_per_mission = round(task_counts / total_missions, 2) if total_missions > 0 else 0.0

        # Top agents (by tasks completed in verified state)
        task_filter = [OrchestrationRun.workspace_id == ctx.workspace_id]
        if days is not None:
            task_filter.append(OrchestrationTask.created_at >= cutoff)

        top_agent_rows = (
            db.query(
                OrchestrationTask.assigned_agent_id,
                Agent.name,
                func.count(OrchestrationTask.id).label("tasks_completed"),
                func.avg(OrchestrationTask.tokens_used).label("avg_tokens"),
            )
            .join(OrchestrationRun, OrchestrationTask.run_id == OrchestrationRun.id)
            .join(Agent, OrchestrationTask.assigned_agent_id == Agent.id)
            .filter(
                *task_filter,
                OrchestrationTask.state == TaskState.VERIFIED.value,
                OrchestrationTask.assigned_agent_id.isnot(None),
            )
            .group_by(OrchestrationTask.assigned_agent_id, Agent.name)
            .order_by(func.count(OrchestrationTask.id).desc())
            .limit(5)
            .all()
        )

        top_agents = [
            TopAgentStats(
                agent_id=row[0],
                agent_name=row[1],
                tasks_completed=row[2],
                avg_tokens_per_task=round(float(row[3] or 0), 2),
            )
            for row in top_agent_rows
        ]

        # Common failure reasons
        failure_rows = (
            db.query(
                OrchestrationTask.failure_reason_code,
                func.count(OrchestrationTask.id),
            )
            .join(OrchestrationRun, OrchestrationTask.run_id == OrchestrationRun.id)
            .filter(
                *task_filter,
                OrchestrationTask.state == TaskState.FAILED.value,
                OrchestrationTask.failure_reason_code.isnot(None),
            )
            .group_by(OrchestrationTask.failure_reason_code)
            .order_by(func.count(OrchestrationTask.id).desc())
            .limit(10)
            .all()
        )
        common_failure_reasons = {row[0]: row[1] for row in failure_rows}

        return MissionStatsResponse(
            total_missions=total_missions,
            success_rate=success_rate,
            avg_duration_minutes=avg_duration_minutes,
            avg_tokens_used=avg_tokens_used,
            avg_tasks_per_mission=avg_tasks_per_mission,
            top_agents=top_agents,
            common_failure_reasons=common_failure_reasons,
            period=period,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get mission stats: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{mission_id}")
async def get_mission(
    mission_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get full mission detail: run + tasks + recent events."""
    try:
        run = _get_run_for_workspace(db, mission_id, ctx.workspace_id)

        tasks = (
            db.query(OrchestrationTask)
            .filter(OrchestrationTask.run_id == run.id)
            .order_by(OrchestrationTask.sequence_number)
            .all()
        )

        events = (
            db.query(OrchestrationEvent)
            .filter(OrchestrationEvent.run_id == run.id)
            .order_by(OrchestrationEvent.created_at.desc())
            .limit(50)
            .all()
        )

        result = _run_to_response(run)
        result["tasks"] = [_task_to_response(t) for t in tasks]
        result["recent_events"] = [_event_to_response(e) for e in events]
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get mission %s: %s", mission_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{mission_id}/events")
async def get_mission_events(
    mission_id: UUID,
    event_type: Optional[str] = Query(None, description="Filter by event_type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Paginated events for a mission run, optionally filtered by event_type."""
    try:
        # Validate mission belongs to workspace
        _get_run_for_workspace(db, mission_id, ctx.workspace_id)

        query = db.query(OrchestrationEvent).filter(
            OrchestrationEvent.run_id == mission_id,
        )

        if event_type:
            # Validate event_type value
            valid_types = {e.value for e in EventType}
            if event_type not in valid_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid event_type: '{event_type}'. Valid types: {sorted(valid_types)}",
                )
            query = query.filter(OrchestrationEvent.event_type == event_type)

        total = query.count()

        events = (
            query
            .order_by(OrchestrationEvent.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return PaginatedEventsResponse(
            events=[_event_to_response(e) for e in events],
            total=total,
            limit=limit,
            offset=offset,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get events for mission %s: %s", mission_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{mission_id}/cost")
async def get_mission_cost(
    mission_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Token usage breakdown and estimated cost for a mission."""
    try:
        run = _get_run_for_workspace(db, mission_id, ctx.workspace_id)

        tasks = (
            db.query(OrchestrationTask)
            .filter(OrchestrationTask.run_id == run.id)
            .order_by(OrchestrationTask.sequence_number)
            .all()
        )

        task_breakdowns = [
            TaskCostBreakdown(
                task_id=str(t.id),
                title=t.title,
                tokens_used=t.tokens_used or 0,
            )
            for t in tasks
        ]

        total_tokens = sum(tb.tokens_used for tb in task_breakdowns)
        cost_rate = config.COORDINATOR_COST_PER_1K_TOKENS
        estimated_cost = round((total_tokens / 1000.0) * cost_rate, 6)

        return MissionCostResponse(
            mission_id=str(run.id),
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            cost_per_1k_tokens=cost_rate,
            tasks=task_breakdowns,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get cost for mission %s: %s", mission_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{mission_id}/approve")
async def approve_plan(
    mission_id: UUID,
    body: MissionApproveRequest = MissionApproveRequest(),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Approve a mission plan and start execution."""
    try:
        run = _get_run_for_workspace(db, mission_id, ctx.workspace_id)

        if RunState(run.state) != RunState.AWAITING_APPROVAL:
            raise HTTPException(
                status_code=400,
                detail=f"Mission is in '{run.state}' state, expected 'awaiting_approval'",
            )

        coordinator = get_coordinator_service()
        run = coordinator.approve_plan(
            db=db,
            run_id=run.id,
            actor_id=ctx.user.id or "unknown",
            modifications=body.modifications,
        )
        db.commit()
        return _run_to_response(run)

    except HTTPException:
        raise
    except (ConflictError, InvalidTransitionError) as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        db.rollback()
        logger.error("Failed to approve mission %s: %s", mission_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{mission_id}/reject")
async def reject_plan(
    mission_id: UUID,
    body: MissionRejectRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Reject a mission plan."""
    try:
        run = _get_run_for_workspace(db, mission_id, ctx.workspace_id)

        if RunState(run.state) != RunState.AWAITING_APPROVAL:
            raise HTTPException(
                status_code=400,
                detail=f"Mission is in '{run.state}' state, expected 'awaiting_approval'",
            )

        coordinator = get_coordinator_service()
        run = coordinator.reject_plan(
            db=db,
            run_id=run.id,
            actor_id=ctx.user.id or "unknown",
            reason=body.reason,
        )
        db.commit()
        return _run_to_response(run)

    except HTTPException:
        raise
    except (ConflictError, InvalidTransitionError) as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        db.rollback()
        logger.error("Failed to reject mission %s: %s", mission_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{mission_id}/review")
async def review_mission(
    mission_id: UUID,
    body: MissionReviewRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Submit human review: accept or reject with per-task feedback."""
    try:
        run = _get_run_for_workspace(db, mission_id, ctx.workspace_id)

        if RunState(run.state) != RunState.AWAITING_HUMAN:
            raise HTTPException(
                status_code=400,
                detail=f"Mission is in '{run.state}' state, expected 'awaiting_human'",
            )

        coordinator = get_coordinator_service()
        run = coordinator.review_mission(
            db=db,
            run_id=run.id,
            actor_id=ctx.user.id or "unknown",
            verdict=body.verdict,
            task_feedback=body.task_feedback,
        )
        db.commit()
        return _run_to_response(run)

    except HTTPException:
        raise
    except (ConflictError, InvalidTransitionError) as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        db.rollback()
        logger.error("Failed to review mission %s: %s", mission_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{mission_id}/pause")
async def pause_mission(
    mission_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Pause a running mission. Running tasks continue but no new dispatches."""
    try:
        run = _get_run_for_workspace(db, mission_id, ctx.workspace_id)

        if RunState(run.state) != RunState.RUNNING:
            raise HTTPException(
                status_code=400,
                detail=f"Mission is in '{run.state}' state, expected 'running'",
            )

        coordinator = get_coordinator_service()
        run = coordinator.pause_mission(
            db=db,
            run_id=run.id,
            actor_id=ctx.user.id or "unknown",
        )
        db.commit()
        return _run_to_response(run)

    except HTTPException:
        raise
    except (ConflictError, InvalidTransitionError) as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        db.rollback()
        logger.error("Failed to pause mission %s: %s", mission_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{mission_id}/resume")
async def resume_mission(
    mission_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Resume a paused mission."""
    try:
        run = _get_run_for_workspace(db, mission_id, ctx.workspace_id)

        if RunState(run.state) != RunState.PAUSED:
            raise HTTPException(
                status_code=400,
                detail=f"Mission is in '{run.state}' state, expected 'paused'",
            )

        coordinator = get_coordinator_service()
        run = coordinator.resume_mission(
            db=db,
            run_id=run.id,
            actor_id=ctx.user.id or "unknown",
        )
        db.commit()
        return _run_to_response(run)

    except HTTPException:
        raise
    except (ConflictError, InvalidTransitionError) as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        db.rollback()
        logger.error("Failed to resume mission %s: %s", mission_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{mission_id}/cancel")
async def cancel_mission(
    mission_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Cancel a mission. Running tasks continue to completion; no new dispatches."""
    try:
        run = _get_run_for_workspace(db, mission_id, ctx.workspace_id)

        # Cancel allowed from any non-terminal state
        if RunState(run.state) in TERMINAL_RUN_STATES:
            raise HTTPException(
                status_code=400,
                detail=f"Mission is already in terminal state '{run.state}'",
            )

        coordinator = get_coordinator_service()
        run = coordinator.cancel_mission(
            db=db,
            run_id=run.id,
            actor_id=ctx.user.id or "unknown",
        )
        db.commit()
        return _run_to_response(run)

    except HTTPException:
        raise
    except (ConflictError, InvalidTransitionError) as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        db.rollback()
        logger.error("Failed to cancel mission %s: %s", mission_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Agent telemetry router (PRD-82B US-004)
# ---------------------------------------------------------------------------

agent_telemetry_router = APIRouter(prefix="/api/agents", tags=["agent-telemetry"])


@agent_telemetry_router.get("/{agent_id}/mission-history")
async def get_agent_mission_history(
    agent_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Agent mission performance: tasks completed, failure rate, recent missions."""
    try:
        # Verify agent exists and belongs to workspace
        agent = (
            db.query(Agent)
            .filter(
                Agent.id == agent_id,
                Agent.workspace_id == ctx.workspace_id,
            )
            .first()
        )
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        # All tasks assigned to this agent in this workspace
        base_query = (
            db.query(OrchestrationTask)
            .join(OrchestrationRun, OrchestrationTask.run_id == OrchestrationRun.id)
            .filter(
                OrchestrationRun.workspace_id == ctx.workspace_id,
                OrchestrationTask.assigned_agent_id == agent_id,
            )
        )

        # Tasks completed (verified state)
        tasks_completed = (
            base_query
            .filter(OrchestrationTask.state == TaskState.VERIFIED.value)
            .count()
        )

        # Average tokens per task (verified tasks only)
        avg_tokens = (
            db.query(func.avg(OrchestrationTask.tokens_used))
            .join(OrchestrationRun, OrchestrationTask.run_id == OrchestrationRun.id)
            .filter(
                OrchestrationRun.workspace_id == ctx.workspace_id,
                OrchestrationTask.assigned_agent_id == agent_id,
                OrchestrationTask.state == TaskState.VERIFIED.value,
            )
            .scalar()
        )
        avg_tokens_per_task = round(float(avg_tokens), 2) if avg_tokens else 0.0

        # Failure rate
        total_terminal = (
            base_query
            .filter(
                OrchestrationTask.state.in_(
                    [TaskState.VERIFIED.value, TaskState.FAILED.value]
                )
            )
            .count()
        )
        failed_count = (
            base_query
            .filter(OrchestrationTask.state == TaskState.FAILED.value)
            .count()
        )
        failure_rate = round(failed_count / total_terminal, 4) if total_terminal > 0 else 0.0

        # Recent missions (last 10 distinct runs)
        recent_tasks = (
            db.query(
                OrchestrationRun.id,
                OrchestrationRun.goal,
                OrchestrationRun.state,
                OrchestrationTask.title,
                OrchestrationTask.state.label("task_state"),
                OrchestrationTask.tokens_used,
                OrchestrationTask.completed_at,
            )
            .join(OrchestrationRun, OrchestrationTask.run_id == OrchestrationRun.id)
            .filter(
                OrchestrationRun.workspace_id == ctx.workspace_id,
                OrchestrationTask.assigned_agent_id == agent_id,
            )
            .order_by(OrchestrationTask.created_at.desc())
            .limit(10)
            .all()
        )

        recent_missions = [
            {
                "mission_id": str(row[0]),
                "goal": (row[1][:100] if row[1] else None),
                "mission_state": row[2],
                "task_title": row[3],
                "task_state": row[4],
                "tokens_used": row[5] or 0,
                "completed_at": row[6].isoformat() if row[6] else None,
            }
            for row in recent_tasks
        ]

        return AgentMissionHistoryResponse(
            agent_id=agent_id,
            agent_name=agent.name,
            tasks_completed=tasks_completed,
            avg_tokens_per_task=avg_tokens_per_task,
            failure_rate=failure_rate,
            recent_missions=recent_missions,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get mission history for agent %s: %s", agent_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
