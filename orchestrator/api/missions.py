"""
Missions REST API — PRD-82A Sequential Mission Coordinator
==========================================================

CRUD + lifecycle endpoints for missions. API uses "mission" terminology;
DB/backend uses "orchestration" (PRD-82A Section 10).

9 endpoints:
  POST   /api/missions           — create mission
  GET    /api/missions           — list missions (paginated, filterable)
  GET    /api/missions/{id}      — get mission detail
  POST   /api/missions/{id}/approve  — approve plan
  POST   /api/missions/{id}/reject   — reject plan
  POST   /api/missions/{id}/review   — submit human review
  POST   /api/missions/{id}/pause    — pause mission
  POST   /api/missions/{id}/resume   — resume mission
  POST   /api/missions/{id}/cancel   — cancel mission

Source: PRD-82A Section 12, Phase 5 (US-021)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, validator
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.orchestration import (
    OrchestrationEvent,
    OrchestrationRun,
    OrchestrationTask,
)
from core.models.orchestration_enums import (
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
