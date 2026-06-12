"""
Missions REST API — PRD-82A/82B Sequential Mission Coordinator
===============================================================

CRUD + lifecycle + telemetry endpoints for missions. API uses "mission"
terminology; DB/backend uses "orchestration" (PRD-82A Section 10).

15 endpoints:
  POST   /api/missions           — create mission
  GET    /api/missions           — list missions (paginated, filterable)
  GET    /api/missions/stats     — aggregate mission stats (PRD-82B US-004)
  GET    /api/missions/{id}      — get mission detail
  GET    /api/missions/{id}/events  — paginated events (PRD-82B US-004)
  GET    /api/missions/{id}/cost    — token/cost breakdown (PRD-82B US-004)
  POST   /api/missions/{id}/approve  — approve plan
  POST   /api/missions/{id}/reject   — reject plan
  POST   /api/missions/{id}/review   — submit human review
  POST   /api/missions/{id}/replan   — replan failed mission (PRD-82B US-005)
  POST   /api/missions/{id}/save-as-routine  — save completed mission as routine (PRD-82B US-008)
  POST   /api/missions/{id}/pause    — pause mission
  POST   /api/missions/{id}/resume   — resume mission
  POST   /api/missions/{id}/cancel   — cancel mission
  DELETE /api/missions/{id}           — delete terminal mission

Agent telemetry:
  GET    /api/agents/{agent_id}/mission-history  — agent mission perf (PRD-82B US-004)

Source: PRD-82A Section 12, PRD-82B US-004/US-005/US-008
"""

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field, validator
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import Agent, BoardTask, WorkflowTemplate
from core.models.orchestration import (
    OrchestrationArchive,
    OrchestrationEvent,
    OrchestrationRun,
    OrchestrationTask,
    OrchestrationTaskDependency,
)
from core.models.orchestration_enums import (
    EventType,
    RunState,
    TaskState,
    TERMINAL_RUN_STATES,
)
from modules.coordination.planner import PlanValidationError
import boto3
from botocore.config import Config as BotoConfig

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
    goal: str = Field(..., min_length=1, max_length=10000, description="Natural-language goal")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional mission config overrides")
    template_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Template ID hint — bypass LLM matching and use this template directly",
    )
    plan_only: bool = Field(
        False,
        description="PRD-163 S2: plan only — produce the plan and await approval, never auto-execute",
    )


ALLOWED_MODIFICATION_KEYS = {"task_overrides", "notes", "agent_overrides"}


class MissionApproveRequest(BaseModel):
    modifications: Optional[Dict[str, Any]] = Field(None, description="Optional plan modifications")
    max_concurrent_override: Optional[int] = Field(
        None, ge=1, le=10, description="Override max_concurrent for this mission"
    )
    token_budget_override: Optional[int] = Field(
        None, ge=1000, description="Override token budget estimate for this mission"
    )
    skip_verification: Optional[bool] = Field(
        None, description="Skip task verification (for benchmarks/testing)",
    )

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
    feedback: Optional[str] = Field(
        None,
        max_length=5000,
        description="General rejection feedback (when rejecting without flagging specific tasks).",
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
    estimated_tokens: int = 4000
    complexity: Optional[str] = None
    parallel_group: Optional[str] = None
    failure_reason_code: Optional[str] = None
    failure_detail: Optional[str] = None
    output_excerpt: Optional[str] = None
    output: Optional[str] = None
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
    max_concurrent: int = 1
    replan_count: int = 0
    complexity_tier: Optional[str] = None
    parallel_groups: List[str] = []
    has_synthesis_tasks: bool = False
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
    # Extract parallel groups and synthesis info from tasks in the plan JSONB
    plan = run.plan or {}
    plan_tasks = plan.get("tasks", [])
    parallel_groups = sorted({
        t.get("parallel_group")
        for t in plan_tasks
        if t.get("parallel_group")
    })
    has_synthesis_tasks = any(
        t.get("task_type") == "synthesis" for t in plan_tasks
    )

    # Derive complexity tier from max_concurrent
    max_concurrent = run.max_concurrent or 1
    if max_concurrent >= 3:
        complexity_tier = "complex"
    elif max_concurrent >= 2:
        complexity_tier = "moderate"
    else:
        complexity_tier = "simple"

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
        "max_concurrent": max_concurrent,
        "replan_count": run.replan_count or 0,
        "complexity_tier": complexity_tier,
        "parallel_groups": parallel_groups,
        "has_synthesis_tasks": has_synthesis_tasks,
        "created_by": run.created_by,
        "stop_reason": getattr(run, "stop_reason", None),
        "stop_detail": getattr(run, "stop_detail", None),
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
        "estimated_tokens": getattr(task, "estimated_tokens", None) or 4000,
        "complexity": getattr(task, "complexity", None),
        "parallel_group": getattr(task, "parallel_group", None),
        "failure_reason_code": task.failure_reason_code,
        "failure_detail": task.failure_detail,
        "output_excerpt": (task.output[:500] if task.output else None),
        "output": task.output,
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

    # Merge template_id into config so it flows to planner
    mission_config = dict(body.config) if body.config else {}
    if body.template_id:
        mission_config["template_id"] = body.template_id
    if body.plan_only:
        mission_config["plan_only"] = True

    try:
        run = await coordinator.create_mission(
            db=db,
            workspace_id=ctx.workspace_id,
            goal=body.goal,
            created_by=ctx.user.id or "unknown",
            config=mission_config or None,
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


class PlanImportRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=10000, description="Mission goal")
    plan: Dict[str, Any] = Field(..., description="Pre-built plan: {tasks:[...], dependencies:[...]}")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional mission config overrides")


@router.post("/import-plan", status_code=201)
async def import_mission_plan(
    body: PlanImportRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """PRD-163 S2: create a mission from a pre-built (possibly chat-edited) plan and
    execute it verbatim — the planner is NOT re-run, so the executed DAG matches the
    given plan exactly (Q54). The mission lands in awaiting_approval."""
    coordinator = get_coordinator_service()
    try:
        run = coordinator.import_plan(
            db=db,
            workspace_id=ctx.workspace_id,
            goal=body.goal,
            plan=body.plan,
            created_by=ctx.user.id or "unknown",
            config=body.config,
        )
        db.commit()
        return _run_to_response(run)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        logger.error("Failed to import plan: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


class ApprovalPolicyRequest(BaseModel):
    policy: Optional[str] = Field(None, description="always_ask | auto_below_budget | full_auto")
    approval_dollar_ceiling: Optional[float] = Field(None, ge=0)
    auto_proceed_after_seconds: Optional[int] = Field(None, ge=1)


@router.get("/approval-policy")
async def get_mission_approval_policy(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """PRD-163 S3: the workspace's mission approval policy."""
    from core.services.approval_policy import load_approval_policy

    return load_approval_policy(db, ctx.workspace_id)


@router.put("/approval-policy")
async def set_mission_approval_policy(
    body: ApprovalPolicyRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """PRD-163 S3: update the workspace's mission approval policy."""
    from core.services.approval_policy import set_approval_policy

    try:
        result = set_approval_policy(
            db,
            ctx.workspace_id,
            policy=body.policy,
            approval_dollar_ceiling=body.approval_dollar_ceiling,
            auto_proceed_after_seconds=body.auto_proceed_after_seconds,
        )
        db.commit()
        return result
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))


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
        avg_duration_minutes = round(float(avg_duration_row) / 60.0, 2) if avg_duration_row else None

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


@router.get("/archive")
async def list_archived_missions(
    search: Optional[str] = Query(None, description="Search archived missions by goal text"),
    state: Optional[str] = Query(None, description="Filter by terminal state"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List archived missions for the current workspace, with optional search and state filter."""
    try:
        query = db.query(OrchestrationArchive).filter(
            OrchestrationArchive.workspace_id == ctx.workspace_id,
        )

        if state:
            try:
                RunState(state)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid state: '{state}'. Valid terminal states: completed, failed, cancelled",
                )
            query = query.filter(OrchestrationArchive.state == state)

        if search:
            query = query.filter(
                OrchestrationArchive.goal.ilike(f"%{search}%"),
            )

        total = query.count()

        archives = (
            query
            .order_by(OrchestrationArchive.archived_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return {
            "archives": [
                {
                    "id": str(a.id),
                    "original_run_id": str(a.original_run_id),
                    "goal": a.goal,
                    "state": a.state,
                    "created_by": a.created_by,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                    "completed_at": a.completed_at.isoformat() if a.completed_at else None,
                    "archived_at": a.archived_at.isoformat() if a.archived_at else None,
                    "task_count": len(a.archive_data.get("tasks", [])) if a.archive_data else 0,
                    "tokens_used": (
                        a.archive_data.get("run", {}).get("tokens_used", 0)
                        if a.archive_data
                        else 0
                    ),
                }
                for a in archives
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list archived missions: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/archive/{archive_id}")
async def get_archived_mission(
    archive_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get full archived mission detail including the complete snapshot."""
    try:
        archive = (
            db.query(OrchestrationArchive)
            .filter(
                and_(
                    OrchestrationArchive.id == archive_id,
                    OrchestrationArchive.workspace_id == ctx.workspace_id,
                )
            )
            .first()
        )
        if archive is None:
            raise HTTPException(status_code=404, detail="Archived mission not found")

        return {
            "id": str(archive.id),
            "original_run_id": str(archive.original_run_id),
            "goal": archive.goal,
            "state": archive.state,
            "created_by": archive.created_by,
            "created_at": archive.created_at.isoformat() if archive.created_at else None,
            "completed_at": archive.completed_at.isoformat() if archive.completed_at else None,
            "archived_at": archive.archived_at.isoformat() if archive.archived_at else None,
            "archive_data": archive.archive_data,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get archived mission %s: %s", archive_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

_UPLOAD_MAX_BYTES = 20 * 1024 * 1024  # 20 MB
_UPLOAD_ALLOWED_EXTS = {".pdf", ".md", ".txt", ".doc", ".docx", ".json", ".csv", ".xlsx"}


@router.post("/upload")
async def upload_mission_file(
    file: UploadFile = File(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Upload a file attachment for use in a mission.

    Max 20 MB. Allowed types: pdf, md, txt, doc, docx, json, csv, xlsx.
    Runs through DocumentManager pipeline (S3 + chunking + embedding for RAG).
    Returns document_id and file metadata.
    """
    from pathlib import Path
    from api.documents import get_document_manager

    # --- validate extension ---------------------------------------------------
    filename = file.filename or "upload"
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in _UPLOAD_ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Accepted: {sorted(_UPLOAD_ALLOWED_EXTS)}",
        )

    # --- read & validate size -------------------------------------------------
    content = await file.read()
    if len(content) > _UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content)} bytes). Max {_UPLOAD_MAX_BYTES} bytes (20 MB).",
        )

    # --- write to temp file for DocumentManager -------------------------------
    upload_dir = Path("/tmp/automatos_mission_uploads")
    upload_dir.mkdir(exist_ok=True)
    temp_filename = f"{uuid4().hex}{ext}"
    temp_path = upload_dir / temp_filename

    try:
        with open(temp_path, "wb") as f:
            f.write(content)

        doc_manager = get_document_manager(str(ctx.workspace_id))
        document_id = await doc_manager.upload_document(
            file_path=str(temp_path),
            filename=filename,
            tags=["mission-attachment"],
            description="Uploaded as mission reference document",
            created_by=ctx.user.id if ctx.user else "system",
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "size": len(content),
            "content_type": file.content_type or "application/octet-stream",
        }
    except Exception as exc:
        logger.error("Mission upload failed for %s: %s", filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="File upload failed")
    finally:
        if temp_path.exists():
            temp_path.unlink()


# ---------------------------------------------------------------------------
# Mission detail & telemetry
# ---------------------------------------------------------------------------


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

        # PRD-123 Pattern #5: Extract permission denial events
        permission_denials = [
            {
                "agent_name": (e.payload or {}).get("agent_name"),
                "tool_name": (e.payload or {}).get("tool_name"),
                "reason": (e.payload or {}).get("reason"),
                "denied_at": e.created_at,
            }
            for e in events
            if e.event_type == "permission_denied"
        ]

        result = _run_to_response(run)
        result["tasks"] = [_task_to_response(t) for t in tasks]
        result["recent_events"] = [_event_to_response(e) for e in events]
        result["permission_denials"] = permission_denials
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


@router.get("/{mission_id}/field")
async def get_mission_field(
    mission_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get the live state of the mission's shared semantic field (PRD-108).

    Returns all patterns with decayed strengths, stability metrics,
    and instrumentation data for the field visualizer.

    `status` values:
      - `not_created` — mission has no field_id (mission is queued or
        was created before PRD-108 shipped)
      - `missing` — field_id present but the underlying Qdrant
        collection is gone (stale id; coordinator will recreate on
        next tick if mission is still running — see PR #312)
      - `empty` — collection exists but no patterns injected yet
      - `active` — collection exists with patterns
      - `unavailable` — shared context backend is down (Qdrant
        unreachable, etc)
    """
    try:
        run = _get_run_for_workspace(db, mission_id, ctx.workspace_id)
        field_id = (run.config or {}).get("field_id")

        if not field_id:
            return {
                "status": "not_created",
                "field_id": None,
                "backend": None,
                "patterns": [],
                "stability": {"stability": 0.0, "pattern_count": 0},
                "metrics": None,
            }

        from modules.context.factory import get_shared_context

        field = get_shared_context()
        if not field:
            return {
                "status": "unavailable",
                "field_id": field_id,
                "backend": None,
                "patterns": [],
                "stability": {"stability": 0.0, "pattern_count": 0},
                "metrics": None,
            }

        # Detect stale field_id (collection destroyed). PR #312 auto-heals
        # on next coordinator tick; the panel surfaces this state in the UI.
        collection_missing = False
        if hasattr(field, "context_exists"):
            try:
                collection_missing = not await field.context_exists(field_id)
            except Exception:
                pass

        # Get patterns and stability from the inner (unwrapped) backend
        inner = field._inner
        patterns = []
        stability = {"stability": 0.0, "pattern_count": 0}

        if hasattr(inner, "get_patterns"):
            patterns = await inner.get_patterns(field_id)
        if hasattr(inner, "measure_stability"):
            stability = await inner.measure_stability(field_id)

        # measure_stability marks {"missing": True} on collection 404.
        if stability.get("missing"):
            collection_missing = True

        if collection_missing:
            status = "missing"
        elif patterns:
            status = "active"
        else:
            status = "empty"

        # Get instrumentation metrics if available
        metrics_data = None
        metrics = field.get_metrics(field_id)
        if metrics:
            metrics_data = metrics.to_dict()

        return {
            "status": status,
            "field_id": field_id,
            "backend": field._backend_name,
            "patterns": patterns,
            "stability": stability,
            "metrics": metrics_data,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get field for mission %s: %s", mission_id, exc, exc_info=True)
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

        # Apply overrides before approval
        if body.max_concurrent_override is not None:
            run.max_concurrent = body.max_concurrent_override
        if body.token_budget_override is not None:
            run.token_budget_estimate = body.token_budget_override
        if body.skip_verification is not None:
            run.config = {**(run.config or {}), "skip_verification": body.skip_verification}

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
            feedback=body.feedback,
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


class MissionReplanRequest(BaseModel):
    notes: Optional[str] = Field(
        None,
        max_length=5000,
        description="Optional user guidance for the replanner",
    )


@router.post("/{mission_id}/replan")
async def replan_mission(
    mission_id: UUID,
    body: MissionReplanRequest = MissionReplanRequest(),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Replan a failed mission — generate replacement tasks for the failed subtree."""
    try:
        run = _get_run_for_workspace(db, mission_id, ctx.workspace_id)

        if RunState(run.state) != RunState.FAILED:
            raise HTTPException(
                status_code=400,
                detail=f"Mission is in '{run.state}' state, expected 'failed'",
            )

        max_replans = config.COORDINATOR_MAX_REPLANS
        current_replans = run.replan_count or 0
        if current_replans >= max_replans:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Mission has been replanned {current_replans} times, "
                    f"maximum is {max_replans}"
                ),
            )

        coordinator = get_coordinator_service()
        run = await coordinator.replan_mission(
            db=db,
            run_id=run.id,
            actor_id=ctx.user.id or "unknown",
            notes=body.notes,
        )
        db.commit()
        return _run_to_response(run)

    except HTTPException:
        raise
    except PlanValidationError as exc:
        db.rollback()
        raise HTTPException(
            status_code=422,
            detail=f"Replan validation failed: {exc}",
        )
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except (ConflictError, InvalidTransitionError) as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        db.rollback()
        logger.error("Failed to replan mission %s: %s", mission_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Save-as-routine (PRD-82B US-008)
# ---------------------------------------------------------------------------


class SaveAsRoutineRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Routine name")
    description: Optional[str] = Field(None, max_length=2000, description="Optional routine description")
    tags: Optional[List[str]] = Field(default_factory=list, description="Optional tags for categorisation")

    @validator("tags")
    def validate_tags(cls, v: Optional[List[str]]) -> List[str]:
        if v is None:
            return []
        if len(v) > 20:
            raise ValueError("Too many tags (max 20)")
        return [t.strip()[:50] for t in v if t.strip()]


class SaveAsRoutineResponse(BaseModel):
    template_id: str
    name: str
    task_count: int


def _slugify(text: str, max_length: int = 80) -> str:
    """Convert text to a URL-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "-", slug).strip("-")
    return slug[:max_length]


def _extract_routine_template(
    tasks: List[OrchestrationTask],
    dependencies: List[OrchestrationTaskDependency],
    goal: str,
) -> Dict[str, Any]:
    """
    Extract a reusable template definition from a completed mission's tasks.

    Templatizes task descriptions by replacing goal-specific content with {goal}
    placeholder, and preserves the dependency graph structure.
    """
    # Build dependency map: task_id -> list of prerequisite task_ids
    dep_map: Dict[str, List[str]] = {}
    for dep in dependencies:
        task_id_str = str(dep.task_id)
        depends_on_str = str(dep.depends_on_task_id)
        dep_map.setdefault(task_id_str, []).append(depends_on_str)

    # Map original task UUIDs to sequential temp_ids for the template
    task_id_to_temp: Dict[str, str] = {}
    sorted_tasks = sorted(tasks, key=lambda t: t.sequence_number)
    for idx, task in enumerate(sorted_tasks, 1):
        task_id_to_temp[str(task.id)] = f"task_{idx}"

    task_templates = []
    for task in sorted_tasks:
        task_id_str = str(task.id)
        temp_id = task_id_to_temp[task_id_str]

        # Templatize: replace goal text in title/description with {goal}
        title = (task.title or "").replace(goal, "{goal}")
        description = (task.description or "").replace(goal, "{goal}")

        # Resolve dependencies to temp_ids
        depends_on = [
            task_id_to_temp[dep_id]
            for dep_id in dep_map.get(task_id_str, [])
            if dep_id in task_id_to_temp
        ]

        task_templates.append({
            "temp_id": temp_id,
            "sequence": task.sequence_number,
            "agent_role": task.agent_role or "researcher",
            "title_pattern": title,
            "description_pattern": description,
            "required_tools": [],
            "verification_criteria": task.verification_criteria or [],
            "dependencies": depends_on,
        })

    return {
        "id": None,  # Will be set to template_id after slug generation
        "task_count": len(task_templates),
        "output_format": "markdown",
        "task_templates": task_templates,
    }


@router.post("/{mission_id}/save-as-routine", status_code=201)
async def save_as_routine(
    mission_id: UUID,
    body: SaveAsRoutineRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Save a completed mission as a reusable routine template."""
    try:
        run = _get_run_for_workspace(db, mission_id, ctx.workspace_id)

        if RunState(run.state) != RunState.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Mission is in '{run.state}' state, expected 'completed'",
            )

        # Load tasks and dependencies for this run
        tasks = (
            db.query(OrchestrationTask)
            .filter(OrchestrationTask.run_id == run.id)
            .order_by(OrchestrationTask.sequence_number)
            .all()
        )

        if not tasks:
            raise HTTPException(
                status_code=400,
                detail="Mission has no tasks to convert into a routine",
            )

        dependencies = (
            db.query(OrchestrationTaskDependency)
            .filter(
                OrchestrationTaskDependency.task_id.in_(
                    [t.id for t in tasks]
                )
            )
            .all()
        )

        # Extract template definition
        template_def = _extract_routine_template(tasks, dependencies, run.goal)

        # Generate a unique template_id slug
        base_slug = f"mission-{_slugify(body.name)}"
        template_id_slug = base_slug

        # Check for collisions and append suffix if needed
        existing = (
            db.query(WorkflowTemplate.template_id)
            .filter(WorkflowTemplate.template_id.like(f"{base_slug}%"))
            .all()
        )
        existing_ids = {row[0] for row in existing}
        if template_id_slug in existing_ids:
            counter = 2
            while f"{base_slug}-{counter}" in existing_ids:
                counter += 1
            template_id_slug = f"{base_slug}-{counter}"

        template_def["id"] = template_id_slug

        # Build description from user input or generate from goal
        description = body.description or f"Routine created from mission: {run.goal[:200]}"

        # Create WorkflowTemplate record
        recipe = WorkflowTemplate(
            template_id=template_id_slug,
            name=body.name,
            description=description,
            workspace_id=ctx.workspace_id,
            owner_type="workspace",
            owner_id=str(ctx.workspace_id),
            tags=body.tags or [],
            template_definition=template_def,
            steps=[
                {
                    "step_id": tt["temp_id"],
                    "order": tt["sequence"],
                    "agent_id": None,  # Agent role, not a specific agent
                    "agent_role": tt["agent_role"],
                    "prompt_template": tt["description_pattern"],
                    "dependencies": tt["dependencies"],
                }
                for tt in template_def["task_templates"]
            ],
            inputs={"goal": {"type": "string", "required": True}},
            outputs={"report": {"type": "string"}},
            execution_config={"mode": "sequential", "max_retries": 3},
            created_by=ctx.user.id or "unknown",
            is_public=False,
            is_system=False,
        )

        db.add(recipe)
        db.flush()

        logger.info(
            "Mission %s saved as routine '%s' (template_id=%s, %d tasks)",
            mission_id,
            body.name,
            template_id_slug,
            len(tasks),
        )

        db.commit()

        return SaveAsRoutineResponse(
            template_id=template_id_slug,
            name=body.name,
            task_count=len(tasks),
        )

    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        logger.error(
            "Failed to save mission %s as routine: %s",
            mission_id,
            exc,
            exc_info=True,
        )
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
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Resume a paused mission, optionally increasing its budget."""
    try:
        run = _get_run_for_workspace(db, mission_id, ctx.workspace_id)

        if RunState(run.state) != RunState.PAUSED:
            raise HTTPException(
                status_code=400,
                detail=f"Mission is in '{run.state}' state, expected 'paused'",
            )

        # Optional: increase budget on resume
        try:
            body = await request.json()
        except Exception:
            body = {}

        additional_tokens = body.get("additional_tokens")
        additional_cost = body.get("additional_cost")

        if additional_tokens or additional_cost:
            config = run.budget_config or {}
            if additional_tokens:
                config["max_tokens"] = config.get("max_tokens", 0) + int(additional_tokens)
            if additional_cost:
                config["max_cost"] = config.get("max_cost", 0) + float(additional_cost)
            run.budget_config = config
            db.flush()
            logger.info(
                "Increased budget for mission %s: +%s tokens, +$%s",
                mission_id,
                additional_tokens,
                additional_cost,
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


@router.delete("/{mission_id}", status_code=204)
async def delete_mission(
    mission_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Delete a mission. Only allowed for terminal-state missions."""
    try:
        run = _get_run_for_workspace(db, mission_id, ctx.workspace_id)

        if RunState(run.state) not in TERMINAL_RUN_STATES:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete mission in state '{run.state}'. Cancel it first.",
            )

        # Clean up linked board tasks
        db.query(BoardTask).filter(
            BoardTask.orchestration_run_id == run.id,
        ).delete(synchronize_session="fetch")
        db.query(BoardTask).filter(
            BoardTask.orchestration_task_id.in_(
                db.query(OrchestrationTask.id).filter(
                    OrchestrationTask.run_id == run.id
                )
            ),
        ).delete(synchronize_session="fetch")

        # Delete run (cascades to tasks, events, dependencies, archives)
        db.delete(run)
        db.commit()

        logger.info("Deleted mission %s by user %s", mission_id, ctx.user.id)
        return None

    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        logger.error("Failed to delete mission %s: %s", mission_id, exc, exc_info=True)
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


# ---------------------------------------------------------------------------
# PRD-123 Pattern #8: Session Checkpoints
# ---------------------------------------------------------------------------


@router.get("/{mission_id}/checkpoints")
async def list_mission_checkpoints(
    mission_id: str,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """List available checkpoints for a mission."""
    from services.checkpoint_service import list_checkpoints

    run = _get_run_for_workspace(db, mission_id, ctx.workspace_id)
    checkpoints = await list_checkpoints(run.id)
    return {
        "mission_id": str(run.id),
        "checkpoint_count": run.checkpoint_count,
        "checkpoints": checkpoints,
    }
