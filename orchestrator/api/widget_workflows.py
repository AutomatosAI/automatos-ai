"""
Widget Workflow Control API
============================

US-014: Backend API endpoints for workflow control.
Provides pause / resume / cancel actions and a status endpoint
for the Workflow-Control widget panel.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import and_, desc
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models import (
    ExecutionStatus,
    Workflow,
    WorkflowExecution,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/workflows", tags=["widget-workflow-control"])

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class WorkflowStepStatus(BaseModel):
    """Status of a single step / stage within a workflow execution."""

    stage: int
    stage_name: str
    status: str = Field(
        ..., description="pending | running | completed | failed | skipped"
    )
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
    result: Optional[Dict[str, Any]] = None


class WorkflowStatusResponse(BaseModel):
    """Full workflow status including execution steps."""

    workflow_id: int
    workflow_name: str
    workflow_status: str
    execution_id: Optional[int] = None
    execution_status: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    steps: List[WorkflowStepStatus] = Field(default_factory=list)
    progress_pct: float = Field(
        0.0, description="Overall progress percentage (0-100)"
    )


class WorkflowActionResponse(BaseModel):
    """Response returned from pause / resume / cancel actions."""

    workflow_id: int
    execution_id: Optional[int] = None
    action: str
    previous_status: str
    new_status: str
    timestamp: str


# ---------------------------------------------------------------------------
# Stage definitions (mirrors WorkflowStageTracker.STAGES)
# ---------------------------------------------------------------------------

_STAGES: Dict[int, str] = {
    1: "Task Decomposition",
    2: "Agent Selection",
    3: "Context Engineering",
    4: "Agent Execution",
    5: "Result Aggregation",
    6: "Learning Update",
    7: "Quality Assessment",
    8: "Memory Storage",
    9: "Response Generation",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_workflow_or_404(
    workflow_id: int, workspace_id: Any, db: Session
) -> Workflow:
    """Fetch a workflow scoped to the current workspace or raise 404."""
    workflow = (
        db.query(Workflow)
        .filter(
            and_(
                Workflow.id == workflow_id,
                Workflow.workspace_id == workspace_id,
            )
        )
        .first()
    )
    if not workflow:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow {workflow_id} not found in this workspace",
        )
    return workflow


def _get_latest_execution(
    workflow_id: int, workspace_id: Any, db: Session
) -> Optional[WorkflowExecution]:
    """Return the most recent execution for the given workflow."""
    return (
        db.query(WorkflowExecution)
        .filter(
            and_(
                WorkflowExecution.workflow_id == workflow_id,
                WorkflowExecution.workspace_id == workspace_id,
            )
        )
        .order_by(desc(WorkflowExecution.started_at))
        .first()
    )


def _build_steps(execution: Optional[WorkflowExecution]) -> List[WorkflowStepStatus]:
    """
    Derive step statuses from execution metadata.

    The ``execution_metadata`` JSON column stores per-stage data written by
    ``WorkflowStageTracker``.  When no metadata exists we fall back to a
    skeleton derived from the static stage list.
    """
    if execution is None:
        return []

    meta: Dict[str, Any] = execution.execution_metadata or {}
    stages_meta: Dict[str, Any] = meta.get("stages", {})

    steps: List[WorkflowStepStatus] = []
    for stage_num, stage_name in _STAGES.items():
        stage_data = stages_meta.get(str(stage_num), {})
        step_status = stage_data.get("status", "pending")

        # If the execution has a terminal status, mark remaining pending
        # stages as "skipped" to keep the UI honest.
        if execution.status in (
            ExecutionStatus.FAILED.value,
            ExecutionStatus.CANCELLED.value,
            "paused",
        ) and step_status == "pending":
            step_status = "skipped"

        steps.append(
            WorkflowStepStatus(
                stage=stage_num,
                stage_name=stage_name,
                status=step_status,
                started_at=stage_data.get("started_at"),
                completed_at=stage_data.get("completed_at"),
                duration_ms=stage_data.get("duration_ms"),
                result=stage_data.get("result"),
            )
        )
    return steps


def _progress_pct(steps: List[WorkflowStepStatus]) -> float:
    """Calculate overall progress as a percentage (0-100)."""
    if not steps:
        return 0.0
    completed = sum(1 for s in steps if s.status == "completed")
    return round((completed / len(steps)) * 100, 1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/{workflow_id}/status",
    response_model=WorkflowStatusResponse,
    summary="Get workflow status with steps",
)
async def get_workflow_status(
    workflow_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> WorkflowStatusResponse:
    """
    Return the current status of a workflow together with per-stage step
    information for the most recent execution.
    """
    workflow = _get_workflow_or_404(workflow_id, ctx.workspace_id, db)
    execution = _get_latest_execution(workflow_id, ctx.workspace_id, db)

    steps = _build_steps(execution)
    progress = _progress_pct(steps)

    return WorkflowStatusResponse(
        workflow_id=workflow.id,
        workflow_name=workflow.name,
        workflow_status=workflow.status or "draft",
        execution_id=execution.id if execution else None,
        execution_status=execution.status if execution else None,
        started_at=(
            execution.started_at.isoformat()
            if execution and execution.started_at
            else None
        ),
        completed_at=(
            execution.completed_at.isoformat()
            if execution and execution.completed_at
            else None
        ),
        error_message=execution.error_message if execution else None,
        steps=steps,
        progress_pct=progress,
    )


@router.post(
    "/{workflow_id}/pause",
    response_model=WorkflowActionResponse,
    summary="Pause a running workflow",

    dependencies=[Depends(require_workspace_permission("missions:execute"))],
)
async def pause_workflow(
    workflow_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> WorkflowActionResponse:
    """
    Pause a currently running workflow execution.

    Only executions in ``running`` status can be paused.
    """
    workflow = _get_workflow_or_404(workflow_id, ctx.workspace_id, db)
    execution = _get_latest_execution(workflow_id, ctx.workspace_id, db)

    if execution is None:
        raise HTTPException(
            status_code=404,
            detail=f"No execution found for workflow {workflow_id}",
        )

    if execution.status != ExecutionStatus.RUNNING.value:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot pause workflow in '{execution.status}' status. Only 'running' workflows can be paused.",
        )

    previous_status = execution.status
    execution.status = "paused"
    db.commit()
    db.refresh(execution)

    logger.info(
        "Workflow %s execution %s paused by user %s",
        workflow_id,
        execution.id,
        ctx.user.email or ctx.user.id,
    )

    return WorkflowActionResponse(
        workflow_id=workflow_id,
        execution_id=execution.id,
        action="pause",
        previous_status=previous_status,
        new_status=execution.status,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.post(
    "/{workflow_id}/resume",
    response_model=WorkflowActionResponse,
    summary="Resume a paused workflow",

    dependencies=[Depends(require_workspace_permission("missions:execute"))],
)
async def resume_workflow(
    workflow_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> WorkflowActionResponse:
    """
    Resume a previously paused workflow execution.

    Only executions in ``paused`` status can be resumed.
    """
    workflow = _get_workflow_or_404(workflow_id, ctx.workspace_id, db)
    execution = _get_latest_execution(workflow_id, ctx.workspace_id, db)

    if execution is None:
        raise HTTPException(
            status_code=404,
            detail=f"No execution found for workflow {workflow_id}",
        )

    if execution.status != "paused":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot resume workflow in '{execution.status}' status. Only 'paused' workflows can be resumed.",
        )

    previous_status = execution.status
    execution.status = ExecutionStatus.RUNNING.value
    db.commit()
    db.refresh(execution)

    logger.info(
        "Workflow %s execution %s resumed by user %s",
        workflow_id,
        execution.id,
        ctx.user.email or ctx.user.id,
    )

    return WorkflowActionResponse(
        workflow_id=workflow_id,
        execution_id=execution.id,
        action="resume",
        previous_status=previous_status,
        new_status=execution.status,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.post(
    "/{workflow_id}/cancel",
    response_model=WorkflowActionResponse,
    summary="Cancel a running or paused workflow",

    dependencies=[Depends(require_workspace_permission("missions:execute"))],
)
async def cancel_workflow(
    workflow_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> WorkflowActionResponse:
    """
    Cancel a workflow execution.

    Executions in ``running`` or ``paused`` status can be cancelled.
    Terminal states (completed, failed, cancelled) are rejected.
    """
    workflow = _get_workflow_or_404(workflow_id, ctx.workspace_id, db)
    execution = _get_latest_execution(workflow_id, ctx.workspace_id, db)

    if execution is None:
        raise HTTPException(
            status_code=404,
            detail=f"No execution found for workflow {workflow_id}",
        )

    cancellable = {ExecutionStatus.RUNNING.value, ExecutionStatus.PENDING.value, "paused"}
    if execution.status not in cancellable:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot cancel workflow in '{execution.status}' status. "
                f"Only workflows in {sorted(cancellable)} status can be cancelled."
            ),
        )

    previous_status = execution.status
    execution.status = ExecutionStatus.CANCELLED.value
    execution.completed_at = datetime.utcnow()
    db.commit()
    db.refresh(execution)

    logger.info(
        "Workflow %s execution %s cancelled by user %s",
        workflow_id,
        execution.id,
        ctx.user.email or ctx.user.id,
    )

    return WorkflowActionResponse(
        workflow_id=workflow_id,
        execution_id=execution.id,
        action="cancel",
        previous_status=previous_status,
        new_status=execution.status,
        timestamp=datetime.utcnow().isoformat(),
    )
