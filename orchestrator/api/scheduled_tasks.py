"""
Scheduled Tasks API (PRD-77)
==============================
REST endpoints for viewing and managing agent-scheduled tasks.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from services.board_sla import PRIORITY_SLA_HOURS
from services.scheduled_task_service import DELIVER_BOARD_TASK, ScheduledTaskService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/scheduled-tasks", tags=["Scheduled Tasks"])


class UpdateStatusRequest(BaseModel):
    status: str  # 'cancelled' | 'paused' | 'active'


VALID_REVIEW_MODES = ("auto", "human", "llm")


class CreateScheduledBoardTaskRequest(BaseModel):
    """A board ticket to file later: what the Create Task dialog sends when the
    operator picks a time instead of "now"."""

    title: str
    description: str = ""
    schedule: str  # ISO datetime for one_shot, 5-field cron (UTC) for recurring
    task_type: str = "one_shot"
    priority: str = "medium"
    assigned_agent_id: Optional[int] = None
    review_mode: str = "auto"
    tags: List[str] = Field(default_factory=list)
    max_runs: Optional[int] = None


@router.get("")
async def list_scheduled_tasks(
    agent_id: Optional[int] = Query(None, description="Filter by agent (creator or target)"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List all scheduled tasks for the workspace."""
    svc = ScheduledTaskService(db, ctx.workspace_id)
    return await svc.list_tasks(
        agent_id=agent_id,
        status=status,
        limit=limit,
        offset=offset,
    )


@router.post("", dependencies=[Depends(require_workspace_permission("missions:create"))])
async def create_scheduled_board_task(
    body: CreateScheduledBoardTaskRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Schedule a board ticket for later.

    The row waits on the Command Centre calendar and is filed on the board when
    it fires — assigned to ``assigned_agent_id`` (the dispatcher then runs it) or
    into the Inbox. The operator is recorded on the row: on the local edition
    their scheduling is the PRD-234 consent when the ticket is filed.
    """
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="title is required")
    if body.priority not in PRIORITY_SLA_HOURS:
        raise HTTPException(status_code=422, detail=f"Invalid priority: {body.priority}")
    if body.review_mode not in VALID_REVIEW_MODES:
        raise HTTPException(status_code=422, detail=f"Invalid review_mode: {body.review_mode}")

    svc = ScheduledTaskService(db, ctx.workspace_id)
    result = await svc.create_task(
        created_by_agent_id=None,
        target_agent_id=body.assigned_agent_id,
        task_type=body.task_type,
        description=body.description.strip() or title,
        schedule=body.schedule,
        max_runs=body.max_runs,
        deliver_as=DELIVER_BOARD_TASK,
        payload={
            "title": title,
            "priority": body.priority,
            "review_mode": body.review_mode,
            "tags": [t for t in body.tags if t],
        },
        created_by_user_id=str(ctx.user.clerk_user_id or ctx.user.id),
    )
    if not result.get("success"):
        error_msg = result.get("error", "")
        status_code = 404 if "not found" in error_msg else 400
        raise HTTPException(status_code=status_code, detail=error_msg)
    return result


@router.patch("/{task_id}/status", dependencies=[Depends(require_workspace_permission("missions:update"))])
async def update_task_status(
    task_id: int,
    body: UpdateStatusRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Cancel, pause, or resume a scheduled task."""
    svc = ScheduledTaskService(db, ctx.workspace_id)
    result = await svc.update_task_status(task_id, body.status)
    if not result.get("success"):
        error_msg = result.get("error", "")
        status_code = 400 if "must be one of" in error_msg else 404
        raise HTTPException(status_code=status_code, detail=error_msg)
    return result
