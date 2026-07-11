"""
Scheduled Tasks API (PRD-77)
==============================
REST endpoints for viewing and managing agent-scheduled tasks.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from services.scheduled_task_service import ScheduledTaskService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/scheduled-tasks", tags=["Scheduled Tasks"])


class UpdateStatusRequest(BaseModel):
    status: str  # 'cancelled' | 'paused' | 'active'


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
