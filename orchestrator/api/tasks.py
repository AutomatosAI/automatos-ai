"""
Workspace Task API Endpoints
==============================
PRD-56: Infrastructure Scaling & Physical Workspaces

Exposes task lifecycle over REST:
  GET  /api/tasks           — list recent tasks for workspace
  GET  /api/tasks/{id}      — get task status + result
  POST /api/tasks/{id}/cancel — cancel a queued/running task
  GET  /api/tasks/{id}/events — SSE stream of real-time task events
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.task_runner import get_task_runner
from core.task_runner.models import TaskHandle, TaskStatusEnum

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tasks", tags=["tasks"])


# ---------------------------------------------------------------------------
# GET /api/tasks — List tasks for current workspace
# ---------------------------------------------------------------------------
@router.get("")
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List recent tasks for the current workspace."""
    from core.models import Base
    from sqlalchemy import text

    query = text("""
        SELECT id, task_type, agent_id, status, priority, runner_backend,
               submitted_at, started_at, completed_at,
               tokens_used, execution_time_ms, error_message,
               correlation_id, worker_id
        FROM task_executions
        WHERE workspace_id = :workspace_id
        {}
        {}
        ORDER BY submitted_at DESC
        LIMIT :limit OFFSET :offset
    """.format(
        "AND status = :status" if status else "",
        "AND task_type = :task_type" if task_type else "",
    ))

    params = {"workspace_id": str(ctx.workspace_id), "limit": limit, "offset": offset}
    if status:
        params["status"] = status
    if task_type:
        params["task_type"] = task_type

    rows = db.execute(query, params).fetchall()

    # Total count
    count_query = text("""
        SELECT COUNT(*) FROM task_executions
        WHERE workspace_id = :workspace_id
        {}
        {}
    """.format(
        "AND status = :status" if status else "",
        "AND task_type = :task_type" if task_type else "",
    ))
    total = db.execute(count_query, params).scalar()

    tasks = []
    for row in rows:
        tasks.append({
            "id": str(row.id),
            "task_type": row.task_type,
            "agent_id": row.agent_id,
            "status": row.status,
            "priority": row.priority,
            "runner_backend": row.runner_backend,
            "submitted_at": row.submitted_at.isoformat() if row.submitted_at else None,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "tokens_used": row.tokens_used,
            "execution_time_ms": row.execution_time_ms,
            "error_message": row.error_message,
            "correlation_id": row.correlation_id,
            "worker_id": row.worker_id,
        })

    return {"tasks": tasks, "total": total, "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# GET /api/tasks/{task_id} — Get task detail (status + result)
# ---------------------------------------------------------------------------
@router.get("/{task_id}")
async def get_task(
    task_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get full task detail including result payload."""
    from sqlalchemy import text

    row = db.execute(text("""
        SELECT id, task_type, agent_id, prompt, configuration, status,
               priority, runner_backend, resources_requested, resources_used,
               submitted_at, started_at, completed_at,
               result, error_message, tokens_used, execution_time_ms,
               parent_execution_id, correlation_id,
               worker_id, workspace_path
        FROM task_executions
        WHERE id = :task_id AND workspace_id = :workspace_id
    """), {"task_id": task_id, "workspace_id": str(ctx.workspace_id)}).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "id": str(row.id),
        "task_type": row.task_type,
        "agent_id": row.agent_id,
        "prompt": row.prompt,
        "configuration": row.configuration,
        "status": row.status,
        "priority": row.priority,
        "runner_backend": row.runner_backend,
        "resources_requested": row.resources_requested,
        "resources_used": row.resources_used,
        "submitted_at": row.submitted_at.isoformat() if row.submitted_at else None,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        "result": row.result,
        "error_message": row.error_message,
        "tokens_used": row.tokens_used,
        "execution_time_ms": row.execution_time_ms,
        "parent_execution_id": str(row.parent_execution_id) if row.parent_execution_id else None,
        "correlation_id": row.correlation_id,
        "worker_id": row.worker_id,
        "workspace_path": row.workspace_path,
    }


# ---------------------------------------------------------------------------
# POST /api/tasks/{task_id}/cancel — Cancel a queued or running task
# ---------------------------------------------------------------------------
@router.post("/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Cancel a queued or running task."""
    from sqlalchemy import text
    from uuid import UUID

    # Verify task belongs to workspace
    row = db.execute(text("""
        SELECT id, status FROM task_executions
        WHERE id = :task_id AND workspace_id = :workspace_id
    """), {"task_id": task_id, "workspace_id": str(ctx.workspace_id)}).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    if row.status in ("completed", "failed", "cancelled", "timed_out"):
        raise HTTPException(status_code=409, detail=f"Task already in terminal state: {row.status}")

    runner = get_task_runner()

    if runner.backend_name == "queued":
        handle = TaskHandle(
            task_id=task_id,
            workspace_id=ctx.workspace_id,
            status=TaskStatusEnum(row.status),
        )
        cancelled = await runner.cancel_task(handle)
        if not cancelled:
            raise HTTPException(status_code=500, detail="Failed to cancel task in queue")

    # Update DB status
    db.execute(text("""
        UPDATE task_executions SET status = 'cancelled', updated_at = NOW()
        WHERE id = :task_id AND workspace_id = :workspace_id
    """), {"task_id": task_id, "workspace_id": str(ctx.workspace_id)})
    db.commit()

    logger.info("Task %s cancelled by user (workspace=%s)", task_id[:8], ctx.workspace_id)
    return {"id": task_id, "status": "cancelled"}


# ---------------------------------------------------------------------------
# GET /api/tasks/{task_id}/events — SSE stream for real-time task updates
# ---------------------------------------------------------------------------
@router.get("/{task_id}/events")
async def stream_task_events(
    task_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Stream real-time task events via Server-Sent Events (SSE)."""
    from sqlalchemy import text

    # Verify task belongs to workspace
    row = db.execute(text("""
        SELECT id, status FROM task_executions
        WHERE id = :task_id AND workspace_id = :workspace_id
    """), {"task_id": task_id, "workspace_id": str(ctx.workspace_id)}).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    runner = get_task_runner()

    if runner.backend_name != "queued":
        raise HTTPException(status_code=400, detail="Event streaming only available with queued backend")

    handle = TaskHandle(
        task_id=task_id,
        workspace_id=ctx.workspace_id,
        status=TaskStatusEnum(row.status),
    )

    async def event_generator():
        try:
            async for event in runner.stream_updates(handle):
                yield f"event: {event.event_type.value}\ndata: {json.dumps(event.data)}\n\n"

                # Stop streaming on terminal states
                if event.event_type == "status_changed" and event.data.get("status") in (
                    "completed", "failed", "cancelled", "timed_out"
                ):
                    break
        except Exception as e:
            logger.error("SSE stream error for task %s: %s", task_id[:8], e)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
