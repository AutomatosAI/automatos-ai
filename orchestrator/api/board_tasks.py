"""
Board Tasks API
===============

CRUD + planning endpoints for the lightweight task board (PRD-72).
Tasks follow a Kanban lifecycle: inbox -> assigned -> in_progress -> review -> blocked -> done.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from config import config
from core.auth.hybrid import get_request_context_hybrid, require_task_context
from core.auth.workspace_permission import require_workspace_permission
from core.auth.dependencies import RequestContext
from core.auth.scopes import TASKS_READ
from core.database.database import get_db
from core.models.core import BoardTask
from core.models import Agent
from core.utils.exception_telemetry import record_error
from core.utils.background_tasks import launch_guarded
from services.board_dispatcher import notify_task_available
from services.board_events import board_event_stream, notify_board_event

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/tasks", tags=["board-tasks"])

VALID_STATUSES = {"inbox", "assigned", "in_progress", "review", "blocked", "done", "failed"}
VALID_PRIORITIES = {"urgent", "high", "medium", "low"}
VALID_REVIEW_MODES = {"human", "llm", "auto"}

# PRD-171 F025: source_types the board must NOT self-execute on drag/PATCH.
# 'recipe' runs through the recipe executor; 'orchestration'/'orchestration_task'
# are mission mirrors the mission engine already owns (orchestration_board_bridge)
# — firing a board execution on them double-runs work the mission drives.
_NON_EXECUTABLE_SOURCE_TYPES = frozenset(
    {"recipe", "orchestration", "orchestration_task"}
)

# Priority → SLA deadline hours
_PRIORITY_SLA_HOURS: dict[str, int] = {
    "urgent": 4,
    "high": 12,
    "medium": 24,
    "low": 72,
}


# ── Auto-report creation (mirrors heartbeat_service._auto_create_report) ───
async def _auto_create_task_report(
    db: Session,
    workspace_id: str,
    task: BoardTask,
    exec_result: Dict[str, Any],
) -> None:
    """
    Persist an agent_reports row for a completed task so it shows up in
    Reports / Deliverables / Activity Feed — same pattern heartbeats use.
    Always non-blocking: never raises, just warns on failure.
    """
    try:
        from services.report_service import ReportService, compute_execution_metrics

        agent_name = "Unknown Agent"
        if task.assigned_agent_id:
            agent = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first()
            if agent:
                agent_name = agent.name

        # Source the body from the agent's actual response, falling back to whatever
        # text was captured in task.result.
        llm_text = (
            exec_result.get("result")
            or exec_result.get("response")
            or exec_result.get("output")
            or exec_result.get("content")
            or task.result
            or ""
        )

        # Pull cost/model/duration rollup from llm_usage (window = task started→completed)
        exec_metrics = compute_execution_metrics(
            db,
            workspace_id,
            agent_id=task.assigned_agent_id,
            execution_id=getattr(task, "execution_id", None),
            started_at=getattr(task, "started_at", None),
            completed_at=getattr(task, "completed_at", None),
            extra={
                "task_id": task.id,
                "task_status": task.status,
                "trigger": "task",
            },
        )

        # Honour upstream-supplied tokens if the rollup found nothing
        if not exec_metrics.get("tokens_used"):
            usage = exec_result.get("usage") or {}
            fallback_tokens = (
                usage.get("total_tokens")
                or exec_result.get("tokens_used")
                or 0
            )
            if fallback_tokens:
                exec_metrics["tokens_used"] = fallback_tokens

        report_status = "ok" if task.status in ("done", "review") else "warning"
        if task.error_message:
            report_status = "critical"

        # Render the same shape heartbeat reports use so consumers stay uniform.
        lines = [
            f"# {agent_name} — Task Report",
            f"**Task:** {task.title}",
            f"**Status:** {task.status}",
            "",
        ]
        if task.error_message:
            lines.append("## Error")
            lines.append(str(task.error_message))
            lines.append("")
        if llm_text:
            lines.append("## Result")
            lines.append(str(llm_text))
            lines.append("")
        lines.append("## Execution Metrics")
        lines.append(f"- Model: {exec_metrics.get('model') or 'unknown'}")
        lines.append(f"- LLM calls: {exec_metrics.get('llm_calls', 0)}")
        lines.append(f"- Tokens (in/out/total): "
                     f"{exec_metrics.get('input_tokens', 0)} / "
                     f"{exec_metrics.get('output_tokens', 0)} / "
                     f"{exec_metrics.get('tokens_used', 0)}")
        lines.append(f"- Cost: ${exec_metrics.get('cost_usd', 0):.4f}")
        if exec_metrics.get("duration_ms") is not None:
            lines.append(f"- Duration: {exec_metrics['duration_ms']} ms")
        content = "\n".join(lines)

        # Summary is the first non-empty body line — same convention as heartbeat reports.
        summary = None
        for line in str(llm_text).split("\n"):
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                summary = (stripped[:497] + "...") if len(stripped) > 497 else stripped
                break

        svc = ReportService(db, workspace_id)
        report_result = await svc.create_report(
            agent_id=task.assigned_agent_id,
            agent_name=agent_name,
            title=f"Task: {task.title}",
            content=content,
            report_type="task",
            status=report_status,
            summary=summary,
            metrics=exec_metrics,
        )
        if not report_result.get("success"):
            logger.warning(
                "[BoardTasks] Auto-report creation failed for task=%s: %s",
                task.id, report_result.get("error"),
            )
    except Exception:
        logger.error(
            "[BoardTasks] _auto_create_task_report raised for task=%s",
            getattr(task, "id", "?"),
            exc_info=True,
        )


# ── PRD-128: Unified notification dispatch ─────────────────────────
async def _dispatch_task_complete(db: Session, workspace_id, task: BoardTask) -> None:
    """Fire a ``task_complete`` event through NotificationDispatcher.

    Uses the caller's DB session so the notification row joins the
    existing transaction (no extra commits). Dispatcher never raises on
    delivery failures, but we still wrap in try/except so any programming
    error cannot block the task-completion flow.
    """
    try:
        from core.services.notification_dispatcher import NotificationDispatcher

        agent_name = None
        if task.assigned_agent_id:
            agent = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first()
            agent_name = agent.name if agent else f"agent-{task.assigned_agent_id}"

        message = None
        if task.result:
            message = str(task.result)[:500]
        elif task.description:
            message = task.description[:500]

        dispatcher = NotificationDispatcher(db, str(workspace_id))
        await dispatcher.dispatch(
            event_type="task_complete",
            title=f"Task: {task.title}",
            message=message,
            link_type="task",
            link_id=str(task.id),
            agent_id=task.assigned_agent_id,
            agent_name=agent_name,
            status="ok",
        )
    except Exception:
        logger.error(
            "[BoardTasks] task_complete dispatch failed for task %s",
            getattr(task, "id", "?"),
            exc_info=True,
        )

    # PRD-204 S3: board-task terminal choke point (success) — every
    # completion path funnels through this helper. Fail-soft.
    from services.watch_hooks import watch_ingest_terminal

    watch_ingest_terminal(
        db,
        workspace_id=workspace_id,
        target_type="board_task",
        target_id=str(task.id),
        terminal_state="completed",
        summary=(str(task.result)[:500] if task.result else None),
    )


async def _dispatch_task_failed(db: Session, workspace_id, task: BoardTask) -> None:
    """Fire a ``task_failed`` event (PRD-161 S3).

    Mirrors ``_dispatch_task_complete`` but signals an error terminal state, so
    the user is told the task did NOT succeed — previously a crashed execution
    closed silently as 'done'. Never raises into the execution flow.
    """
    try:
        from core.services.notification_dispatcher import NotificationDispatcher

        agent_name = None
        if task.assigned_agent_id:
            agent = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first()
            agent_name = agent.name if agent else f"agent-{task.assigned_agent_id}"

        dispatcher = NotificationDispatcher(db, str(workspace_id))
        await dispatcher.dispatch(
            event_type="task_failed",
            title=f"Task failed: {task.title}",
            message=(task.error_message or "Execution failed")[:500],
            link_type="task",
            link_id=str(task.id),
            agent_id=task.assigned_agent_id,
            agent_name=agent_name,
            status="error",
        )
    except Exception:
        logger.error(
            "[BoardTasks] task_failed dispatch failed for task %s",
            getattr(task, "id", "?"),
            exc_info=True,
        )

    # PRD-204 S3: board-task terminal choke point (failure). Fail-soft.
    from services.watch_hooks import watch_ingest_terminal

    watch_ingest_terminal(
        db,
        workspace_id=workspace_id,
        target_type="board_task",
        target_id=str(task.id),
        terminal_state="failed",
        summary=(task.error_message or "Execution failed")[:500],
    )


# ── Helpers ──────────────────────────────────────────────────────────

def _enrich_with_agents(tasks: list, db: Session, workspace_id) -> list:
    """Join agent info onto task dicts.

    Agents are resolved within ``workspace_id`` only: a task whose
    ``assigned_agent_id`` points at another workspace's agent yields no ``agent``
    block rather than leaking that agent's name/icon (defense-in-depth tenant
    isolation — board reads are now reachable by per-workspace SDK keys).
    """
    agent_ids = {t.assigned_agent_id for t in tasks if t.assigned_agent_id}
    if not agent_ids:
        return [t.to_dict() for t in tasks]

    agents = {
        a.id: a
        for a in db.query(Agent)
        .filter(Agent.id.in_(agent_ids), Agent.workspace_id == workspace_id)
        .all()
    }

    result = []
    for t in tasks:
        d = t.to_dict()
        agent = agents.get(t.assigned_agent_id)
        if agent:
            d["agent"] = {
                "id": agent.id,
                "name": agent.name,
                "agent_icon": getattr(agent, "premium_icon", None),
            }
        result.append(d)
    return result


# ── CRUD ─────────────────────────────────────────────────────────────

@router.post("", dependencies=[Depends(require_workspace_permission("missions:create"))])
async def create_task(
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create a new board task."""
    body = await request.json()

    title = (body.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=422, detail="title is required")

    assigned_agent_id = body.get("assigned_agent_id")
    if assigned_agent_id is not None:
        assigned_agent_id = int(assigned_agent_id)
        agent = db.query(Agent).filter(
            Agent.id == assigned_agent_id,
            Agent.workspace_id == ctx.workspace_id,
        ).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Assigned agent not found in workspace")

    priority = body.get("priority", "medium")
    if priority not in VALID_PRIORITIES:
        raise HTTPException(status_code=422, detail=f"Invalid priority: {priority}")

    review_mode = body.get("review_mode", "auto")
    if review_mode not in VALID_REVIEW_MODES:
        raise HTTPException(status_code=422, detail=f"Invalid review_mode: {review_mode}")

    planning_data = body.get("planning_data")
    # Auto-set review status when approval_action is present
    if planning_data and isinstance(planning_data, dict) and planning_data.get("approval_action"):
        status = "review"
    else:
        status = "assigned" if assigned_agent_id else "inbox"

    # PRD-127: ephemeral attachments
    attachment_ids = body.get("attachment_ids", [])
    if attachment_ids and not isinstance(attachment_ids, list):
        raise HTTPException(status_code=422, detail="attachment_ids must be a list")

    task = BoardTask(
        workspace_id=ctx.workspace_id,
        title=title,
        description=body.get("description"),
        raw_prompt=body.get("raw_prompt"),
        status=status,
        priority=priority,
        review_mode=review_mode,
        assigned_agent_id=assigned_agent_id,
        created_by_type="user",
        created_by_id=ctx.user.clerk_user_id or ctx.user.id,
        parent_task_id=body.get("parent_task_id"),
        tags=body.get("tags", []),
        planning_data=planning_data,
        attachment_ids=attachment_ids,  # PRD-127
        sla_deadline=datetime.now(timezone.utc) + timedelta(hours=_PRIORITY_SLA_HOURS.get(priority, 24)),
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # PRD-180 S1 (F090): push the new card to subscribed Command Centres.
    notify_board_event(
        db, workspace_id=ctx.workspace_id, task_id=task.id, status=task.status,
        event="task_created",
    )

    # PRD-161: assignment = immediate dispatch via the board loop (Q39/Q40).
    # A created-as-assigned task notifies the claimant; the dispatch loop claims
    # it (FOR UPDATE SKIP LOCKED) and runs it — no inline launch, no heartbeat wait.
    if task.status == "assigned" and task.assigned_agent_id and task.source_type != "recipe":
        notify_task_available(db, workspace_id=ctx.workspace_id, task_id=task.id)

    logger.info("[BoardTasks] Created task %d in workspace %s", task.id, ctx.workspace_id)
    return task.to_dict()


@router.get("")
async def list_tasks(
    ctx: RequestContext = Depends(require_task_context(TASKS_READ)),
    db: Session = Depends(get_db),
    status: Optional[str] = Query(None, description="Comma-separated statuses"),
    agent_id: Optional[int] = Query(None),
    priority: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    parent_task_id: Optional[int] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List board tasks with optional filters."""
    query = db.query(BoardTask).filter(BoardTask.workspace_id == ctx.workspace_id)

    if status:
        statuses = [s.strip() for s in status.split(",") if s.strip()]
        invalid = [s for s in statuses if s not in VALID_STATUSES]
        if invalid:
            raise HTTPException(status_code=422, detail=f"Invalid status values: {invalid}")
        query = query.filter(BoardTask.status.in_(statuses))

    if agent_id is not None:
        query = query.filter(BoardTask.assigned_agent_id == agent_id)

    if priority:
        if priority not in VALID_PRIORITIES:
            raise HTTPException(status_code=422, detail=f"Invalid priority: {priority}")
        query = query.filter(BoardTask.priority == priority)

    if parent_task_id is not None:
        query = query.filter(BoardTask.parent_task_id == parent_task_id)

    if search:
        like_term = f"%{search}%"
        query = query.filter(BoardTask.title.ilike(like_term))

    # PRD-161 S5: archive — done tasks completed longer ago than the configured
    # window drop off the active board (retained in the DB, just not surfaced).
    archive_before = datetime.now(timezone.utc) - timedelta(days=config.BOARD_ARCHIVE_DONE_DAYS)
    query = query.filter(
        ~(
            (BoardTask.status == "done")
            & BoardTask.completed_at.isnot(None)
            & (BoardTask.completed_at < archive_before)
        )
    )

    total = query.count()
    tasks = (
        query.order_by(BoardTask.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "tasks": _enrich_with_agents(tasks, db, ctx.workspace_id),
        "total": total,
    }


@router.get("/stream")
async def stream_board_events(
    ctx: RequestContext = Depends(require_task_context(TASKS_READ)),
):
    """Real-time board SSE via Postgres ``LISTEN/NOTIFY`` (PRD-180 S1, F090).

    Replaces the old timed ping: the stream ``LISTEN``s the ``board_events``
    channel and forwards each board-task mutation (insert / status change /
    claim / requeue) to this client sub-second, scoped to the caller's
    workspace. A heartbeat comment keeps the connection alive but does not drive
    refreshes — real NOTIFY events do. Rides the read-only ``TASKS_READ`` scope
    (Q42): the shared hybrid auth is untouched and no write scope is introduced.
    """
    workspace_id = str(ctx.workspace_id)

    return StreamingResponse(
        board_event_stream(
            workspace_id, heartbeat_seconds=config.BOARD_SSE_HEARTBEAT_SECONDS
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{task_id}")
async def get_task(
    task_id: int,
    ctx: RequestContext = Depends(require_task_context(TASKS_READ)),
    db: Session = Depends(get_db),
):
    """Get a single board task."""
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    enriched = _enrich_with_agents([task], db, ctx.workspace_id)
    return enriched[0]


@router.patch("/{task_id}", dependencies=[Depends(require_workspace_permission("missions:update"))])
async def update_task(
    task_id: int,
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update a board task (partial)."""
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    body = await request.json()

    if "title" in body:
        title = (body["title"] or "").strip()
        if not title:
            raise HTTPException(status_code=422, detail="title cannot be empty")
        task.title = title

    if "description" in body:
        task.description = body["description"]

    if "status" in body:
        new_status = body["status"]
        if new_status not in VALID_STATUSES:
            raise HTTPException(status_code=422, detail=f"Invalid status: {new_status}")
        old_status = task.status
        task.status = new_status
        if new_status == "in_progress" and not task.started_at:
            task.started_at = datetime.now(timezone.utc)
        if new_status in ("done", "review"):
            task.completed_at = datetime.now(timezone.utc)
        if new_status == "blocked" and task.blocked_at is None:
            task.blocked_at = datetime.now(timezone.utc)
            task.blocked_reason = body.get("blocked_reason")
        if new_status != "blocked" and old_status == "blocked":
            task.blocked_at = None
            task.blocked_reason = None

    if "priority" in body:
        if body["priority"] not in VALID_PRIORITIES:
            raise HTTPException(status_code=422, detail=f"Invalid priority: {body['priority']}")
        task.priority = body["priority"]

    if "review_mode" in body:
        if body["review_mode"] not in VALID_REVIEW_MODES:
            raise HTTPException(status_code=422, detail=f"Invalid review_mode: {body['review_mode']}")
        task.review_mode = body["review_mode"]

    if "assigned_agent_id" in body:
        agent_id_val = body["assigned_agent_id"]
        if agent_id_val is not None:
            agent_id_val = int(agent_id_val)
            agent = db.query(Agent).filter(
                Agent.id == agent_id_val,
                Agent.workspace_id == ctx.workspace_id,
            ).first()
            if not agent:
                raise HTTPException(status_code=404, detail="Assigned agent not found in workspace")
        task.assigned_agent_id = agent_id_val
        # Auto-transition from inbox to assigned when an agent is set
        if agent_id_val and task.status == "inbox":
            task.status = "assigned"

    if "result" in body:
        task.result = body["result"]

    if "error_message" in body:
        task.error_message = body["error_message"]

    if "tags" in body:
        task.tags = body["tags"]

    if "planning_data" in body:
        task.planning_data = body["planning_data"]

    # Check if we need to trigger execution
    # PRD-171 F025: only user-owned board tasks self-execute on a status flip.
    # Recipe + mission-mirror rows are driven by their own engines.
    trigger_execution = (
        "status" in body
        and body["status"] == "in_progress"
        and task.assigned_agent_id
        and task.source_type not in _NON_EXECUTABLE_SOURCE_TYPES
    )

    # PRD-128: dispatch task_complete on terminal transition
    if "status" in body and body["status"] == "done":
        await _dispatch_task_complete(db, ctx.workspace_id, task)

    db.commit()
    db.refresh(task)

    # PRD-180 S1 (F090): push the mutation to subscribed Command Centres.
    notify_board_event(
        db, workspace_id=ctx.workspace_id, task_id=task.id, status=task.status,
        event="task_updated",
    )

    if trigger_execution:
        _launch_task_execution(
            task_id=task.id,
            agent_id=task.assigned_agent_id,
            workspace_id=str(ctx.workspace_id),
            prompt=task.raw_prompt or task.description or task.title,
            review_mode=task.review_mode or "auto",
            attachment_ids=task.attachment_ids,  # PRD-127
        )
    elif (
        "assigned_agent_id" in body
        and task.status == "assigned"
        and task.assigned_agent_id
        and task.source_type != "recipe"
    ):
        # PRD-161: assigning notifies the dispatch loop (single spine); the loop
        # claims 'assigned' tasks only, so re-assigning a running task is a no-op.
        notify_task_available(db, workspace_id=ctx.workspace_id, task_id=task.id)

    logger.info("[BoardTasks] Updated task %d", task.id)
    return task.to_dict()


@router.delete("/{task_id}", dependencies=[Depends(require_workspace_permission("missions:delete"))])
async def delete_task(
    task_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Delete a board task."""
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    db.delete(task)
    db.commit()

    logger.info("[BoardTasks] Deleted task %d", task.id)
    return {"success": True, "deleted_id": task_id}


# ── Status shortcut (drag-and-drop) ─────────────────────────────────

@router.post("/{task_id}/approve", dependencies=[Depends(require_workspace_permission("missions:update"))])
async def approve_task(
    task_id: int,
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Approve a board task in review status.

    If the task has an approval_action in planning_data, execute it
    (e.g., publish a blog post). Then move the task to done.
    """
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "review":
        raise HTTPException(status_code=422, detail=f"Task must be in review status (currently: {task.status})")

    body = await request.json()
    action_result = None

    # Execute approval_action if present
    approval_action = (task.planning_data or {}).get("approval_action")
    if approval_action:
        action_type = approval_action.get("type")
        try:
            if action_type == "publish_blog":
                from core.services.blog_service import BlogService
                post_id = approval_action.get("post_id")
                if not post_id:
                    raise HTTPException(status_code=422, detail="approval_action missing post_id")
                svc = BlogService(db, ctx.workspace_id)
                post = svc.publish_post(UUID(post_id))
                if not post:
                    raise HTTPException(status_code=404, detail=f"Blog post {post_id} not found")
                action_result = {
                    "type": "publish_blog",
                    "post_id": str(post.id),
                    "title": post.title,
                    "slug": post.slug,
                    "status": post.status,
                    "url": f"/api/widgets/blog/posts/{post.slug}?workspace_id={ctx.workspace_id}",
                }
                logger.info("[BoardTasks] Approved: published blog post %s (%s)", post.id, post.title)
            elif action_type == "create_blog":
                # Used by VECTOR (and any agent) to suggest a blog topic for
                # founder approval. On approve, fire the standard blog mission.
                from modules.tools.discovery.handlers_blog import (
                    create_blog_post_from_topic,
                )
                topic = approval_action.get("topic")
                category = approval_action.get("category") or "AI & Automation"
                if not topic:
                    raise HTTPException(status_code=422, detail="approval_action missing topic")
                user_id = ctx.user.clerk_user_id if ctx.user else None
                result = await create_blog_post_from_topic(
                    db,
                    ctx.workspace_id,
                    {"topic": topic, "category": category, "_user_id": user_id},
                )
                if not result.get("success"):
                    raise HTTPException(
                        status_code=500,
                        detail=f"Blog mission start failed: {result.get('error', 'unknown')}",
                    )
                action_result = {
                    "type": "create_blog",
                    "mission_id": result.get("mission_id"),
                    "topic": topic,
                    "category": category,
                    "task_count": result.get("task_count", 0),
                }
                logger.info(
                    "[BoardTasks] Approved: created blog mission %s for topic '%s'",
                    result.get("mission_id"), topic,
                )
            else:
                logger.warning("[BoardTasks] Unknown approval_action type: %s", action_type)
                action_result = {"type": action_type, "warning": "Unknown action type, task approved without side-effect"}
        except HTTPException:
            raise
        except Exception as e:
            logger.error("[BoardTasks] Approval action failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Approval action failed: {e}")

    # Move to done
    task.status = "done"
    task.completed_at = datetime.now(timezone.utc)
    if action_result:
        task.result = json.dumps(action_result) if not task.result else task.result

    # PRD-128: dispatch task_complete before commit so notification row
    # joins the same transaction as the task update.
    await _dispatch_task_complete(db, ctx.workspace_id, task)

    db.commit()
    db.refresh(task)

    logger.info("[BoardTasks] Task %d approved and moved to done", task.id)
    return {
        "success": True,
        "task_id": task.id,
        "status": task.status,
        "action_result": action_result,
    }


@router.post("/{task_id}/reject", dependencies=[Depends(require_workspace_permission("missions:update"))])
async def reject_task(
    task_id: int,
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Reject a board task in review status with reviewer feedback (PRD-161 Q44).

    Returns the task to the SAME agent as 'assigned' — not dumped back to inbox —
    with the feedback carried into the next execution's context (review_feedback),
    so the agent redoes the work with the correction. The dispatch loop picks the
    re-assigned task up immediately.
    """
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "review":
        raise HTTPException(status_code=422, detail=f"Task must be in review status (currently: {task.status})")
    if not task.assigned_agent_id:
        raise HTTPException(status_code=422, detail="Cannot reject a task with no assigned agent")

    body = await request.json()
    feedback = (body.get("feedback") or "").strip()

    # Q44: back to the same agent for another attempt, feedback in context.
    task.status = "assigned"
    task.started_at = None
    task.completed_at = None
    task.result = None
    task.lease_until = None
    task.attempts = 0  # a human-driven redo is a fresh attempt cycle
    task.review_feedback = feedback or None
    db.commit()
    db.refresh(task)

    # Wake the dispatch loop so the redo starts immediately (single spine).
    if task.source_type != "recipe":
        notify_task_available(db, workspace_id=ctx.workspace_id, task_id=task.id)

    logger.info("[BoardTasks] Task %d rejected → re-assigned to agent %s%s",
                task.id, task.assigned_agent_id,
                f" with feedback: {feedback}" if feedback else "")
    return {
        "success": True,
        "task_id": task.id,
        "status": task.status,
        "assigned_agent_id": task.assigned_agent_id,
        "feedback": feedback or None,
    }


@router.post("/{task_id}/run-now", dependencies=[Depends(require_workspace_permission("missions:execute"))])
async def run_task_now(
    task_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """PRD-161 S5: dispatch a task immediately.

    Resets the task to a fresh ``assigned`` claim (clears lease + attempts) and
    notifies the dispatch loop, so a failed, idle, or just-created task can be
    re-run on demand from the board. A task already in_progress is left alone.
    """
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if not task.assigned_agent_id:
        raise HTTPException(status_code=422, detail="Assign an agent before running the task")
    if task.status == "in_progress":
        raise HTTPException(status_code=409, detail="Task is already running")

    task.status = "assigned"
    task.lease_until = None
    task.attempts = 0
    task.completed_at = None
    task.started_at = None
    db.commit()
    db.refresh(task)

    if task.source_type != "recipe":
        notify_task_available(db, workspace_id=ctx.workspace_id, task_id=task.id)

    logger.info("[BoardTasks] Run Now → task %d re-dispatched to agent %s",
                task.id, task.assigned_agent_id)
    return {"success": True, "task_id": task.id, "status": task.status}


@router.patch("/{task_id}/status", dependencies=[Depends(require_workspace_permission("missions:update"))])
async def update_task_status(
    task_id: int,
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update only the status of a task (for drag-and-drop on the board)."""
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    body = await request.json()
    new_status = body.get("status", "").strip()
    if new_status not in VALID_STATUSES:
        raise HTTPException(status_code=422, detail=f"Invalid status: {new_status}")

    old_status = task.status
    task.status = new_status
    if new_status == "in_progress":
        task.started_at = datetime.now(timezone.utc)
        task.completed_at = None
        task.error_message = None
        task.result = None
    if new_status in ("done", "review") and not task.completed_at:
        task.completed_at = datetime.now(timezone.utc)
    if new_status == "blocked" and task.blocked_at is None:
        task.blocked_at = datetime.now(timezone.utc)
    if new_status != "blocked" and old_status == "blocked":
        task.blocked_at = None
        task.blocked_reason = None

    # PRD-128: dispatch task_complete on drag-to-done transitions
    if new_status == "done":
        await _dispatch_task_complete(db, ctx.workspace_id, task)

    db.commit()
    db.refresh(task)

    # PRD-180 S1 (F090): push the drag-and-drop status change to Command Centres.
    notify_board_event(
        db, workspace_id=ctx.workspace_id, task_id=task.id, status=task.status,
        event="status_changed",
    )

    # Fire-and-forget: trigger agent execution when moved to in_progress.
    # PRD-171 F025: exclude recipe + mission-mirror rows — dragging a mission
    # mirror to in_progress must not re-run work the mission engine owns.
    if (
        new_status == "in_progress"
        and task.assigned_agent_id
        and task.source_type not in _NON_EXECUTABLE_SOURCE_TYPES
    ):
        _launch_task_execution(
            task_id=task.id,
            agent_id=task.assigned_agent_id,
            workspace_id=str(ctx.workspace_id),
            prompt=task.raw_prompt or task.description or task.title,
            review_mode=task.review_mode or "auto",
            attachment_ids=task.attachment_ids,  # PRD-127
        )

    return {"id": task.id, "status": task.status}


# ── Immediate execution (fire-and-forget) ────────────────────────────

async def _lease_heartbeat(task_id: int) -> None:
    """PRD-171 F024: keep a long-running task's dispatch lease alive.

    Runs concurrently with the execution and renews ``lease_until`` every half
    the lease window on its OWN short-lived session, so a legitimately long run
    (> ``BOARD_DISPATCH_LEASE_SECONDS``) is never swept back to ``assigned`` and
    re-claimed. Cancelled the moment the run finishes; if the process crashes the
    heartbeat dies with it, the lease truly lapses, and the sweeper requeues the
    dead run — exactly the intended behaviour. Best-effort throughout: a failed
    renewal is logged and retried, never propagated into the run.
    """
    from core.database.database import SessionLocal
    from services.board_dispatcher import renew_lease

    lease_seconds = config.BOARD_DISPATCH_LEASE_SECONDS
    # Renew well within the window so a slow tick never lets the lease lapse.
    interval = max(1.0, lease_seconds / 2)
    try:
        while True:
            await asyncio.sleep(interval)
            hb = SessionLocal()
            try:
                if not renew_lease(hb, task_id, lease_seconds=lease_seconds):
                    # Row is no longer in_progress (finished/failed/requeued) —
                    # nothing more to renew.
                    break
            except Exception:
                logger.warning(
                    "[BoardTasks] lease heartbeat failed for task %d", task_id,
                    exc_info=True,
                )
                hb.rollback()
            finally:
                hb.close()
    except asyncio.CancelledError:
        raise


def _estimate_board_task_cost_usd(db, task_id: int, agent_id: int) -> float:
    """PRD-192 S3: a REAL dollar estimate for the pending board task.

    Before this the approval gate always received the ``estimated_cost_usd=0.0``
    default, so an ``auto_below_budget`` policy auto-approved every task (C.5 —
    the ceiling could never bind). Estimator: prompt tokens of the task's text
    (``raw_prompt`` else title+description) + the agent's configured output cap
    (``model_config.max_tokens``, else the model registry's own ceiling),
    priced by ``modules.policy.pricing`` against the task agent's model — the
    flat rate applies only inside pricing as the registry-miss last resort.

    Never raises: 0.0 when nothing is resolvable (the gate then behaves as
    before for that task).
    """
    try:
        from core.context_guard import count_tokens
        from modules.policy import pricing as _pricing

        task = db.query(BoardTask).get(task_id)
        if task is None:
            return 0.0
        prompt_text = task.raw_prompt or " ".join(
            p for p in (task.title, task.description) if p
        )
        est_in = count_tokens(prompt_text or "")

        model_id = None
        est_out = 0
        agent = db.query(Agent).get(agent_id) if agent_id else None
        mc = getattr(agent, "model_config", None) or {}
        if isinstance(mc, dict):
            model_id = mc.get("model_id")
            try:
                est_out = int(mc.get("max_tokens") or 0)
            except (TypeError, ValueError):
                est_out = 0
        if model_id and not est_out:
            try:
                from core.models import LLMModel

                m = db.query(LLMModel).filter_by(model_id=model_id).first()
                est_out = int(m.max_output_tokens or 0) if m else 0
            except Exception:
                est_out = 0

        if model_id:
            priced = _pricing.estimate_cost_usd(db, model_id, est_in, est_out)
            if priced is not None:
                return float(priced)
        # No model / registry miss ⇒ the ONE flat last-resort, inside pricing.
        return _pricing.price_total_tokens_usd(db, model_id, est_in + est_out)
    except Exception:
        logger.warning(
            "[BoardTasks] cost estimate failed for task %s — passing 0.0",
            task_id, exc_info=True,
        )
        return 0.0


def _board_task_blocked_pending_approval(
    db, task_id: int, agent_id: int, workspace_id: str
) -> bool:
    """PRD-181 S2: return True if the board task must wait for human approval.

    - An active (granted, unexpired) grant for this task ⇒ proceed (False).
    - Otherwise evaluate the workspace approval policy. If it asks, block the
      task (status → ``blocked``, ``blocked_reason`` referencing the grant) and
      return True so the caller does NOT execute it.

    Fail posture (PRD-192 S1, locked #4): under the policy plane's ENFORCE
    stages (``destructive`` | ``on``) an approval-gate ERROR blocks the task
    pending approval — an errored governance gate must never launch autonomous
    work. In ``off``/``shadow`` the historical fail-open stands (the per-tool
    PolicyGate still applies to every tool the task's agent invokes).
    """
    try:
        from core.services.approval_grants import find_active_grant
        from core.models.approval_grants import SUBJECT_BOARD_TASK
        from services.board_approval import evaluate_board_task_approval

        # Already authorised? Proceed.
        if find_active_grant(
            db, workspace_id, subject_type=SUBJECT_BOARD_TASK, subject_id=str(task_id)
        ) is not None:
            return False

        outcome = evaluate_board_task_approval(
            db, workspace_id=workspace_id, task_id=task_id, agent_id=agent_id,
            # PRD-192 S3: a real priced figure — auto_below_budget can bind (C.5).
            estimated_cost_usd=_estimate_board_task_cost_usd(db, task_id, agent_id),
        )
        if not outcome.requires_approval:
            return False

        # Block the task until a human grants the pending grant.
        task = db.query(BoardTask).get(task_id)
        if task is not None and task.status == "in_progress":
            task.status = "blocked"
            task.blocked_at = datetime.now(timezone.utc)
            grant_id = getattr(outcome.grant, "id", None)
            task.blocked_reason = (
                f"Awaiting human approval (grant #{grant_id}): {outcome.reason}"
            )
            db.commit()
            logger.info(
                "[BoardTasks] task %s blocked pending approval grant #%s",
                task_id, grant_id,
            )
        return True
    except Exception:
        try:
            from modules.policy.flag import enforcement_active

            _fail_closed = enforcement_active()
        except Exception:
            _fail_closed = False

        if not _fail_closed:
            logger.warning(
                "[BoardTasks] approval gate errored for task %s — proceeding "
                "(per-tool PolicyGate still applies)", task_id, exc_info=True,
            )
            return False

        # Enforce stage: BLOCK pending approval, never launch on a gate error.
        logger.error(
            "[BoardTasks] approval gate errored for task %s — BLOCKED pending "
            "approval (policy plane enforce stage fails closed)", task_id,
            exc_info=True,
        )
        try:
            db.rollback()  # the failed gate may have poisoned the transaction
            task = db.query(BoardTask).get(task_id)
            if task is not None and task.status == "in_progress":
                task.status = "blocked"
                task.blocked_at = datetime.now(timezone.utc)
                task.blocked_reason = (
                    "Approval gate errored — blocked pending approval "
                    "(policy plane enforce stage fails closed)"
                )
                db.commit()
        except Exception:
            logger.warning(
                "[BoardTasks] could not mark task %s blocked after gate error "
                "— task still NOT launched", task_id, exc_info=True,
            )
        return True


def _launch_task_execution(
    task_id: int,
    agent_id: int,
    workspace_id: str,
    prompt: str,
    review_mode: str = "auto",
    attachment_ids: Optional[list] = None,  # PRD-127
):
    """Launch agent execution for a board task as a background coroutine."""

    async def _run():
        from core.database.database import SessionLocal
        db = SessionLocal()
        # PRD-171 F024: heartbeat the dispatch lease for the life of the run.
        heartbeat = asyncio.ensure_future(_lease_heartbeat(task_id))
        try:
            # PRD-181 S2 (F060): board-task approval gate. Before an autonomous
            # board task executes, run it through the SAME approval primitive
            # missions use. If the workspace policy asks (always_ask / over the
            # dollar ceiling), a durable, revocable, expiring approval-grant is
            # created and the task is BLOCKED until a human grants it — not run,
            # not auto-allowed. On grant, the grant API re-queues the task.
            if _board_task_blocked_pending_approval(db, task_id, agent_id, workspace_id):
                heartbeat.cancel()
                db.close()
                return

            from modules.agents.factory.agent_factory import AgentFactory

            factory = AgentFactory(db_session=db)
            exec_result = await factory.execute_with_prompt(
                agent=agent_id,
                prompt=prompt,
                context={
                    "source": "board_task",
                    "task_id": task_id,
                    "workspace_id": workspace_id,
                },
                use_memory=False,
                attachment_ids=attachment_ids,  # PRD-127
            )

            # PRD-171 F023: the executor reports its true terminal status. A
            # run that failed (e.g. the loop raised) returns {"status":"error"}
            # — closing that as 'done' masks the failure. Fail honestly, exactly
            # as the crash-handler below and the mission path already do.
            exec_status = (exec_result or {}).get("status")

            # Extract response text
            llm_text = (
                exec_result.get("result")
                or exec_result.get("response")
                or exec_result.get("output")
                or exec_result.get("content")
                or ""
            )

            task = db.query(BoardTask).get(task_id)
            if task and task.status == "in_progress":
                if exec_status == "error":
                    task.status = "failed"
                    task.error_message = str(
                        exec_result.get("error") or "Agent execution failed"
                    )[:500]
                    task.completed_at = datetime.now(timezone.utc)
                    db.commit()
                    await _dispatch_task_failed(db, workspace_id, task)
                    # Surface failures the same way successes are surfaced.
                    await _auto_create_task_report(
                        db, workspace_id, task,
                        {"result": "", "tokens_used": 0},
                    )
                    db.commit()
                else:
                    task.result = str(llm_text) if llm_text else None
                    task.status = "done" if review_mode == "auto" else "review"
                    task.completed_at = datetime.now(timezone.utc)
                    # PRD-128: dispatch task_complete only on terminal 'done'
                    if task.status == "done":
                        await _dispatch_task_complete(db, workspace_id, task)
                    # Persist a report row for every completed task so it surfaces
                    # in Reports / Deliverables / Activity Feed (mirrors heartbeats).
                    await _auto_create_task_report(db, workspace_id, task, exec_result)
                    db.commit()

            logger.info(
                "[BoardTasks] Agent %d completed task %d → %s",
                agent_id, task_id, task.status if task else "?",
            )
        except Exception as e:
            logger.error(
                "[BoardTasks] Task %d execution failed: %s", task_id, e, exc_info=True,
            )
            record_error(
                subsystem="board",
                operation="execute_task",
                error=e,
                workspace_id=workspace_id,
                agent_id=agent_id,
                extra={"task_id": task_id},
            )
            try:
                task = db.query(BoardTask).get(task_id)
                if task and task.status == "in_progress":
                    # PRD-161 S3: fail honestly — a crashed execution becomes
                    # terminal 'failed', not a silent 'done' with an error blob.
                    task.status = "failed"
                    task.error_message = str(e)[:500]
                    task.completed_at = datetime.now(timezone.utc)
                    db.commit()
                    await _dispatch_task_failed(db, workspace_id, task)
                    # Surface failures the same way successes are surfaced.
                    await _auto_create_task_report(
                        db, workspace_id, task,
                        {"result": "", "tokens_used": 0},
                    )
                    db.commit()
            except Exception:
                db.rollback()
        finally:
            # PRD-171 F024: stop heartbeating the moment the run ends — the task
            # has reached its terminal state, so the lease should now be allowed
            # to lapse for any genuinely-abandoned row.
            heartbeat.cancel()
            try:
                await heartbeat
            except (asyncio.CancelledError, Exception):
                pass
            db.close()

    # Guarded launch: a strong ref prevents GC-cancellation mid-run and an
    # uncaught crash is recorded. _run() already records its own caught
    # failures; the boot reaper (W1-S6) recovers any row stranded by a restart.
    launch_guarded(
        _run(),
        subsystem="board",
        operation="execute_task",
        workspace_id=workspace_id,
        agent_id=agent_id,
        extra={"task_id": task_id},
    )


# ── Planning mode ────────────────────────────────────────────────────

# Tight pack budget for board planning: enough to ground questions in workspace
# knowledge + prior failures without dominating a question-generation prompt.
_BOARD_PLANNING_PACK_TOKENS = 4000


async def _board_planning_context(db: Session, workspace_id, goal: str) -> str:
    """The ONE platform planning pack (PRD-164 S1, Q61) for board planning.

    Same assembler as MissionPlanner and AutoBrain —
    ``ContextService.build_planning_context``. Empty string on any failure so
    planning never breaks because context assembly did.
    """
    try:
        from modules.context.service import ContextService

        pack = await ContextService(db).build_planning_context(
            goal=goal,
            workspace_id=str(workspace_id),
            max_tokens=_BOARD_PLANNING_PACK_TOKENS,
        )
        return pack.content if not pack.is_empty else ""
    except Exception:
        logger.warning(
            "[BoardTasks] planning context pack unavailable — continuing without it",
            exc_info=True,
        )
        return ""


@router.post("/plan", dependencies=[Depends(require_workspace_permission("missions:create"))])
async def plan_task(
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Start planning mode: accepts a raw prompt, returns 3-4 clarifying
    multiple-choice questions from the orchestrator LLM.
    """
    body = await request.json()
    raw_prompt = (body.get("raw_prompt") or "").strip()
    if not raw_prompt:
        raise HTTPException(status_code=422, detail="raw_prompt is required")

    from core.llm.manager import LLMManager

    llm = LLMManager(
        service_name="orchestrator",
        workspace_id=str(ctx.workspace_id),
        request_type="planning",
    )

    system = (
        "You are a task planning assistant. The user wants to create a task for an AI agent.\n"
        "Generate exactly 3-4 multiple choice questions to clarify the task scope.\n\n"
        "Return JSON ONLY in this format:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "id": "q1",\n'
        '      "question": "What is the scope?",\n'
        '      "options": ["Option A", "Option B", "Option C"],\n'
        '      "default": 0\n'
        "    }\n"
        "  ],\n"
        '  "suggested_title": "A clear task title",\n'
        '  "suggested_priority": "medium"\n'
        "}"
    )

    # PRD-164 S1 (Q61): ground the questions in what the platform knows —
    # workspace knowledge, prior mission failures, roster.
    messages = [{"role": "system", "content": system}]
    planning_context = await _board_planning_context(db, ctx.workspace_id, raw_prompt)
    if planning_context:
        messages.append({
            "role": "system",
            "content": (
                f"{planning_context}\n\n"
                "Use this platform context: ground your questions in the "
                "workspace's actual documents and agents, and if similar work "
                "previously failed, ask questions that steer the task away "
                "from the failed approach."
            ),
        })
    messages.append({"role": "user", "content": f"Plan this task: {raw_prompt}"})

    response = await llm.generate_response(messages=messages)

    # Extract text from response
    text = _extract_llm_text(response)

    # Parse JSON from the response
    try:
        parsed = _parse_json_from_text(text)
    except ValueError:
        logger.warning("[BoardTasks] Could not parse planning JSON: %s", text[:500])
        parsed = {
            "questions": [],
            "suggested_title": raw_prompt[:100],
            "suggested_priority": "medium",
            "raw_response": text,
        }

    return {"planning": parsed, "raw_prompt": raw_prompt}


@router.post("/plan/refine", dependencies=[Depends(require_workspace_permission("missions:create"))])
async def refine_task(
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Refine a task: accepts raw_prompt + answers to planning questions,
    returns a refined description + suggested title + priority.
    """
    body = await request.json()
    raw_prompt = (body.get("raw_prompt") or "").strip()
    answers = body.get("answers", [])

    if not raw_prompt:
        raise HTTPException(status_code=422, detail="raw_prompt is required")

    from core.llm.manager import LLMManager

    llm = LLMManager(
        service_name="orchestrator",
        workspace_id=str(ctx.workspace_id),
        request_type="planning",
    )

    system = (
        "You are a task planning assistant. Based on the user's task description and their answers "
        "to clarifying questions, generate a refined, clear task description that an AI agent can execute.\n\n"
        "Return JSON ONLY:\n"
        "{\n"
        '  "title": "Clear task title",\n'
        '  "description": "Detailed task description with specific instructions based on the answers",\n'
        '  "priority": "medium",\n'
        '  "suggested_tags": ["tag1", "tag2"]\n'
        "}"
    )

    answers_text = "\n".join(
        [f"Q: {a.get('question', '')} -> A: {a.get('answer', '')}" for a in answers]
    )

    response = await llm.generate_response(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Original request: {raw_prompt}\n\nAnswers:\n{answers_text}"},
        ]
    )

    text = _extract_llm_text(response)

    try:
        parsed = _parse_json_from_text(text)
    except ValueError:
        logger.warning("[BoardTasks] Could not parse refine JSON: %s", text[:500])
        parsed = {
            "title": raw_prompt[:100],
            "description": raw_prompt,
            "priority": "medium",
            "suggested_tags": [],
            "raw_response": text,
        }

    return {"refined": parsed, "raw_prompt": raw_prompt}


# ── Internal helpers ─────────────────────────────────────────────────

def _extract_llm_text(response) -> str:
    """Pull plain text out of an LLM response (handles LLMResponse, dict, or string)."""
    # Handle LLMResponse objects (have .content attribute)
    if hasattr(response, "content"):
        return str(response.content or "")

    if isinstance(response, dict):
        text = (
            response.get("content")
            or response.get("result")
            or response.get("response")
            or response.get("output")
            or ""
        )
        if isinstance(text, dict):
            text = text.get("content") or str(text)
        # Handle choices array (OpenAI format)
        if not text and "choices" in response:
            choices = response["choices"]
            if choices and isinstance(choices, list):
                msg = choices[0].get("message", {})
                text = msg.get("content", "")
        return str(text)
    return str(response)


def _parse_json_from_text(text: str) -> dict:
    """Extract JSON from LLM text that may contain markdown fences."""
    cleaned = text.strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in response")
