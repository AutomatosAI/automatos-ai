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
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.database.database import get_db
from core.models.core import BoardTask
from core.models import Agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/tasks", tags=["board-tasks"])

VALID_STATUSES = {"inbox", "assigned", "in_progress", "review", "blocked", "done"}
VALID_PRIORITIES = {"urgent", "high", "medium", "low"}
VALID_REVIEW_MODES = {"human", "llm", "auto"}

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
        from services.report_service import ReportService

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

        usage = exec_result.get("usage") or {}
        tokens = (
            usage.get("total_tokens")
            or exec_result.get("tokens_used")
            or 0
        )

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
        lines.append("## Metrics")
        lines.append(f"- Tokens used: {tokens}")
        lines.append(f"- Duration: {task.duration_seconds() if hasattr(task, 'duration_seconds') else 'n/a'}")
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
            metrics={
                "tokens_used": tokens,
                "task_id": task.id,
                "task_status": task.status,
            },
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


# ── Helpers ──────────────────────────────────────────────────────────

def _enrich_with_agents(tasks: list, db: Session) -> list:
    """Join agent info onto task dicts."""
    agent_ids = {t.assigned_agent_id for t in tasks if t.assigned_agent_id}
    if not agent_ids:
        return [t.to_dict() for t in tasks]

    agents = {a.id: a for a in db.query(Agent).filter(Agent.id.in_(agent_ids)).all()}

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

@router.post("")
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

    logger.info("[BoardTasks] Created task %d in workspace %s", task.id, ctx.workspace_id)
    return task.to_dict()


@router.get("")
async def list_tasks(
    ctx: RequestContext = Depends(get_request_context_hybrid),
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

    total = query.count()
    tasks = (
        query.order_by(BoardTask.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "tasks": _enrich_with_agents(tasks, db),
        "total": total,
    }


@router.get("/{task_id}")
async def get_task(
    task_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get a single board task."""
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    enriched = _enrich_with_agents([task], db)
    return enriched[0]


@router.patch("/{task_id}")
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
    trigger_execution = (
        "status" in body
        and body["status"] == "in_progress"
        and task.assigned_agent_id
        and task.source_type != 'recipe'  # Recipe executor handles its own execution
    )

    # PRD-128: dispatch task_complete on terminal transition
    if "status" in body and body["status"] == "done":
        await _dispatch_task_complete(db, ctx.workspace_id, task)

    db.commit()
    db.refresh(task)

    if trigger_execution:
        _launch_task_execution(
            task_id=task.id,
            agent_id=task.assigned_agent_id,
            workspace_id=str(ctx.workspace_id),
            prompt=task.raw_prompt or task.description or task.title,
            review_mode=task.review_mode or "auto",
            attachment_ids=task.attachment_ids,  # PRD-127
        )

    logger.info("[BoardTasks] Updated task %d", task.id)
    return task.to_dict()


@router.delete("/{task_id}")
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

@router.post("/{task_id}/approve")
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


@router.post("/{task_id}/reject")
async def reject_task(
    task_id: int,
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Reject a board task in review status with optional feedback.
    Moves task back to inbox with feedback stored in error_message.
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
    feedback = (body.get("feedback") or "").strip()

    task.status = "inbox"
    task.started_at = None
    task.completed_at = None
    if feedback:
        task.error_message = f"Rejected: {feedback}"

    db.commit()
    db.refresh(task)

    logger.info("[BoardTasks] Task %d rejected%s", task.id, f" with feedback: {feedback}" if feedback else "")
    return {
        "success": True,
        "task_id": task.id,
        "status": task.status,
        "feedback": feedback or None,
    }


@router.patch("/{task_id}/status")
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

    # Fire-and-forget: trigger agent execution when moved to in_progress
    if new_status == "in_progress" and task.assigned_agent_id and task.source_type != 'recipe':
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
        try:
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
            try:
                task = db.query(BoardTask).get(task_id)
                if task:
                    task.status = "inbox"
                    task.error_message = str(e)[:500]
                    task.started_at = None
                    db.commit()
                    # Surface failures the same way successes are surfaced.
                    await _auto_create_task_report(
                        db, workspace_id, task,
                        {"result": "", "tokens_used": 0},
                    )
                    db.commit()
            except Exception:
                db.rollback()
        finally:
            db.close()

    asyncio.create_task(_run())


# ── Planning mode ────────────────────────────────────────────────────

@router.post("/plan")
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

    response = await llm.generate_response(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Plan this task: {raw_prompt}"},
        ]
    )

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


@router.post("/plan/refine")
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
