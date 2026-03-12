"""
Board Tasks API
===============

CRUD + planning endpoints for the lightweight task board (PRD-72).
Tasks follow a Kanban lifecycle: inbox -> assigned -> in_progress -> review -> done.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.database.database import get_db
from core.models.core import BoardTask
from core.models import Agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/tasks", tags=["board-tasks"])

VALID_STATUSES = {"inbox", "assigned", "in_progress", "review", "done"}
VALID_PRIORITIES = {"urgent", "high", "medium", "low"}
VALID_REVIEW_MODES = {"human", "llm", "auto"}


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

    status = "assigned" if assigned_agent_id else "inbox"

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
        created_by_id=ctx.user.clerk_user_id or ctx.user.user_id,
        parent_task_id=body.get("parent_task_id"),
        tags=body.get("tags", []),
        planning_data=body.get("planning_data"),
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
        task.status = new_status
        if new_status == "in_progress" and not task.started_at:
            task.started_at = datetime.now(timezone.utc)
        if new_status in ("done", "review"):
            task.completed_at = datetime.now(timezone.utc)

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
    )

    db.commit()
    db.refresh(task)

    if trigger_execution:
        _launch_task_execution(
            task_id=task.id,
            agent_id=task.assigned_agent_id,
            workspace_id=str(ctx.workspace_id),
            prompt=task.raw_prompt or task.description or task.title,
            review_mode=task.review_mode or "auto",
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

    task.status = new_status
    if new_status == "in_progress" and not task.started_at:
        task.started_at = datetime.now(timezone.utc)
    if new_status in ("done", "review") and not task.completed_at:
        task.completed_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(task)

    # Fire-and-forget: trigger agent execution when moved to in_progress
    if new_status == "in_progress" and task.assigned_agent_id:
        _launch_task_execution(
            task_id=task.id,
            agent_id=task.assigned_agent_id,
            workspace_id=str(ctx.workspace_id),
            prompt=task.raw_prompt or task.description or task.title,
            review_mode=task.review_mode or "auto",
        )

    return {"id": task.id, "status": task.status}


# ── Immediate execution (fire-and-forget) ────────────────────────────

def _launch_task_execution(
    task_id: int,
    agent_id: int,
    workspace_id: str,
    prompt: str,
    review_mode: str = "auto",
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
                task.result = str(llm_text)[:4000] if llm_text else None
                task.status = "done" if review_mode == "auto" else "review"
                task.completed_at = datetime.now(timezone.utc)
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
    """Pull plain text out of an LLM response (handles dict or string)."""
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
