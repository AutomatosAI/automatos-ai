"""Scheduling handlers for PlatformActionExecutor (PRD-77) + NL2SQL query_data (PRD-79)."""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def schedule_task(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Schedule a follow-up task for self or another agent."""
    from services.scheduled_task_service import ScheduledTaskService

    task_type = params.get("task_type")
    description = params.get("description")
    schedule = params.get("schedule")

    if not task_type or not description or not schedule:
        return {"success": False, "error": "task_type, description, and schedule are required"}

    # Resolve calling agent
    created_by_agent_id = params.get("_agent_id")
    if not created_by_agent_id:
        return {"success": False, "error": "Could not determine calling agent"}

    from services.scheduled_task_service import DELIVER_BOARD_TASK, DELIVER_CHAT
    deliver_as = params.get("deliver_as") or DELIVER_CHAT
    if deliver_as not in (DELIVER_CHAT, DELIVER_BOARD_TASK):
        return {"success": False, "error": f"deliver_as must be '{DELIVER_CHAT}' or '{DELIVER_BOARD_TASK}'"}

    # Resolve target agent. Chat delivery defaults to self; a board ticket with
    # no named agent is filed unassigned (Inbox), never silently self-assigned.
    target_agent_id = created_by_agent_id if deliver_as == DELIVER_CHAT else None
    target_name = params.get("target_agent_name")
    if target_name:
        from core.models import Agent
        target = db.query(Agent).filter(
            Agent.workspace_id == workspace_id,
            func.lower(Agent.name) == target_name.lower(),
        ).first()
        if not target:
            return {"success": False, "error": f"Agent '{target_name}' not found in workspace"}
        target_agent_id = target.id

    payload = None
    if deliver_as == DELIVER_BOARD_TASK:
        payload = {
            "title": (params.get("title") or (str(description).strip().splitlines() or ["Scheduled task"])[0])[:255],
            "priority": params.get("priority") or "medium",
            "review_mode": params.get("review_mode") or "auto",
            "tags": [str(t) for t in (params.get("tags") or []) if t],
        }

    svc = ScheduledTaskService(db, workspace_id)
    return await svc.create_task(
        created_by_agent_id=created_by_agent_id,
        target_agent_id=target_agent_id,
        task_type=task_type,
        description=description,
        schedule=schedule,
        max_runs=params.get("max_runs"),
        # PRD-205 S6: server-injected originating conversation (never an
        # LLM-supplied arg) — the delivered output posts back here.
        origin_chat_id=params.get("_origin_chat_id"),
        deliver_as=deliver_as,
        payload=payload,
        # The driving human behind a chat tool call (server-injected, PRD-234):
        # on the local edition their scheduling IS the consent when the ticket is
        # filed and assigned at fire time. Autonomous runs thread none.
        created_by_user_id=params.get("_user_id"),
    )


async def list_scheduled_tasks(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List scheduled tasks for the workspace."""
    from services.scheduled_task_service import ScheduledTaskService

    # Resolve optional agent_name to agent_id
    agent_id = None
    agent_name = params.get("agent_name")
    if agent_name:
        from core.models import Agent
        agent = db.query(Agent).filter(
            Agent.workspace_id == workspace_id,
            func.lower(Agent.name) == agent_name.lower(),
        ).first()
        if not agent:
            return {"success": False, "error": f"Agent '{agent_name}' not found in workspace"}
        agent_id = agent.id

    svc = ScheduledTaskService(db, workspace_id)
    return await svc.list_tasks(
        agent_id=agent_id,
        status=params.get("status"),
    )


async def cancel_scheduled_task(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Cancel a scheduled task by ID."""
    from services.scheduled_task_service import ScheduledTaskService

    task_id = params.get("task_id")
    if not task_id:
        return {"success": False, "error": "task_id is required"}

    svc = ScheduledTaskService(db, workspace_id)
    return await svc.update_task_status(task_id, "cancelled")


async def get_schedule(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return the workspace's unified schedule — the SAME DB-first truth the
    calendar shows (PRD-162): heartbeat routines, cron-scheduled playbooks, and
    agent-scheduled tasks, each with its next run time. Workspace-scoped."""
    from services.activity_service import ActivityService

    try:
        range_days = int(params.get("range_days") or 30)
    except (TypeError, ValueError):
        range_days = 30

    result = ActivityService(db, workspace_id).get_schedule(range_days=range_days)
    items = result.get("scheduled", [])
    return {
        "success": True,
        "count": len(items),
        "scheduled": items,
        "scheduler_active": result.get("scheduler_active", True),
    }


async def query_data(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Query a connected database using natural language.

    F077: both old paths were broken. A NAME in ``database_id`` was compared
    with the integer id in SQL (InvalidTextRepresentation — and the aborted
    transaction then failed every later tool in the turn), and no
    ``database_id`` built ``DatabaseKnowledgeService()`` with none of its five
    dependencies (TypeError). It now runs the same in-process NL2SQL path as
    ``query_database`` / ``smart_query_database``: the one service construction
    site, workspace-scoped resolution by id OR name, one audit row.
    """
    question = params.get("question")
    if not question or not str(question).strip():
        return {"success": False, "error": "question is required"}

    reference = params.get("database_id")
    if reference is not None and (isinstance(reference, bool) or not isinstance(reference, (int, str))):
        return {"success": False, "error": "database_id must be a database source's id or its name"}

    from modules.tools.execution.exec_research import run_nl2sql

    try:
        result = await run_nl2sql(
            method="query_database",
            parameters={"query": question, "database_name": reference},
            agent_id=params.get("_agent_id"),
            workspace_id=workspace_id,
            caller_context={"user_id": params.get("_user_id")},
            db_session=db,
        )
        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "Query execution failed"),
                "sql": result.get("sql"),
            }

        # Format for agent consumption
        data = result.get("data", [])
        columns = result.get("columns", [])
        row_count = result.get("row_count", len(data))

        # Build readable table (truncate large results)
        display_rows = data[:50]
        table_text = ""
        if columns and display_rows:
            header = " | ".join(str(c) for c in columns)
            separator = "-+-".join("-" * min(len(str(c)), 20) for c in columns)
            rows_text = "\n".join(
                " | ".join(str(row.get(c, ""))[:50] for c in columns)
                for row in display_rows
            )
            table_text = f"{header}\n{separator}\n{rows_text}"
            if row_count > 50:
                table_text += f"\n... ({row_count - 50} more rows)"

        return {
            "success": True,
            "answer": table_text or "Query returned no rows.",
            "sql": result.get("sql"),
            "row_count": row_count,
            "columns": columns,
            "data": display_rows,
            "explanation": result.get("explanation"),
            "confidence": result.get("confidence"),
        }

    except Exception as e:
        # The request session is shared with every other tool in this turn:
        # never hand it back aborted.
        db.rollback()
        logger.error("[PlatformExecutor] query_data failed: %s", e, exc_info=True)
        return {"success": False, "error": "Database query failed."}
