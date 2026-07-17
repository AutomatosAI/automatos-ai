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

    # Resolve target agent (default: self)
    target_agent_id = created_by_agent_id
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

    svc = ScheduledTaskService(db, workspace_id)
    return await svc.create_task(
        created_by_agent_id=created_by_agent_id,
        target_agent_id=target_agent_id,
        task_type=task_type,
        description=description,
        schedule=schedule,
        max_runs=params.get("max_runs"),
        # PRD-205 S4: executor-injected origin (anti-spoof -- stripped and
        # re-set from caller_context by the executor) so S6 can deliver the
        # task's output back to the scheduling conversation / user.
        origin_chat_id=params.get("_origin_chat_id"),
        created_by=params.get("_created_by"),
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
    """Query a connected database using natural language."""
    from modules.nl2sql.service import DatabaseKnowledgeService
    from core.models.database_knowledge import DatabaseKnowledgeSource

    question = params.get("question")
    if not question:
        return {"success": False, "error": "question is required"}

    database_id = params.get("database_id")

    try:
        # Resolve database source
        if database_id:
            source = db.query(DatabaseKnowledgeSource).filter(
                DatabaseKnowledgeSource.id == database_id,
                DatabaseKnowledgeSource.workspace_id == workspace_id,
                DatabaseKnowledgeSource.is_active.is_(True),
            ).first()
            if not source:
                return {
                    "success": False,
                    "error": f"Database source {database_id} not found or not active in this workspace",
                }
        else:
            # Use first active database in workspace
            source = db.query(DatabaseKnowledgeSource).filter(
                DatabaseKnowledgeSource.workspace_id == workspace_id,
                DatabaseKnowledgeSource.is_active.is_(True),
            ).order_by(DatabaseKnowledgeSource.id).first()
            if not source:
                return {
                    "success": False,
                    "error": (
                        "No connected databases found. Connect a database first "
                        "via Settings -> Data Sources."
                    ),
                }

        # Execute via DatabaseKnowledgeService
        service = DatabaseKnowledgeService()
        agent_id = str(params.get("_agent_id", "")) or None
        user_id = str(params.get("_user_id", "")) or str(workspace_id)

        result = await service.query_database(
            source_id=str(source.id),
            natural_language_query=question,
            user_id=user_id,
            agent_id=agent_id,
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
            "database": source.name,
        }

    except Exception as e:
        logger.error("[PlatformExecutor] query_data failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Database query failed: {str(e)[:200]}"}
