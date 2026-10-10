"""Scheduling handlers for PlatformActionExecutor (PRD-77) + NL2SQL query_data (PRD-79)."""

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from modules.tools.discovery.agent_refs import resolve_active_agent, takes_the_agent_id
from modules.tools.discovery.brief_sends import REVIEW_MODE, reviewed_by_a_person, says_it_is_reviewed

logger = logging.getLogger(__name__)


# F300/F301: what the database tool adds beside the rows (modules.nl2sql.agent_answer).
ANSWER_CONTEXT_KEYS = ("schema", "notes", "derived", "not_recorded")
NO_ROWS = "Query returned no rows."
ANSWER_ROWS_SHOWN = 50
ANSWER_CELL_CHARS = 50
ANSWER_HEADER_CHARS = 20
# PRD-256 FX-013: the refusal names the key it wants (night 12 sent the question as "query" 31 times).
MISSING_QUESTION = ("Missing required parameter: question. Send the user's question in 'question' "
                    "(or 'query'), e.g. {\"question\": \"How many active subscriptions do we have?\"}. Nothing ran.")
# platform_schedule_task: the target's key, and the board ticket's defaults.
TARGET_NAME = "target_agent_name"
NOT_IN_WORKSPACE = "Agent '{said}' not found in workspace"
DEFAULT_TITLE = "Scheduled task"
TITLE_CHARS = 255
DEFAULT_PRIORITY = "medium"
DEFAULT_REVIEW = "auto"


def answer_table(columns: List[Any], rows: List[Dict[str, Any]], row_count: int) -> str:
    """The rows as a readable table, at most ``ANSWER_ROWS_SHOWN`` of them."""
    if not columns or not rows:
        return ""
    header = " | ".join(str(c) for c in columns)
    separator = "-+-".join("-" * min(len(str(c)), ANSWER_HEADER_CHARS) for c in columns)
    rows_text = "\n".join(
        " | ".join(str(row.get(c, ""))[:ANSWER_CELL_CHARS] for c in columns) for row in rows
    )
    table_text = f"{header}\n{separator}\n{rows_text}"
    if row_count > ANSWER_ROWS_SHOWN:
        table_text += f"\n... ({row_count - ANSWER_ROWS_SHOWN} more rows)"
    return table_text


@takes_the_agent_id(TARGET_NAME)  # P256-FIX-RVW-23: agent_id beside the name; an id wins
async def schedule_task(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Schedule a follow-up task for self or another agent.

    P256-FIX-RVW-23: the target is an ACTIVE agent, by id or by a name one active agent
    carries (a clash lists the candidates); a board ticket whose brief sends or orders,
    scheduled from a person's chat, is filed for their review (FX-010, Decision D7)."""
    from services.scheduled_task_service import DELIVER_BOARD_TASK, DELIVER_CHAT, ScheduledTaskService

    task_type = params.get("task_type")
    description = params.get("description")
    schedule = params.get("schedule")

    if not task_type or not description or not schedule:
        return {"success": False, "error": "task_type, description, and schedule are required"}

    # Resolve calling agent
    created_by_agent_id = params.get("_agent_id")
    if not created_by_agent_id:
        return {"success": False, "error": "Could not determine calling agent"}

    deliver_as = params.get("deliver_as") or DELIVER_CHAT
    if deliver_as not in (DELIVER_CHAT, DELIVER_BOARD_TASK):
        return {"success": False, "error": f"deliver_as must be '{DELIVER_CHAT}' or '{DELIVER_BOARD_TASK}'"}

    # Chat delivery defaults to self; a board ticket with no named agent is filed
    # unassigned (Inbox), never silently self-assigned.
    target_agent_id, refusal = _target_agent(
        db, workspace_id, params, created_by_agent_id if deliver_as == DELIVER_CHAT else None)
    if refusal:
        return {"success": False, "error": refusal}

    held = False
    payload = None
    if deliver_as == DELIVER_BOARD_TASK:
        params, held = reviewed_by_a_person(params)  # FX-010 (D7): a brief that sends or orders
        payload = _ticket(params, description)

    svc = ScheduledTaskService(db, workspace_id)
    result = await svc.create_task(
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
    return _said_reviewed(result, payload, held)


def _target_agent(db: Session, workspace_id: UUID, params: Dict[str, Any],
                  default: Any) -> Tuple[Any, Optional[str]]:
    """(the agent the task runs as, None) or (None, why not): the id ``takes_the_agent_id``
    bound, or the one ACTIVE agent carrying the name (P256-FIX-RVW-23: never a switched-off
    namesake, never the first of several); ``default`` when none is named."""
    said = params.get(TARGET_NAME)
    if said in (None, ""):
        return default, None
    agent, refusal = resolve_active_agent(db, workspace_id, said)
    if agent is None:
        return None, refusal or NOT_IN_WORKSPACE.format(said=said)
    return agent.id, None


def _ticket(params: Dict[str, Any], description: Any) -> Dict[str, Any]:
    """The board ticket a ``board_task`` row files when it fires."""
    title = params.get("title") or (str(description).strip().splitlines() or [DEFAULT_TITLE])[0]
    return {
        "title": title[:TITLE_CHARS],
        "priority": params.get("priority") or DEFAULT_PRIORITY,
        REVIEW_MODE: params.get(REVIEW_MODE) or DEFAULT_REVIEW,
        "tags": [str(t) for t in (params.get("tags") or []) if t],
    }


def _said_reviewed(result: Any, payload: Optional[Dict[str, Any]], held: bool) -> Any:
    """The answer, saying the ticket waits for the owner's review when ``reviewed_by_a_person`` held it."""
    if not held or not isinstance(result, dict) or not result.get("success"):
        return result
    return says_it_is_reviewed({**result, REVIEW_MODE: (payload or {}).get(REVIEW_MODE)}, held)


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

    F300/F301 (night 9): board agents reach this through ``platform_execute``
    too (28 of the night's Decimal failures). The database's schema and any
    note on dates past the data go out with the answer, as they do from
    ``smart_query_database``.
    """
    question = params.get("question")
    if not question or not str(question).strip():
        return {"success": False, "error": MISSING_QUESTION}

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
        context = {key: result[key] for key in ANSWER_CONTEXT_KEYS if result.get(key)}
        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "Query execution failed"),
                "sql": result.get("sql"),
                **context,
            }

        # Format for agent consumption
        data = result.get("data", [])
        columns = result.get("columns", [])
        # F301 B1: a date past the data has no count to give, not a count of 0.
        row_count = None if result.get("not_recorded") else result.get("row_count", len(data))
        display_rows = data[:ANSWER_ROWS_SHOWN]

        return {
            "success": True,
            "answer": result.get("answer") or answer_table(columns, display_rows, row_count) or NO_ROWS,
            "sql": result.get("sql"),
            "row_count": row_count,
            "columns": columns,
            "data": display_rows,
            "explanation": result.get("explanation"),
            "confidence": result.get("confidence"),
            **context,
        }

    except Exception as e:
        # The request session is shared with every other tool in this turn:
        # never hand it back aborted.
        db.rollback()
        logger.error("[PlatformExecutor] query_data failed: %s", e, exc_info=True)
        return {"success": False, "error": "Database query failed."}
