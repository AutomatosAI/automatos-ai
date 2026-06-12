"""Database / research tool executors — workspace-scoped, in-process (PRD-160 S1).

The natural-language-to-SQL tools — ``query_database`` (execute_database_tool)
and ``smart_query_database`` (execute_smart_database_tool) — were turned OFF by
PRD-156 S3 because the old path executed raw LLM-generated SQL against the ENTIRE
main database with no workspace filter (a confirmed cross-tenant leak) and made
unauthenticated HTTP self-calls to the knowledge API.

PRD-160 S1 re-enables them as a first-class Auto tool, but *safely*:

  * In-process — no HTTP self-call. We call ``DatabaseKnowledgeService`` directly
    (the deleted ``query_main_database`` unscoped-SQL helper stays deleted).
  * Workspace-scoped — every call resolves a source *within the caller's
    workspace* (``resolve_source_id``) and threads ``workspace_id`` through to
    ``_get_source``, which fails closed on a cross-workspace source. An agent
    can only ever address sources by name inside its own workspace.
  * Fail-closed — a call with no ``workspace_id`` is refused outright; this is
    the defense-in-depth backstop if the path is reached without scope (e.g. a
    Playbook step).
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _error(message: str, *, disabled: bool = False) -> Dict[str, Any]:
    """Structured, leak-free failure response matching the success shape."""
    resp: Dict[str, Any] = {
        "success": False,
        "error": message,
        "data": [],
        "columns": [],
        "row_count": 0,
    }
    if disabled:
        resp["disabled"] = True
    return resp


async def _run_nl2sql(
    *,
    method: str,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id: Optional[Any],
    caller_context: Optional[Dict[str, Any]],
    db_session: Optional[Any] = None,
) -> Dict[str, Any]:
    """Shared in-process NL2SQL invocation for both database tools.

    ``method`` is ``"smart_query"`` (intelligent router) or ``"query_database"``
    (direct). Resolution and execution are workspace-scoped end to end.
    """
    # Fail-closed: NL2SQL must never run without a workspace scope.
    if not workspace_id:
        logger.warning(
            "Agent %s invoked '%s' without a workspace scope — refused (PRD-160 S1)",
            agent_id,
            method,
        )
        return _error(
            "Natural-language database querying requires a workspace context; "
            "none was supplied."
        )

    query = (parameters or {}).get("query")
    if not query or not str(query).strip():
        return _error("A natural-language 'query' is required.")

    database_name = (parameters or {}).get("database_name")
    ws_id = str(workspace_id)
    user_id = str((caller_context or {}).get("user_id") or "")

    from modules.nl2sql import get_database_knowledge_service

    service = get_database_knowledge_service()

    # Resolve the target source *within the caller's workspace*. Reuse the
    # executor's request session when present (one fewer pooled connection).
    source_id = await service.resolve_source_id(ws_id, database_name, db_session=db_session)
    if not source_id:
        if database_name:
            return _error(
                f"No active database source named '{database_name}' is available "
                "in this workspace."
            )
        return _error(
            "No database source is configured for this workspace, or several are "
            "and none was named — pass 'database_name' to choose one."
        )

    try:
        if method == "smart_query":
            return await service.smart_query(
                source_id=source_id,
                text=query,
                user_id=user_id,
                agent_id=str(agent_id) if agent_id is not None else None,
                workspace_id=ws_id,
            )
        return await service.query_database(
            source_id=source_id,
            natural_language_query=query,
            user_id=user_id,
            agent_id=str(agent_id) if agent_id is not None else None,
            workspace_id=ws_id,
        )
    except Exception as e:  # noqa: BLE001 — surface a safe message, never leak internals
        logger.error(
            "Agent %s NL2SQL '%s' failed (workspace=%s): %s",
            agent_id,
            method,
            ws_id,
            e,
        )
        return _error("Database query failed.")


async def execute_database_tool(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id: Optional[Any] = None,
    caller_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """``query_database`` — direct NL→SQL, workspace-scoped, in-process (PRD-160 S1)."""
    return await _run_nl2sql(
        method="query_database",
        parameters=parameters,
        agent_id=agent_id,
        workspace_id=workspace_id,
        caller_context=caller_context,
        db_session=getattr(executor, "db", None),
    )


async def execute_smart_database_tool(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id: Optional[Any] = None,
    caller_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """``smart_query_database`` — intelligent NL→SQL/analysis router, workspace-scoped,
    in-process (PRD-160 S1)."""
    return await _run_nl2sql(
        method="smart_query",
        parameters=parameters,
        agent_id=agent_id,
        workspace_id=workspace_id,
        caller_context=caller_context,
        db_session=getattr(executor, "db", None),
    )
