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


# F077 (A, refresh 4): the person's own words for this turn go to NL2SQL beside Auto's
# restatement. Auto turned "How many subscribers ... not counting cancelled ones?" into
# "How many ACTIVE subscribers ...", and the SQL counted active only (383, not 400).
OWNER_WORDS_CHARS = 2000


def owner_words(caller_context: Optional[Dict[str, Any]]) -> Optional[str]:
    """What the person typed this turn (``user_query``, set by the chat server-side;
    never a tool argument), bounded, or None for a lane nobody typed into."""
    if not isinstance(caller_context, dict):
        return None
    return str(caller_context.get("user_query") or "").strip()[:OWNER_WORDS_CHARS] or None


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


async def run_nl2sql(
    *,
    method: str,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id: Optional[Any],
    caller_context: Optional[Dict[str, Any]],
    db_session: Optional[Any] = None,
) -> Dict[str, Any]:
    """Shared in-process NL2SQL invocation for every database tool —
    ``query_database``, ``smart_query_database`` and ``platform_query_data``
    (F077): one service, one resolver, one audit row.

    ``method`` is ``"smart_query"`` (intelligent router) or ``"query_database"``
    (direct). Resolution and execution are workspace-scoped end to end;
    ``database_name`` may be a source's name or its id.
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
        available = await _available_sources(service, ws_id, db_session)
        if database_name not in (None, ""):
            return _error(
                f"No active database source named '{str(database_name)[:100]}' is available "
                f"in this workspace.{available}"
            )
        return _error(
            "No database source is configured for this workspace, or several are "
            f"and none was named — pass 'database_name' to choose one.{available}"
        )

    agent = str(agent_id) if agent_id is not None else None
    owner_question = owner_words(caller_context)
    try:
        if method == "smart_query":
            result = await service.smart_query(
                source_id=source_id,
                text=query,
                user_id=user_id,
                agent_id=agent,
                workspace_id=ws_id,
                owner_question=owner_question,
            )
        else:
            result = await service.query_database(
                source_id=source_id,
                natural_language_query=query,
                user_id=user_id,
                agent_id=agent,
                workspace_id=ws_id,
                owner_question=owner_question,
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

    # PRD-160 S4: every NL query lands one audit row (best-effort).
    try:
        await service.write_nl_audit(
            source_id=source_id,
            user_id=user_id or None,
            agent_id=agent,
            nl_query=query,
            result=result if isinstance(result, dict) else {},
        )
    except Exception:  # noqa: BLE001
        pass
    return result


async def _available_sources(service: Any, ws_id: str, db_session: Optional[Any]) -> str:
    """The ``Available: a (#36), b (#37).`` suffix, so the model can name a
    source instead of asking the owner which database. Best-effort: empty on
    any failure."""
    try:
        sources = await service.active_sources(ws_id, db_session=db_session)
    except Exception:  # noqa: BLE001 — the message is a courtesy, never the failure
        return ""
    if not sources:
        return ""
    return " Available: " + ", ".join(f"{name} (#{sid})" for sid, name in sources[:10]) + "."


async def execute_database_tool(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id: Optional[Any] = None,
    caller_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """``query_database`` — direct NL→SQL, workspace-scoped, in-process (PRD-160 S1)."""
    return await run_nl2sql(
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
    return await run_nl2sql(
        method="smart_query",
        parameters=parameters,
        agent_id=agent_id,
        workspace_id=workspace_id,
        caller_context=caller_context,
        db_session=getattr(executor, "db", None),
    )
