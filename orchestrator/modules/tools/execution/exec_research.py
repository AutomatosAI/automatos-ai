"""Database / research tool executors — DISABLED pending PRD-160 (PRD-156 S3).

The natural-language-to-SQL tools — ``query_database`` (execute_database_tool)
and ``smart_query_database`` (execute_smart_database_tool) — executed raw
LLM-generated SQL against the ENTIRE main database with NO workspace filter (a
confirmed cross-tenant leak), and ``execute_database_tool`` additionally made
unauthenticated HTTP self-calls to the knowledge API. PRD-156 S3 turns the
unsafe path OFF (not shimmed); PRD-160 rebuilds a workspace-scoped, in-process
path.

Both executors now return a structured disabled response, and the deleted
``query_main_database`` helper (which ran the unscoped SQL) is gone. These tools
are also removed from the chat tool surface (intent_classifier / service.py) so
the LLM never selects them — this executor-level disable is the defense-in-depth
backstop if the path is reached another way (e.g. a Playbook step).
"""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_DISABLED_RESPONSE: Dict[str, Any] = {
    "success": False,
    "disabled": True,
    "error": (
        "Natural-language database querying is temporarily disabled pending "
        "workspace-scoped re-enablement (PRD-160)."
    ),
    "data": [],
    "columns": [],
    "row_count": 0,
}


async def execute_database_tool(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
) -> Dict[str, Any]:
    """DISABLED (PRD-156 S3). Previously: unauthenticated HTTP self-calls to the
    knowledge API + ``query_main_database`` fallback (raw unscoped SQL on the
    main DB). Re-enabled, workspace-scoped and in-process, by PRD-160."""
    logger.warning(
        "Agent %s invoked disabled database tool '%s' "
        "(PRD-156 S3 disabled NL2SQL; PRD-160 re-enables it scoped)",
        agent_id,
        tool_name,
    )
    return dict(_DISABLED_RESPONSE)


async def execute_smart_database_tool(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
) -> Dict[str, Any]:
    """DISABLED (PRD-156 S3). Previously: SmartNL2SQLAgent generated SQL that was
    executed against the main DB via a fresh SessionLocal with no workspace
    scope. Re-enabled, workspace-scoped and in-process, by PRD-160."""
    logger.warning(
        "Agent %s invoked disabled smart database tool '%s' "
        "(PRD-156 S3 disabled NL2SQL; PRD-160 re-enables it scoped)",
        agent_id,
        tool_name,
    )
    return dict(_DISABLED_RESPONSE)
