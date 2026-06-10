"""
PRD-139: Universal tool execution telemetry hook.

Non-blocking telemetry writer that logs every tool execution to
tool_execution_logs regardless of dispatch path (platform, workspace,
composio, MCP).  Failure to write telemetry MUST NOT fail the tool call.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def write_telemetry(
    db: Session,
    *,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: Optional[int],
    workspace_id: Optional[UUID],
    result: Dict[str, Any],
    execution_time_ms: int,
    caller_context: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a single telemetry row to tool_execution_logs.

    This function is designed to be fired-and-forgotten via asyncio.create_task.
    It catches all exceptions internally so it never propagates failures.
    """
    try:
        from core.models.composio_cache import ToolExecutionLog

        ctx = caller_context or {}

        # Determine app_name from tool_name prefix
        if tool_name.startswith("platform_"):
            app_name = "PLATFORM"
        elif tool_name.startswith("workspace_"):
            app_name = "WORKSPACE"
        elif tool_name.startswith("composio_") or tool_name.startswith("COMPOSIO_"):
            app_name = tool_name.split("_")[0].upper()
        else:
            app_name = tool_name.split("_")[0].upper() if "_" in tool_name else tool_name.upper()

        # Resolve agent_id: use None for non-agent calls (never write 0 with FK)
        resolved_agent_id = agent_id if agent_id and agent_id > 0 else None

        status = "success" if result.get("success", result.get("successful")) else "error"

        log_entry = ToolExecutionLog(
            agent_id=resolved_agent_id,
            app_name=app_name[:100],
            action_name=tool_name[:255],
            workspace_id=workspace_id,
            user_id=ctx.get("user_id"),
            input_parameters={"keys": list(parameters.keys())} if isinstance(parameters, dict) else {},
            user_query=ctx.get("user_query"),
            status=status,
            error_message=result.get("error") if status == "error" else None,
            execution_time_ms=execution_time_ms,
            router_decision=_build_router_decision(
                ctx, autonomous=result.get("autonomous") is True
            ),
            intent_cluster_id=ctx.get("intent_cluster_id"),
            routing_source=ctx.get("routing_source"),
            telemetry_source=ctx.get("telemetry_source", "production"),
        )

        db.add(log_entry)
        db.commit()
    except Exception as exc:
        logger.debug(f"[telemetry] Failed to write tool execution log: {exc}")
        try:
            db.rollback()
        except Exception:
            pass


def _build_router_decision(
    ctx: Dict[str, Any], autonomous: bool = False
) -> Optional[Dict[str, Any]]:
    """Extract routing metadata from caller_context into router_decision JSONB."""
    decision = {}
    if autonomous:
        # PRD-143: confirmation was skipped by the full-autonomy dial — the
        # distinct, queryable audit marker (router_decision->>'autonomous').
        decision["autonomous"] = True
    if ctx.get("routing_candidates"):
        decision["candidates"] = ctx["routing_candidates"]
    if ctx.get("routing_chain_hints"):
        decision["chain_hints"] = ctx["routing_chain_hints"]
    if ctx.get("conversation_id"):
        decision["conversation_id"] = ctx["conversation_id"]
    if ctx.get("turn_id"):
        decision["turn_id"] = ctx["turn_id"]
    return decision if decision else None


def fire_telemetry(
    db: Session,
    *,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: Optional[int],
    workspace_id: Optional[UUID],
    result: Dict[str, Any],
    execution_time_ms: int,
    caller_context: Optional[Dict[str, Any]] = None,
) -> None:
    """Fire-and-forget telemetry write as a background task.

    Safe to call from sync or async context -- schedules the write on the
    running event loop.  If no loop is running, logs a warning and skips.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(
            write_telemetry(
                db,
                tool_name=tool_name,
                parameters=parameters,
                agent_id=agent_id,
                workspace_id=workspace_id,
                result=result,
                execution_time_ms=execution_time_ms,
                caller_context=caller_context,
            )
        )
    except RuntimeError:
        # No running event loop -- skip telemetry silently
        logger.debug("[telemetry] No event loop available, skipping telemetry write")
