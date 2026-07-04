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


def resolve_action_name(tool_name: str, parameters: Dict[str, Any]) -> str:
    """Resolve the effective action name for telemetry (PRD-177 S1 / F016).

    The composio meta-tool is always dispatched as ``composio_execute`` with the
    real action carried in ``parameters['action']`` (or ``'action_name'``). Left
    unresolved, every one of the 856-app surface's actions collapses to a single
    ``composio_execute`` node and the routing graph never learns per-action
    co-occurrence or success. This resolves the real action (e.g.
    ``SLACK_SEND_MESSAGE``), normalized to the canonical uppercase form used by
    the executor, so the graph learns the true node.

    Privacy: only the ``action`` *identifier* is read — never secret param
    *values* (the keys-only posture in write_telemetry is unchanged).

    Non-composio tools (``platform_*``, ``workspace_*``, per-action composio
    tools whose own name IS the action) pass through unchanged.
    """
    if tool_name != "composio_execute":
        return tool_name
    if not isinstance(parameters, dict):
        return tool_name
    raw_action = parameters.get("action") or parameters.get("action_name")
    if not raw_action:
        return tool_name  # malformed call — keep the meta-tool name, never crash
    return str(raw_action).upper().strip()


# clerk_user_id (str) -> users.id (int). Stable mapping; cached to keep the
# telemetry write off a per-call User lookup on the hot path.
_CLERK_ID_CACHE: Dict[str, int] = {}
_CLERK_ID_CACHE_MAX = 5000


def _coerce_user_id(db: Session, raw: Any) -> Optional[int]:
    """Coerce a caller_context ``user_id`` into the integer ``users.id`` the column expects.

    The chat lane threads a Clerk *string* principal (``driving_clerk``) while
    ``ToolExecutionLog.user_id`` is an ``Integer`` column — binding the raw string
    makes every logged-in tool call's INSERT fail, which is exactly why the table
    carried 0 organic rows across 21 workspaces and the whole learning plane
    starved. Pass ints through; resolve a Clerk id to ``users.id`` (cached); return
    ``None`` when unresolvable so the row STILL lands (a null-user row beats no row).
    """
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    s = str(raw).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    cached = _CLERK_ID_CACHE.get(s)
    if cached is not None:
        return cached
    try:
        from core.models.core import User

        row = db.query(User.id).filter(User.clerk_user_id == s).first()
    except Exception:
        return None
    if not row:
        return None
    uid = row[0]
    if len(_CLERK_ID_CACHE) < _CLERK_ID_CACHE_MAX:
        _CLERK_ID_CACHE[s] = uid
    return uid


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

        # PRD-177 S1 (F016): resolve the real composio action so it is not
        # collapsed to a single ``composio_execute`` node in the routing graph.
        action_name = resolve_action_name(tool_name, parameters)

        # Determine app_name from the RESOLVED action's prefix (e.g. the resolved
        # SLACK_SEND_MESSAGE yields app SLACK, not COMPOSIO).
        if action_name.startswith("platform_"):
            app_name = "PLATFORM"
        elif action_name.startswith("workspace_"):
            app_name = "WORKSPACE"
        else:
            app_name = action_name.split("_")[0].upper() if "_" in action_name else action_name.upper()

        # Resolve agent_id: use None for non-agent calls (never write 0 with FK)
        resolved_agent_id = agent_id if agent_id and agent_id > 0 else None

        status = "success" if result.get("success", result.get("successful")) else "error"

        log_entry = ToolExecutionLog(
            agent_id=resolved_agent_id,
            app_name=app_name[:100],
            action_name=action_name[:255],
            workspace_id=workspace_id,
            user_id=_coerce_user_id(db, ctx.get("user_id")),
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
        # Loud on purpose (PRD-185 S1): a DEBUG swallow here hid a 2-month,
        # 0-organic-rows telemetry outage. Failures must be visible.
        logger.warning(f"[telemetry] Failed to write tool execution log: {exc}")
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
    sel = ctx.get("selection_outcome")
    if isinstance(sel, dict) and sel:
        # PRD-143 S14: per-dispatch selection outcome (narrowed/hit/fallback)
        # — the durable rows behind the su-locked selection-health metric.
        decision["selection"] = sel
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
