"""
Tool Router — Shared Tool Execution Layer
==========================================

Generic tool execution functions used by ALL consumers (chatbot, recipe,
trigger, API).  Promoted from consumers/chatbot/tool_router.py so that
non-chatbot consumers can import without cross-consumer dependencies.

Handles:
- Getting tools from modules.tools.ToolRegistry
- Executing tools via modules.tools.UnifiedToolExecutor
- Formatting results for frontend and LLM
- Capability-based action filtering (PRD-37)
- Execution-time validation (defense in depth)
"""

import json
import logging
import re
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

# Import from submodules directly to avoid circular import
# (modules/tools/__init__.py imports from this file)
from modules.tools.registry import ToolCategory, get_tool_registry as registry_get_tool_registry
from modules.tools.execution import UnifiedToolExecutor
from modules.tools.formatting.result_formatter import ToolResultFormatter
from core.database.database import SessionLocal

# Capability-based filtering imports (PRD-37)
try:
    from modules.tools.services.action_capability_filter import (
        ActionCapabilityFilter,
        get_action_capability_filter,
        ActionFilterResult,
    )
    CAPABILITY_FILTER_AVAILABLE = True
except ImportError:
    CAPABILITY_FILTER_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _get_executor_for_request(db_session):
    """
    Create a new UnifiedToolExecutor per request using the shared registry.
    Avoids mutating a global executor's db_session (race conditions).
    Registry remains singleton; executors are per-request.
    """
    registry = registry_get_tool_registry()
    return UnifiedToolExecutor(db_session, registry=registry)


def _new_trace_id() -> str:
    return uuid4().hex[:12]


def _summarize_args(args: Any) -> str:
    if isinstance(args, dict):
        keys = list(args.keys())
        return f"dict keys={keys[:12]}{'...' if len(keys) > 12 else ''}"
    if isinstance(args, list):
        return f"list len={len(args)}"
    return f"{type(args).__name__}"


def _is_fatal_dependency_error(error: Optional[str]) -> bool:
    if not error:
        return False
    lowered = error.lower()
    return "composio openai sdk not available" in lowered or "composio-openai" in lowered


def _resolve_relative_date_window_utc(intent: str) -> Optional[tuple[datetime.date, datetime.date]]:
    """
    Resolve common relative date phrases to an (after_date, before_date) window in UTC.

    NOTE: before_date is exclusive (like Gmail's `before:`).
    """
    t = (intent or "").lower()
    if not t:
        return None

    today = datetime.now(timezone.utc).date()
    start_of_week = today - timedelta(days=today.weekday())  # Monday start (UTC)
    tomorrow = today + timedelta(days=1)

    # Order matters: match more specific phrases first.
    if "this week" in t:
        return (start_of_week, tomorrow)
    if "last week" in t or "previous week" in t:
        return (start_of_week - timedelta(days=7), start_of_week)
    if "yesterday" in t:
        return (today - timedelta(days=1), today)
    if "today" in t:
        return (today, tomorrow)
    if "past 7 days" in t or "last 7 days" in t:
        return (today - timedelta(days=7), tomorrow)

    return None


_AFTER_BEFORE_DATE_RE = re.compile(r"\b(after|before):\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", re.IGNORECASE)


def _rewrite_query_after_before(query: str, after_date: datetime.date, before_date: datetime.date) -> str:
    """
    Replace any existing after:/before: date filters in a query string and append the resolved window.
    Preserves other query terms.
    """
    q = (query or "").strip()
    q = _AFTER_BEFORE_DATE_RE.sub("", q)
    q = re.sub(r"\s+", " ", q).strip()
    window = f"after:{after_date:%Y/%m/%d} before:{before_date:%Y/%m/%d}"
    return f"{q} {window}".strip() if q else window


@contextmanager
def _session_scope():
    """Lightweight context manager to ensure sessions are always closed."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# =============================================================================
# PUBLIC: Get tools for an agent
# =============================================================================


def get_tools_for_agent(
    agent_id: Optional[int] = None,
    db_session=None,
    workspace_id: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Get tools from modules.tools.ToolRegistry in OpenAI function format.
    SINGLE SOURCE OF TRUTH — no duplicate definitions.

    Renamed from ``get_chatbot_tools`` — zero chatbot-specific logic.
    """
    session_used = db_session or SessionLocal()
    trace_id = _new_trace_id()
    start_time = time.time()
    try:
        registry = registry_get_tool_registry(db_session=session_used)
        all_candidates = registry.get_all_tools(active_only=True)
        filtered_tools = all_candidates
        denied: List[Dict[str, Any]] = []

        # Resolve workspace_id from agent if not provided (needed for Composio gating)
        if agent_id is not None and workspace_id is None:
            try:
                from core.models import Agent as AgentModel
                agent_row = session_used.query(AgentModel).filter(AgentModel.id == agent_id).first()
                if agent_row and getattr(agent_row, "workspace_id", None):
                    workspace_id = agent_row.workspace_id
                    logger.info(f"[tool-trace {trace_id}] Resolved workspace_id from agent {agent_id} for tool list gating")
            except Exception as exc:
                logger.warning(f"[tool-trace {trace_id}] Failed to resolve workspace_id for agent {agent_id}: {exc}")

        if agent_id is not None:
            filtered_tools = []
            for tool in all_candidates:
                allowed, reason = registry.validate_tool_access(
                    agent_id=agent_id,
                    tool_name=tool.name,
                    db=session_used,
                    workspace_id=workspace_id
                )
                if allowed:
                    filtered_tools.append(tool)
                else:
                    denied.append({"tool": tool.name, "reason": reason})

        # Convert to OpenAI function format
        openai_tools = []
        for tool in filtered_tools:
            schema = tool.to_openai_format()

            # Enrich composio_execute with available actions from cache
            if tool.name == "composio_execute" and agent_id is not None:
                try:
                    from core.models.composio_cache import AgentAppAssignment, ComposioActionCache

                    # Get assigned apps for this agent
                    assignments = (
                        session_used.query(AgentAppAssignment)
                        .filter(
                            AgentAppAssignment.agent_id == agent_id,
                            AgentAppAssignment.is_active == True,
                            AgentAppAssignment.app_type == "EXTERNAL"
                        )
                        .all()
                    )

                    if assignments:
                        app_names = [a.app_name for a in assignments if a.app_name]

                        # Get top actions for these apps from cache
                        actions = (
                            session_used.query(ComposioActionCache)
                            .filter(
                                ComposioActionCache.app_name.in_(app_names),
                            )
                            .order_by(ComposioActionCache.app_name, ComposioActionCache.display_name)
                            .limit(50)  # Top 50 actions across all assigned apps
                            .all()
                        )

                        if actions:
                            # Group by app
                            actions_by_app = {}
                            for action in actions:
                                app = action.app_name
                                if app not in actions_by_app:
                                    actions_by_app[app] = []

                                # Use display_name for readability, fallback to action_name
                                desc = action.description or ""
                                actions_by_app[app].append(f"  - {action.action_name}: {desc[:80]}")

                            # Build enriched description
                            action_list = []
                            for app in sorted(actions_by_app.keys()):
                                action_list.append(f"\n{app} actions:")
                                action_list.extend(actions_by_app[app][:10])  # Max 10 per app

                            enriched_desc = (
                                schema["description"] +
                                "\n\nAvailable actions (use exact action_name):" +
                                "\n".join(action_list)
                            )

                            schema["description"] = enriched_desc
                            logger.info(f"[tool-trace {trace_id}] Enriched composio_execute with {len(actions)} actions from {len(app_names)} apps")

                except Exception as e:
                    logger.warning(f"[tool-trace {trace_id}] Failed to enrich composio_execute: {e}")

            openai_tools.append({
                "type": "function",
                "function": schema
            })

        # PRD-64: Append platform action tools from ActionRegistry
        try:
            from modules.tools.discovery import get_action_registry
            action_registry = get_action_registry()
            platform_tools = action_registry.to_openai_tools()
            openai_tools.extend(platform_tools)
            if platform_tools:
                logger.info(f"[tool-trace {trace_id}] Added {len(platform_tools)} platform action tools")
        except Exception as e:
            logger.debug(f"[tool-trace {trace_id}] Platform actions unavailable: {e}")

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[tool-trace {trace_id}] Loaded {len(openai_tools)} tools "
            f"(agent_id={agent_id}, denied={len(denied)}, "
            f"candidates={len(all_candidates)}, {elapsed_ms}ms)"
        )
        if denied:
            sample = denied[:10]
            logger.info(f"[tool-trace {trace_id}] Denied sample: {sample}")
        return openai_tools
    except Exception as e:
        logger.error(f"[tool-trace {trace_id}] Error loading tools from registry: {e}")
        return []
    finally:
        if db_session is None:
            session_used.close()


def get_agent_tools(
    agent_id: Optional[int] = None,
    workspace_id: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper: get tools for an agent with automatic session management.

    Renamed from ``get_chat_tools`` — zero chatbot-specific logic.
    """
    with _session_scope() as session:
        return get_tools_for_agent(agent_id=agent_id, db_session=session, workspace_id=workspace_id)


# =============================================================================
# PUBLIC: Execute a tool
# =============================================================================


async def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    agent_id: int = 1,
    workspace_id: Optional[UUID] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a tool via modules.tools.UnifiedToolExecutor.
    SINGLE ENTRY POINT for all tool execution.

    Uses cached executor to avoid re-initializing tool registry on every call.
    """
    from core.database.database import SessionLocal

    trace = trace_id or _new_trace_id()
    db_session = SessionLocal()
    try:
        if workspace_id is None and agent_id:
            try:
                from core.models import Agent as AgentModel
                agent_row = db_session.query(AgentModel).filter(AgentModel.id == agent_id).first()
                if agent_row and agent_row.workspace_id:
                    workspace_id = agent_row.workspace_id
                    logger.info(f"[tool-trace {trace}] Resolved workspace_id from agent {agent_id}")
                else:
                    logger.warning(f"[tool-trace {trace}] Agent workspace_id missing for agent={agent_id}")
            except Exception as exc:
                logger.warning(f"[tool-trace {trace}] Failed to resolve workspace_id: {exc}")

        executor = _get_executor_for_request(db_session)

        logger.info(
            f"[tool-trace {trace}] execute_tool start tool={tool_name} "
            f"agent={agent_id} workspace={workspace_id} args={_summarize_args(tool_args)}"
        )
        result = await executor.execute_tool(
            tool_name,
            tool_args,
            agent_id,
            workspace_id=workspace_id,
            trace_id=trace
        )
        db_session.commit()
        logger.info(
            f"[tool-trace {trace}] execute_tool done tool={tool_name} "
            f"success={bool(result.get('success'))}"
        )
        return result
    except Exception as e:
        db_session.rollback()
        logger.error(f"[tool-trace {trace}] execute_tool error tool={tool_name}: {e}")
        raise
    finally:
        db_session.close()


# =============================================================================
# PUBLIC: ToolRouter class — execute + format results
# =============================================================================


class ToolRouter:
    """
    Routes tool execution and formats results.
    Generic — no chatbot-specific logic.
    """

    def __init__(self):
        self.formatter = ToolResultFormatter

    async def execute_and_format(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        agent_id: int = 1,
        workspace_id: Optional[UUID] = None,
        original_intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool and return formatted results.

        Returns:
            {
                'success': bool,
                'frontend_data': dict,  # For artifact viewer
                'llm_context': str,     # For LLM injection
                'raw_result': dict      # Original result
            }
        """
        trace_id = _new_trace_id()
        logger.info(
            f"[tool-trace {trace_id}] ToolRouter execute_and_format "
            f"tool={tool_name} agent={agent_id} workspace={workspace_id} args={_summarize_args(tool_args)}"
        )

        try:
            # Deterministic guard: if the user asked for a relative date range,
            # and the tool call includes after:/before: date filters, rewrite them
            # based on current UTC dates to prevent stale/hallucinated years.
            if original_intent and tool_name == "composio_execute" and isinstance(tool_args, dict):
                try:
                    window = _resolve_relative_date_window_utc(original_intent)
                    if window:
                        after_d, before_d = window
                        params = tool_args.get("params") or tool_args.get("parameters")
                        if isinstance(params, dict):
                            q = params.get("query")
                            if isinstance(q, str):
                                params["query"] = _rewrite_query_after_before(q, after_d, before_d)
                except Exception as exc:
                    logger.debug(f"[tool-trace {trace_id}] Relative date rewrite skipped: {exc}")

            # Use validation wrapper for Composio tools to prevent calling unassigned actions
            is_composio = tool_name.startswith("composio_") or tool_name == "composio_execute"
            if is_composio and original_intent:
                result = await execute_tool_with_validation(
                    tool_name=tool_name,
                    tool_args=tool_args if isinstance(tool_args, dict) else {},
                    original_intent=original_intent,
                    agent_id=agent_id,
                    workspace_id=workspace_id,
                    allow_destructive=False,
                )
            else:
                result = await execute_tool(
                    tool_name,
                    tool_args if isinstance(tool_args, dict) else {},
                    agent_id,
                    workspace_id=workspace_id,
                    trace_id=trace_id,
                )

            # Check success (support multiple executor result shapes)
            success = (
                bool(result.get("success"))
                or result.get("status") == "success"
                or bool(result.get("results"))
            )

            if success:
                frontend_data = self.formatter.format_for_frontend(result, tool_name)
                llm_context = self.formatter.format_for_llm(result, tool_name)

                logger.info(f"[tool-trace {trace_id}] {tool_name} succeeded")
                logger.info(f"[tool-trace {trace_id}] frontend_data keys: {list(frontend_data.keys()) if frontend_data else 'EMPTY'}")
                return {
                    "success": True,
                    "frontend_data": frontend_data,
                    "llm_context": llm_context,
                    "raw_result": result,
                    "fatal_error": False,
                    "error_type": None,
                }

            error = result.get("error", "Unknown error")
            error_type = result.get("error_type")
            fatal_error = bool(result.get("fatal")) or _is_fatal_dependency_error(error)
            llm_error = (
                "Tool execution failed due to a server configuration issue. Please restart the backend and try again."
                if fatal_error
                else error
            )
            logger.warning(f"[tool-trace {trace_id}] {tool_name} failed: {error}")
            return {
                "success": False,
                "frontend_data": {},
                "llm_context": f"Tool {tool_name} failed: {llm_error}",
                "raw_result": result,
                "fatal_error": fatal_error,
                "error_type": error_type,
            }

        except Exception as e:
            error_msg = str(e)
            fatal_error = _is_fatal_dependency_error(error_msg)
            llm_error = (
                "Tool execution failed due to a server configuration issue. Please restart the backend and try again."
                if fatal_error
                else error_msg
            )
            logger.error(f"[tool-trace {trace_id}] {tool_name} exception: {error_msg}")
            return {
                "success": False,
                "frontend_data": {},
                "llm_context": f"Tool {tool_name} error: {llm_error}",
                "raw_result": {"success": False, "error": error_msg},
                "fatal_error": fatal_error,
                "error_type": "dependency_missing" if fatal_error else None,
            }

    def truncate_for_llm(
        self,
        result: Dict[str, Any],
        tool_name: str = "unknown",
        max_chars: int = 3000,
    ) -> str:
        """Truncate tool results for LLM context."""
        return self.formatter.format_for_llm(result, tool_name, max_chars)


# Module-level singleton
_tool_router: Optional[ToolRouter] = None


def get_tool_router() -> ToolRouter:
    """Get or create the global ToolRouter instance."""
    global _tool_router
    if _tool_router is None:
        _tool_router = ToolRouter()
    return _tool_router


# =============================================================================
# CAPABILITY-BASED FILTERING (PRD-37)
# =============================================================================


async def get_filtered_composio_actions(
    intent: str,
    enabled_apps: List[str],
    db_session=None,
    include_destructive: bool = False,
    max_actions: int = 15,
) -> Dict[str, Any]:
    """
    Get Composio actions filtered by capability for a given intent.

    This is the core of the capability-based filtering system:
    1. Extracts capabilities from intent text
    2. Filters actions from local metadata (no API calls)
    3. Excludes destructive actions unless explicitly allowed
    4. Returns ranked, relevant actions

    Args:
        intent: User's intent text (e.g., "send a message to #general")
        enabled_apps: List of Composio app IDs user has enabled
        db_session: Optional database session
        include_destructive: Whether to include destructive actions
        max_actions: Maximum number of actions to return

    Returns:
        Dict with filtered actions and metadata
    """
    if not CAPABILITY_FILTER_AVAILABLE:
        logger.warning("Capability filter not available, returning empty result")
        return {
            "success": False,
            "error": "Capability filter not available",
            "actions": [],
            "capabilities": []
        }

    trace_id = _new_trace_id()
    session_used = db_session or SessionLocal()

    try:
        filter_service = get_action_capability_filter(session_used)

        result = await filter_service.get_actions_for_intent(
            intent=intent,
            enabled_apps=enabled_apps,
            include_destructive=include_destructive,
            max_actions=max_actions
        )

        logger.info(
            f"[tool-trace {trace_id}] Capability filter: "
            f"intent='{intent[:50]}...' caps={result.extracted_capabilities} "
            f"matched={result.filtered_count}/{result.total_available} "
            f"fallback={result.fallback_used}"
        )

        # Convert to dict for API response
        actions_data = [
            {
                "action_id": a.action_id,
                "app_id": a.app_id,
                "capabilities": a.capabilities,
                "relevance_score": a.relevance_score,
                "description": a.description,
                "destructive": a.destructive,
                "requires_confirmation": a.requires_confirmation
            }
            for a in result.actions
        ]

        return {
            "success": True,
            "intent": result.intent,
            "extracted_capabilities": result.extracted_capabilities,
            "total_available": result.total_available,
            "filtered_count": result.filtered_count,
            "fallback_used": result.fallback_used,
            "actions": actions_data
        }

    except Exception as e:
        logger.error(f"[tool-trace {trace_id}] Capability filter error: {e}")
        return {
            "success": False,
            "error": str(e),
            "actions": [],
            "capabilities": []
        }
    finally:
        if db_session is None:
            session_used.close()


def validate_action_for_intent(
    action_id: str,
    intent: str,
    db_session=None,
    allow_destructive: bool = False,
) -> Tuple[bool, str]:
    """
    Validate that an action is eligible for execution given an intent.

    This is the EXECUTION-TIME validation gate (per GPT-5.2 recommendation).
    Called before executing any Composio action to ensure the action
    matches the original intent's capabilities.

    Args:
        action_id: The Composio action ID to validate
        intent: The original user intent
        db_session: Optional database session
        allow_destructive: Whether destructive actions are allowed

    Returns:
        (eligible, reason) tuple
    """
    if not CAPABILITY_FILTER_AVAILABLE:
        # Fail open if capability filter not available
        logger.warning("Capability filter not available, allowing action")
        return True, "Capability filter not available (fail open)"

    session_used = db_session or SessionLocal()

    try:
        filter_service = get_action_capability_filter(session_used)

        eligible, reason = filter_service.check_action_eligibility(
            action_id=action_id,
            intent=intent,
            allow_destructive=allow_destructive
        )

        if not eligible:
            logger.warning(
                f"Action validation failed: action={action_id} "
                f"intent='{intent[:30]}...' reason={reason}"
            )

        return eligible, reason

    except Exception as e:
        # Log at debug level if it's a missing table error (expected during development)
        error_str = str(e)
        if "does not exist" in error_str or "UndefinedTable" in error_str:
            logger.debug(f"Action validation skipped (table not created): {e}")
        else:
            logger.warning(f"Action validation error (fail open): {e}")
        # Fail open on errors to avoid blocking legitimate actions
        return True, "Validation skipped (fail open)"
    finally:
        if db_session is None:
            session_used.close()


async def execute_tool_with_validation(
    tool_name: str,
    tool_args: Dict[str, Any],
    original_intent: str,
    agent_id: int = 1,
    workspace_id: Optional[UUID] = None,
    allow_destructive: bool = False,
) -> Dict[str, Any]:
    """
    Execute a tool with intent-based validation.

    This wraps execute_tool() with an additional validation layer
    that checks if the action matches the original intent's capabilities.

    Args:
        tool_name: Tool to execute
        tool_args: Tool arguments
        original_intent: The original user intent (for validation)
        agent_id: Agent ID
        workspace_id: Workspace ID
        allow_destructive: Whether to allow destructive actions

    Returns:
        Tool execution result (with validation info)
    """
    trace_id = _new_trace_id()

    # Only validate Composio actions
    is_composio = tool_name.startswith("composio_") or tool_name == "composio_execute"

    if is_composio and original_intent:
        # Extract action ID from tool name or args
        action_id = None
        if tool_name.startswith("composio_"):
            action_id = tool_name.replace("composio_", "")
        elif isinstance(tool_args, dict):
            action_id = tool_args.get("action") or tool_args.get("action_name")

        if action_id:
            eligible, reason = validate_action_for_intent(
                action_id=action_id,
                intent=original_intent,
                allow_destructive=allow_destructive
            )

            if not eligible:
                logger.warning(
                    f"[tool-trace {trace_id}] Execution blocked by validation: "
                    f"tool={tool_name} action={action_id} reason={reason}"
                )
                return {
                    "success": False,
                    "error": f"Action not allowed for this intent: {reason}",
                    "error_type": "validation_blocked",
                    "blocked_by": "capability_validation",
                    "action_id": action_id,
                    "intent": original_intent[:100]
                }

    # Proceed with normal execution
    return await execute_tool(
        tool_name=tool_name,
        tool_args=tool_args,
        agent_id=agent_id,
        workspace_id=workspace_id,
        trace_id=trace_id
    )


def get_capability_filter_stats(db_session=None) -> Dict[str, Any]:
    """
    Get statistics about the capability filter system.

    Returns info about classified actions, coverage, etc.
    """
    if not CAPABILITY_FILTER_AVAILABLE:
        return {"available": False, "error": "Capability filter not available"}

    session_used = db_session or SessionLocal()

    try:
        filter_service = get_action_capability_filter(session_used)
        stats = filter_service.get_statistics()
        stats["available"] = True
        return stats
    except Exception as e:
        return {"available": False, "error": str(e)}
    finally:
        if db_session is None:
            session_used.close()
