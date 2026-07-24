"""
Platform tool executors -- research tools (RAG, CodeGraph) and platform actions.
Extracted from unified_executor.py.
"""

import logging
from typing import Any, Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


async def execute_platform_tool(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
) -> Dict[str, Any]:
    """Execute research tools via AgentPlatformTools."""
    name_map = {'search_documents': 'search_knowledge', 'search_code': 'search_codebase'}
    canonical_name = name_map.get(tool_name, tool_name)
    return await executor.platform_tools.execute_tool(
        tool_name=canonical_name,
        parameters=parameters,
        agent_id=agent_id,
    )


async def execute_platform_action(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    workspace_id: Optional[UUID] = None,
    trace_id: Optional[str] = None,
    caller_context: Optional[Dict[str, Any]] = None,
    agent_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a platform action via PlatformActionExecutor.

    Args:
        caller_context: Optional dict with keys user_id, system_role, workspace_role.
            Used for admin_only permission gating (US-002/US-003).
            If None, falls back to workspace-scoped check via
            _workspace_has_admin_owner() — admin workspace grants access,
            non-admin workspace denies.
        agent_id: ID of the calling agent. Injected as ``_agent_id`` (and
            ``_agent_name`` resolved from DB) into params so handlers like
            ``platform_submit_report`` can attribute the call. Without this,
            recipe-step calls failed with "Could not determine calling agent".
    """
    if not workspace_id:
        return {
            "success": False,
            "error": "workspace_id required for platform actions",
            "tool": tool_name,
        }

    # Actor identity is server-minted from the trusted runtime ``agent_id``,
    # NEVER from caller/LLM-supplied params. Strip any _agent_id/_agent_name a
    # tool call tried to smuggle in — otherwise an agent could set
    # _agent_id=<a system agent's id> and impersonate it, bypassing the
    # hierarchy permission check (core.security.hierarchy_permissions) entirely.
    # When agent_id is unknown the keys stay absent → the permission check sees
    # no actor and fails closed (anonymous_actor → deny).
    if isinstance(parameters, dict):
        parameters = {
            k: v for k, v in parameters.items()
            if k not in ("_agent_id", "_agent_name")
        }
        if agent_id:
            parameters["_agent_id"] = agent_id
            try:
                from core.models import Agent
                agent = executor.db.query(Agent).filter(
                    Agent.id == agent_id,
                    Agent.workspace_id == workspace_id,
                ).first()
                if agent:
                    parameters["_agent_name"] = agent.name
            except Exception as e:
                logger.debug("[exec_platform] _agent_name lookup failed: %s", e)

    try:
        from modules.tools.discovery.platform_executor import PlatformActionExecutor
        executor_inst = PlatformActionExecutor(db=executor.db, workspace_id=workspace_id)
        result = await executor_inst.execute(tool_name, parameters, caller_context=caller_context)
        logger.info(
            f"[tool-trace {trace_id or 'no-trace'}] Platform action {tool_name} "
            f"success={result.get('success')}"
        )
        return result
    except Exception as e:
        logger.error(f"[tool-trace {trace_id or 'no-trace'}] Platform action error: {e}", exc_info=True)
        return {"success": False, "error": str(e), "tool": tool_name}
