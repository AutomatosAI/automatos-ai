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
) -> Dict[str, Any]:
    """Execute a platform action via PlatformActionExecutor."""
    if not workspace_id:
        return {
            "success": False,
            "error": "workspace_id required for platform actions",
            "tool": tool_name,
        }

    try:
        from modules.tools.discovery.platform_executor import PlatformActionExecutor
        executor_inst = PlatformActionExecutor(db=executor.db, workspace_id=workspace_id)
        result = await executor_inst.execute(tool_name, parameters)
        logger.info(
            f"[tool-trace {trace_id or 'no-trace'}] Platform action {tool_name} "
            f"success={result.get('success')}"
        )
        return result
    except Exception as e:
        logger.error(f"[tool-trace {trace_id or 'no-trace'}] Platform action error: {e}", exc_info=True)
        return {"success": False, "error": str(e), "tool": tool_name}
