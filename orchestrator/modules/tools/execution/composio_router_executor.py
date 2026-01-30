"""
Composio Tool Router Executor
==============================

Handles execution of composio_search_tools and composio_execute_tool.
"""

import logging
from typing import Dict, Any
from uuid import UUID

logger = logging.getLogger(__name__)


class ComposioToolRouterExecutor:
    """Executor for Composio Tool Router meta-tools."""

    def __init__(self, db_session, workspace_id: UUID, agent_id: int):
        self.db_session = db_session
        self.workspace_id = workspace_id
        self.agent_id = agent_id
        self._tool_router = None

    def _get_tool_router(self):
        """Get or create Tool Router instance for this agent."""
        if self._tool_router:
            return self._tool_router

        from modules.tools.composio_tool_router import get_tool_router_for_agent

        self._tool_router = get_tool_router_for_agent(
            db_session=self.db_session,
            workspace_id=self.workspace_id,
            agent_id=self.agent_id
        )

        return self._tool_router

    def search_tools(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search for Composio actions by natural language intent.

        Args:
            query: Natural language description
            max_results: Max results to return

        Returns:
            Dict with search results
        """
        try:
            tool_router = self._get_tool_router()
            result = tool_router.search_tools(query, max_results)

            if result.get("success"):
                logger.info(f"[ToolRouterExecutor] Search succeeded: {len(result.get('results', []))} results")
            else:
                logger.warning(f"[ToolRouterExecutor] Search failed: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"[ToolRouterExecutor] Search exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }

    def execute_tool(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a Composio action.

        Args:
            action: Action name (e.g., "SLACK_SEND_MESSAGE")
            params: Action parameters

        Returns:
            Dict with execution result
        """
        try:
            tool_router = self._get_tool_router()
            result = tool_router.execute_tool(action, params)

            if result.get("success"):
                logger.info(f"[ToolRouterExecutor] Execution succeeded: {action}")
            else:
                logger.warning(f"[ToolRouterExecutor] Execution failed: {action} - {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"[ToolRouterExecutor] Execution exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
