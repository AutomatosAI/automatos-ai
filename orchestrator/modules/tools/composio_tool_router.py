"""
Composio Tool Router Integration
=================================

Replaces generic composio_execute with Composio's intelligent Tool Router.

Tool Router provides 2 meta-tools:
1. COMPOSIO_SEARCH_TOOLS - Find the right action by intent
2. COMPOSIO_EXECUTE_TOOL - Execute the found action

Benefits:
- Better action selection (Composio's AI picks correctly)
- Scoped to agent's assigned apps only
- Works dynamically with all 850+ tools
- No cache/sync/auto-mapper needed for execution
"""

import logging
from typing import Dict, Any, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class ComposioToolRouter:
    """
    Manages Composio Tool Router for intelligent action selection and execution.

    Scoped to agent's assigned apps for faster, more accurate results.
    """

    def __init__(self, db_session, workspace_id: UUID, assigned_apps: List[str]):
        """
        Initialize Tool Router scoped to agent's assigned apps.

        Args:
            db_session: Database session
            workspace_id: Workspace UUID
            assigned_apps: List of app names assigned to agent (e.g., ["GMAIL", "SLACK"])
        """
        self.db_session = db_session
        self.workspace_id = workspace_id
        self.assigned_apps = [app.lower() for app in assigned_apps]  # Composio uses lowercase
        self.session = None

        logger.info(f"[ToolRouter] Initializing for workspace {workspace_id} with apps: {self.assigned_apps}")

    def _get_or_create_session(self):
        """Get or create Tool Router session scoped to assigned apps."""
        if self.session:
            return self.session

        try:
            from core.composio.tool_router_manager import get_tool_router_manager
            from core.composio.client import get_composio_client

            composio_client = get_composio_client()
            tool_router_manager = get_tool_router_manager(composio_client)

            # Create session scoped to ONLY assigned apps
            self.session = tool_router_manager.get_or_create_session(
                workspace_id=self.workspace_id,
                enabled_apps=self.assigned_apps if self.assigned_apps else None
            )

            logger.info(f"✅ [ToolRouter] Session created (scoped to {len(self.assigned_apps)} apps)")
            return self.session

        except Exception as e:
            logger.error(f"[ToolRouter] Failed to create session: {e}")
            raise

    def search_tools(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search for the right tool/action by natural language intent.

        Args:
            query: Natural language description (e.g., "send slack message")
            max_results: Maximum number of results to return

        Returns:
            Dict with success status and list of matching actions
        """
        try:
            session = self._get_or_create_session()

            logger.info(f"[ToolRouter] Searching tools: '{query}' (scoped to {self.assigned_apps})")

            # Call Composio's search API
            # Note: Actual implementation depends on Composio SDK version
            # This is a placeholder - adjust based on actual API
            results = session.search_tools(
                query=query,
                max_results=max_results
            )

            logger.info(f"[ToolRouter] Found {len(results)} matching actions")

            return {
                "success": True,
                "results": results,
                "query": query
            }

        except Exception as e:
            logger.error(f"[ToolRouter] Search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }

    def execute_tool(
        self,
        action: str,
        params: Dict[str, Any],
        entity_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a Composio action.

        Args:
            action: Action name (e.g., "SLACK_SEND_MESSAGE")
            params: Action parameters
            entity_id: Optional entity ID (defaults to workspace entity)

        Returns:
            Dict with execution result
        """
        try:
            from core.composio.client import get_composio_client

            client = get_composio_client()

            # Get entity if not provided
            if not entity_id:
                from core.composio.entity_manager import EntityManager
                entity_manager = EntityManager(self.db_session)
                entity = entity_manager.get_entity_for_workspace(self.workspace_id)
                entity_id = entity.get("composio_entity_id")

                if not entity_id:
                    raise ValueError("No Composio entity found for workspace")

            logger.info(f"[ToolRouter] Executing: {action} with entity {entity_id}")

            # Execute via Composio
            result = client.execute_action(
                action=action,
                params=params,
                entity_id=entity_id
            )

            logger.info(f"[ToolRouter] Execution {'succeeded' if result.get('success') else 'failed'}")

            return result

        except Exception as e:
            logger.error(f"[ToolRouter] Execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }


def get_tool_router_for_agent(
    db_session,
    workspace_id: UUID,
    agent_id: int
) -> ComposioToolRouter:
    """
    Get Tool Router instance scoped to agent's assigned apps.

    Args:
        db_session: Database session
        workspace_id: Workspace UUID
        agent_id: Agent ID

    Returns:
        ComposioToolRouter instance
    """
    from core.models.composio_cache import AgentAppAssignment

    # Get agent's assigned apps
    assignments = (
        db_session.query(AgentAppAssignment)
        .filter(
            AgentAppAssignment.agent_id == agent_id,
            AgentAppAssignment.is_active == True,
            AgentAppAssignment.app_type == "EXTERNAL",
        )
        .all()
    )

    assigned_apps = [a.app_name for a in assignments if a.app_name]

    logger.info(f"[ToolRouter] Agent {agent_id} has {len(assigned_apps)} assigned apps: {assigned_apps}")

    return ComposioToolRouter(
        db_session=db_session,
        workspace_id=workspace_id,
        assigned_apps=assigned_apps
    )
