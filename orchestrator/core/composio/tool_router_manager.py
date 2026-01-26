"""
Composio Tool Router Manager
=============================

Manages Tool Router sessions for workspaces.

Tool Router provides meta-tools (SEARCH, EXECUTE, etc.) instead of exposing
all 656 individual Composio actions, dramatically reducing LLM token usage
and improving decision-making.

Meta-tools:
- COMPOSIO_SEARCH_TOOLS: Search for the right tool by intent
- COMPOSIO_MULTI_EXECUTE_TOOL: Execute one or more tools
- COMPOSIO_GET_TOOL_SCHEMAS: Get parameter schemas
- COMPOSIO_MANAGE_CONNECTIONS: Handle OAuth authentication
- COMPOSIO_REMOTE_BASH_TOOL: Execute bash commands
- COMPOSIO_REMOTE_WORKBENCH: Process files/bulk operations

Usage:
    manager = ToolRouterManager(composio_client)
    
    # Get meta-tools for a workspace
    tools = manager.get_tools_for_workspace(
        workspace_id=uuid,
        enabled_apps=["slack", "gmail"]
    )
    
    # LLM sees only 6 meta-tools + core Automatos tools (~25 total)
"""

import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

logger = logging.getLogger(__name__)


class ToolRouterManager:
    """
    Manages Composio Tool Router sessions per workspace.
    
    Sessions are cached per workspace to avoid recreating them on every request.
    """
    
    def __init__(self, composio_client):
        """
        Initialize Tool Router manager.
        
        Args:
            composio_client: ComposioClient instance
        """
        self.composio = composio_client.composio  # Get the actual Composio SDK client
        self._sessions: Dict[str, Any] = {}  # workspace_id -> ToolRouterSession
    
    def get_or_create_session(
        self,
        workspace_id: UUID,
        enabled_apps: Optional[List[str]] = None
    ) -> Any:
        """
        Get or create a Tool Router session for a workspace.
        
        Args:
            workspace_id: Workspace UUID
            enabled_apps: List of apps to enable (e.g., ["slack", "gmail"])
                         If None, all apps are available
            
        Returns:
            ToolRouterSession instance
        """
        # Use workspace_id as the user_id in Composio
        user_id = str(workspace_id)
        
        # Create cache key from workspace + enabled apps
        apps_key = "-".join(sorted(enabled_apps)) if enabled_apps else "all"
        cache_key = f"{workspace_id}_{apps_key}"
        
        # Return cached session if exists
        if cache_key in self._sessions:
            logger.debug(f"Using cached Tool Router session for workspace {workspace_id}")
            return self._sessions[cache_key]
        
        # Create new session
        try:
            logger.info(f"Creating Tool Router session for workspace {workspace_id} with apps: {enabled_apps}")
            
            session_params = {"user_id": user_id}
            if enabled_apps:
                session_params["toolkits"] = enabled_apps
            
            session = self.composio.create(**session_params)
            
            self._sessions[cache_key] = session
            logger.info(f"✅ Tool Router session created (session_id: {getattr(session, 'session_id', 'unknown')})")
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to create Tool Router session: {e}")
            raise
    
    def get_tools_for_workspace(
        self,
        workspace_id: UUID,
        enabled_apps: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get meta-tools for a workspace in OpenAI function format.
        
        Args:
            workspace_id: Workspace UUID
            enabled_apps: List of enabled apps (e.g., ["slack", "gmail"])
            
        Returns:
            List of meta-tools in OpenAI format (typically 6 tools)
        """
        try:
            session = self.get_or_create_session(workspace_id, enabled_apps)
            tools = session.tools()
            
            logger.info(
                f"Retrieved {len(tools)} meta-tools from Tool Router "
                f"for workspace {workspace_id} (apps: {enabled_apps or 'all'})"
            )
            
            return tools
            
        except Exception as e:
            logger.error(f"Failed to get Tool Router tools: {e}")
            return []
    
    def clear_session(self, workspace_id: UUID):
        """
        Clear cached session for a workspace.
        
        Use this when workspace settings change (e.g., apps enabled/disabled).
        
        Args:
            workspace_id: Workspace UUID
        """
        # Remove all sessions for this workspace
        keys_to_remove = [k for k in self._sessions.keys() if k.startswith(str(workspace_id))]
        for key in keys_to_remove:
            del self._sessions[key]
            logger.info(f"Cleared Tool Router session: {key}")
    
    def get_session_info(self, workspace_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get information about the current session for a workspace.
        
        Args:
            workspace_id: Workspace UUID
            
        Returns:
            Session info or None if no session exists
        """
        # Find any session for this workspace
        for key, session in self._sessions.items():
            if key.startswith(str(workspace_id)):
                return {
                    "session_id": getattr(session, 'session_id', None),
                    "workspace_id": workspace_id,
                    "cache_key": key,
                    "has_tools": hasattr(session, 'tools'),
                }
        return None


# Singleton instance
_tool_router_manager: Optional[ToolRouterManager] = None


def get_tool_router_manager(composio_client=None) -> ToolRouterManager:
    """Get or create the Tool Router manager singleton."""
    global _tool_router_manager
    
    if _tool_router_manager is None:
        if composio_client is None:
            from core.composio.client import get_composio_client
            composio_client = get_composio_client()
        
        _tool_router_manager = ToolRouterManager(composio_client)
    
    return _tool_router_manager
