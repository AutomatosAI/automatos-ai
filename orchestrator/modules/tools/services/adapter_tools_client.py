"""
Adapter Tools Client (PRD-35)
=============================

Client for fetching tool definitions from the Unified Adapter.
Adapter owns tool definitions; this client provides access to them.
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

from config import Config

logger = logging.getLogger(__name__)


class AdapterToolsClient:
    """
    Client for interacting with the Unified Adapter's tool catalog.
    
    The Adapter is the source of truth for tool definitions.
    This client fetches tool metadata for:
    - Displaying available tools in the UI
    - Building tool schemas for agent execution
    - Validating tool references
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout_seconds: float = 20,
    ) -> None:
        config = Config()
        self.base_url = (base_url or getattr(config, 'ADAPTER_ADMIN_BASE_URL', None) or "").rstrip("/")
        self.token = token or getattr(config, 'ADAPTER_ADMIN_TOKEN', None)
        self.timeout_seconds = timeout_seconds
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamp = 0

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def is_configured(self) -> bool:
        """Check if the adapter client is configured."""
        return bool(self.base_url)

    async def list_tools(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """
        List all tools from the Adapter.
        
        Args:
            enabled_only: If True, only return enabled tools
            
        Returns:
            List of tool definitions
        """
        if not self.base_url:
            logger.warning("Adapter base URL not configured, returning empty tool list")
            return []
        
        url = f"{self.base_url}/admin/tools"
        params = {}
        if enabled_only:
            params["enabled_only"] = "true"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url, headers=self._headers(), params=params)
                response.raise_for_status()
                tools = response.json()
                
                # Handle both list and paginated response formats
                if isinstance(tools, dict) and "data" in tools:
                    return tools["data"]
                elif isinstance(tools, list):
                    return tools
                else:
                    logger.warning(f"Unexpected response format from adapter: {type(tools)}")
                    return []
                    
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch tools from adapter: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching tools: {e}")
            return []

    async def get_tool(self, adapter_tool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific tool by ID from the Adapter.
        
        Args:
            adapter_tool_id: The tool's identifier in the Adapter
            
        Returns:
            Tool definition or None if not found
        """
        if not self.base_url:
            logger.warning("Adapter base URL not configured")
            return None
        
        # Try to find in full list (more efficient if we need multiple tools)
        tools = await self.list_tools(enabled_only=False)
        for tool in tools:
            if tool.get("name") == adapter_tool_id or tool.get("adapter_tool_id") == adapter_tool_id:
                return tool
        
        # If not found in list, try direct lookup
        url = f"{self.base_url}/admin/tools/{adapter_tool_id}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url, headers=self._headers())
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch tool {adapter_tool_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching tool: {e}")
            return None

    async def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all tools in a specific category."""
        tools = await self.list_tools()
        return [t for t in tools if t.get("category") == category]

    async def get_tools_by_provider(self, provider: str) -> List[Dict[str, Any]]:
        """Get all tools from a specific provider."""
        tools = await self.list_tools()
        return [t for t in tools if t.get("provider") == provider]

    async def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """
        Search tools by name or description.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching tools
        """
        tools = await self.list_tools()
        query_lower = query.lower()
        
        return [
            t for t in tools
            if query_lower in (t.get("name") or "").lower()
            or query_lower in (t.get("description") or "").lower()
            or query_lower in (t.get("provider") or "").lower()
        ]

    def get_tool_methods(self, tool: Dict[str, Any]) -> List[str]:
        """
        Extract available methods from a tool definition.
        
        Args:
            tool: Tool definition from Adapter
            
        Returns:
            List of method names
        """
        capabilities = tool.get("capabilities") or {}
        methods = capabilities.get("methods") or capabilities.get("tools") or capabilities.get("actions") or []
        
        if isinstance(methods, dict):
            return list(methods.keys())
        elif isinstance(methods, list):
            return methods
        else:
            return []


# Singleton instance
_adapter_client: Optional[AdapterToolsClient] = None


def get_adapter_tools_client() -> AdapterToolsClient:
    """Get or create the singleton AdapterToolsClient instance."""
    global _adapter_client
    if _adapter_client is None:
        _adapter_client = AdapterToolsClient()
    return _adapter_client
