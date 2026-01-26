"""ComposioAPIService - Clean wrapper for Composio API interactions.

This service handles all direct communication with the Composio API.
It is used sparingly - most data comes from local cache.

Usage:
    - Connect/disconnect apps (OAuth flows)
    - Enable/disable actions
    - Execute actions on connected apps
    - Sync metadata during scheduled jobs
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Use httpx if available, fallback to requests
try:
    import httpx
    HTTP_CLIENT = "httpx"
except ImportError:
    import requests
    HTTP_CLIENT = "requests"

logger = logging.getLogger(__name__)


class ComposioAuthScheme(str, Enum):
    """Supported authentication schemes."""
    OAUTH2 = "OAUTH2"
    API_KEY = "API_KEY"
    BEARER = "BEARER"
    BASIC = "BASIC"


@dataclass
class ComposioConfig:
    """Configuration for Composio API."""
    api_key: str
    base_url: str = "https://backend.composio.dev/api/v2"
    timeout: int = 30
    max_retries: int = 3


@dataclass
class ComposioApp:
    """Represents a Composio app."""
    name: str
    key: str
    display_name: str
    description: Optional[str] = None
    logo: Optional[str] = None
    categories: List[str] = None
    auth_schemes: List[str] = None
    actions_count: int = 0
    triggers_count: int = 0
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "ComposioApp":
        """Create from API response."""
        return cls(
            name=data.get("name", "").upper(),
            key=data.get("key", data.get("name", "").lower()),
            display_name=data.get("displayName", data.get("name", "")),
            description=data.get("description"),
            logo=data.get("logo"),
            categories=data.get("categories", []),
            auth_schemes=data.get("auth_schemes", []),
            actions_count=data.get("actionsCount", 0),
            triggers_count=data.get("triggersCount", 0),
        )


@dataclass
class ComposioAction:
    """Represents a Composio action."""
    name: str
    app_name: str
    display_name: str
    description: Optional[str] = None
    parameters: Optional[Dict] = None
    tags: List[str] = None
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any], app_name: str) -> "ComposioAction":
        """Create from API response."""
        return cls(
            name=data.get("name", ""),
            app_name=app_name,
            display_name=data.get("displayName", data.get("name", "")),
            description=data.get("description"),
            parameters=data.get("parameters", {}),
            tags=data.get("tags", []),
        )


class ComposioAPIService:
    """Service for interacting with Composio API.
    
    This service is used for:
    1. OAuth flows (connect/disconnect apps)
    2. Action execution
    3. Metadata sync (scheduled job only)
    
    NOTE: For reading app/action metadata, always use the local cache first.
    """
    
    def __init__(self, config: Optional[ComposioConfig] = None):
        """Initialize with configuration.
        
        Args:
            config: Optional config. If not provided, reads from environment.
        """
        if config:
            self.config = config
        else:
            api_key = os.environ.get("COMPOSIO_API_KEY", "")
            if not api_key:
                logger.warning("COMPOSIO_API_KEY not set in environment")
            self.config = ComposioConfig(api_key=api_key)
        
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                headers={
                    "X-API-KEY": self.config.api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._client
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    # =========================================================================
    # Entity Management
    # =========================================================================
    
    async def get_or_create_entity(self, entity_id: str) -> Dict[str, Any]:
        """Get or create a Composio entity.
        
        An entity represents a user/workspace in Composio.
        
        Args:
            entity_id: Unique identifier for the entity (typically workspace_id)
            
        Returns:
            Entity data from Composio
        """
        try:
            response = await self.client.get(f"/connectedAccounts", params={"entityId": entity_id})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Create new entity
                response = await self.client.post("/entities", json={"id": entity_id})
                response.raise_for_status()
                return response.json()
            raise
    
    # =========================================================================
    # App Connection (OAuth flows)
    # =========================================================================
    
    async def get_connection_url(
        self,
        entity_id: str,
        app_name: str,
        redirect_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get OAuth connection URL for an app.
        
        Args:
            entity_id: Entity/workspace ID
            app_name: Name of the app to connect (e.g., "GMAIL")
            redirect_url: Optional redirect URL after OAuth
            
        Returns:
            Dict with connection URL and metadata
        """
        payload = {
            "entityId": entity_id,
            "appName": app_name.upper(),
        }
        if redirect_url:
            payload["redirectUrl"] = redirect_url
        
        response = await self.client.post("/connectedAccounts/initiate", json=payload)
        response.raise_for_status()
        data = response.json()
        
        return {
            "connection_url": data.get("redirectUrl") or data.get("connectionUrl"),
            "connection_id": data.get("connectedAccountId") or data.get("connectionId"),
            "status": "pending",
        }
    
    async def get_connection_status(
        self,
        entity_id: str,
        app_name: str,
    ) -> Dict[str, Any]:
        """Check connection status for an app.
        
        Args:
            entity_id: Entity/workspace ID
            app_name: Name of the app
            
        Returns:
            Connection status and details
        """
        response = await self.client.get(
            "/connectedAccounts",
            params={"entityId": entity_id, "appName": app_name.upper()}
        )
        response.raise_for_status()
        data = response.json()
        
        # Find matching connection
        connections = data.get("items", [])
        for conn in connections:
            if conn.get("appName", "").upper() == app_name.upper():
                return {
                    "connected": conn.get("status") == "ACTIVE",
                    "connection_id": conn.get("id"),
                    "status": conn.get("status"),
                    "connected_at": conn.get("createdAt"),
                }
        
        return {"connected": False, "status": "not_connected"}
    
    async def disconnect_app(
        self,
        entity_id: str,
        connection_id: str,
    ) -> bool:
        """Disconnect an app.
        
        Args:
            entity_id: Entity/workspace ID
            connection_id: The connection ID to disconnect
            
        Returns:
            True if successful
        """
        response = await self.client.delete(f"/connectedAccounts/{connection_id}")
        response.raise_for_status()
        return True
    
    async def get_connected_apps(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all connected apps for an entity.
        
        Args:
            entity_id: Entity/workspace ID
            
        Returns:
            List of connected apps with status
        """
        response = await self.client.get(
            "/connectedAccounts",
            params={"entityId": entity_id}
        )
        response.raise_for_status()
        data = response.json()
        
        connected = []
        for conn in data.get("items", []):
            connected.append({
                "app_name": conn.get("appName", "").upper(),
                "connection_id": conn.get("id"),
                "status": conn.get("status"),
                "connected_at": conn.get("createdAt"),
            })
        
        return connected
    
    # =========================================================================
    # Action Management
    # =========================================================================
    
    async def enable_action(
        self,
        entity_id: str,
        action_name: str,
    ) -> bool:
        """Enable an action for an entity.
        
        Args:
            entity_id: Entity/workspace ID
            action_name: Full action name (e.g., "GMAIL_FETCH_EMAILS")
            
        Returns:
            True if successful
        """
        response = await self.client.post(
            f"/actions/{action_name}/enable",
            json={"entityId": entity_id}
        )
        response.raise_for_status()
        return True
    
    async def disable_action(
        self,
        entity_id: str,
        action_name: str,
    ) -> bool:
        """Disable an action for an entity.
        
        Args:
            entity_id: Entity/workspace ID
            action_name: Full action name
            
        Returns:
            True if successful
        """
        response = await self.client.post(
            f"/actions/{action_name}/disable",
            json={"entityId": entity_id}
        )
        response.raise_for_status()
        return True
    
    async def get_enabled_actions(
        self,
        entity_id: str,
        app_name: Optional[str] = None,
    ) -> List[str]:
        """Get list of enabled actions for an entity.
        
        Args:
            entity_id: Entity/workspace ID
            app_name: Optional filter by app name
            
        Returns:
            List of enabled action names
        """
        params = {"entityId": entity_id}
        if app_name:
            params["appName"] = app_name.upper()
        
        response = await self.client.get("/actions/enabled", params=params)
        response.raise_for_status()
        data = response.json()
        
        return [action.get("name") for action in data.get("items", [])]
    
    # =========================================================================
    # Action Execution
    # =========================================================================
    
    async def execute_action(
        self,
        entity_id: str,
        action_name: str,
        parameters: Dict[str, Any],
        connection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a Composio action.
        
        Args:
            entity_id: Entity/workspace ID
            action_name: Full action name (e.g., "GMAIL_FETCH_EMAILS")
            parameters: Action parameters
            connection_id: Optional specific connection to use
            
        Returns:
            Action execution result
        """
        payload = {
            "entityId": entity_id,
            "input": parameters,
        }
        if connection_id:
            payload["connectedAccountId"] = connection_id
        
        response = await self.client.post(
            f"/actions/{action_name}/execute",
            json=payload,
            timeout=60.0,  # Longer timeout for execution
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "success": data.get("successfull", data.get("success", False)),
            "data": data.get("data"),
            "error": data.get("error"),
            "execution_id": data.get("executionId"),
        }
    
    # =========================================================================
    # Metadata Fetching (for sync service)
    # =========================================================================
    
    async def fetch_all_apps(self) -> List[ComposioApp]:
        """Fetch all available apps from Composio.
        
        Used by MetadataSyncService during scheduled sync.
        
        Returns:
            List of all Composio apps
        """
        apps = []
        page = 1
        per_page = 100
        
        while True:
            response = await self.client.get(
                "/apps",
                params={"page": page, "perPage": per_page}
            )
            response.raise_for_status()
            data = response.json()
            
            items = data.get("items", [])
            if not items:
                break
            
            for item in items:
                apps.append(ComposioApp.from_api_response(item))
            
            # Check for more pages
            if len(items) < per_page:
                break
            page += 1
        
        logger.info(f"Fetched {len(apps)} apps from Composio")
        return apps
    
    async def fetch_app_actions(self, app_name: str) -> List[ComposioAction]:
        """Fetch all actions for a specific app.
        
        Args:
            app_name: Name of the app
            
        Returns:
            List of actions for the app
        """
        actions = []
        page = 1
        per_page = 100
        
        while True:
            response = await self.client.get(
                "/actions",
                params={
                    "appName": app_name.upper(),
                    "page": page,
                    "perPage": per_page,
                }
            )
            response.raise_for_status()
            data = response.json()
            
            items = data.get("items", [])
            if not items:
                break
            
            for item in items:
                actions.append(ComposioAction.from_api_response(item, app_name))
            
            if len(items) < per_page:
                break
            page += 1
        
        return actions
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    async def health_check(self) -> bool:
        """Check if Composio API is accessible.
        
        Returns:
            True if API is healthy
        """
        try:
            response = await self.client.get("/apps", params={"perPage": 1})
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Composio health check failed: {e}")
            return False
