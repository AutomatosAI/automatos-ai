"""
Composio Pydantic Models (PRD-36)
=================================

Data models for Composio integration.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID
from enum import Enum


class ConnectionStatus(str, Enum):
    """Status of a Composio app connection."""
    PENDING = "pending"
    ACTIVE = "active"
    EXPIRED = "expired"
    ERROR = "error"


class ComposioApp(BaseModel):
    """Represents a Composio app (e.g., GitHub, Slack)."""
    name: str = Field(..., description="App identifier (e.g., 'GITHUB', 'SLACK')")
    display_name: str = Field(..., description="Human-readable name")
    description: Optional[str] = None
    logo_url: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    auth_schemes: List[str] = Field(default_factory=list, description="Supported auth methods")
    triggers: List[Dict[str, Any]] = Field(default_factory=list, description="Available triggers")
    trigger_count: int = 0
    is_connected: bool = False
    action_count: int = 0
    
    class Config:
        from_attributes = True


class ComposioAction(BaseModel):
    """Represents an action within a Composio app."""
    name: str = Field(..., description="Action identifier (e.g., 'github_list_repos')")
    display_name: str = Field(..., description="Human-readable name")
    description: Optional[str] = None
    app_name: str = Field(..., description="Parent app name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters schema")
    enabled: bool = True
    
    class Config:
        from_attributes = True


class ComposioEntity(BaseModel):
    """Represents a Composio entity (maps to workspace)."""
    id: Optional[int] = None
    workspace_id: UUID
    composio_entity_id: str = Field(..., description="Composio's entity identifier")
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ComposioConnection(BaseModel):
    """Represents a connection between an entity and an app."""
    id: Optional[int] = None
    entity_id: int
    app_name: str
    connection_id: Optional[str] = None
    status: ConnectionStatus = ConnectionStatus.PENDING
    connected_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        from_attributes = True


class ToolRouterSession(BaseModel):
    """Represents an active Tool Router session."""
    session_id: str
    entity_id: str
    apps: List[str] = Field(default_factory=list, description="Apps included in session")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class AgentAppFeature(BaseModel):
    """Represents a feature toggle for an agent-app-action combination."""
    id: Optional[int] = None
    agent_id: int
    app_name: str
    action_name: str
    enabled: bool = True
    
    class Config:
        from_attributes = True


class TriggerSubscription(BaseModel):
    """Represents a webhook trigger subscription."""
    id: Optional[int] = None
    entity_id: int
    trigger_name: str = Field(..., description="Trigger type (e.g., 'github_pull_request_opened')")
    callback_url: str
    agent_id: Optional[int] = None
    workflow_id: Optional[int] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Request/Response models for API

class InitiateConnectionRequest(BaseModel):
    """Request to initiate OAuth connection."""
    app_name: str
    callback_url: Optional[str] = None


class InitiateConnectionResponse(BaseModel):
    """Response with OAuth redirect URL."""
    redirect_url: str
    connection_id: str


class FeatureToggleRequest(BaseModel):
    """Request to toggle a feature."""
    action_name: str
    enabled: bool


class BatchFeatureToggleRequest(BaseModel):
    """Request to toggle multiple features."""
    features: List[FeatureToggleRequest]


class ExecuteActionRequest(BaseModel):
    """Request to execute a Composio action."""
    action_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    entity_id: Optional[str] = None


class ExecuteActionResponse(BaseModel):
    """Response from action execution."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None


class WebhookPayload(BaseModel):
    """Payload received from Composio webhook."""
    trigger_name: str
    entity_id: str
    app_name: str
    event_data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
