"""
Tool Assignment Models (PRD-35)
===============================

SQLAlchemy models and Pydantic schemas for:
- TenantToolConfig: Tool enablement per tenant with 1:1 credential mapping
- AgentToolAssignment: Agent-tool assignments (like skills, 1:many)

Part of PRD-35: Tool Catalog & Registry Architecture
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import relationship

from core.models.core import Base


# ============================================================================
# SQLAlchemy Database Models
# ============================================================================

class WorkspaceToolConfig(Base):
    """
    Tool enablement per workspace with 1:1 credential mapping.
    
    When a user enables a tool in Settings > Tools, they must provide credentials.
    This creates a 1:1 mapping between tool and credential for that workspace.
    
    Adapter owns tool definitions; this table owns enablement and credential mapping.
    """
    __tablename__ = 'workspace_tool_config'
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    
    # Reference to Adapter's tool (not a FK - Adapter is separate DB)
    tool_id = Column(String(255), nullable=False, index=True)  # Clean: 'slack', 'github'
    adapter_tool_name = Column(String(255), nullable=False)  # Cached for display
    
    enabled = Column(Boolean, default=True, nullable=False)
    
    # 1:1 credential mapping - set when tool is enabled
    credential_id = Column(Integer, ForeignKey('credentials.id', ondelete='SET NULL'), nullable=True)
    
    # Workspace-specific tool configuration overrides
    configuration = Column(JSONB, default=dict)

    # [NEW] Cache tool definition (methods, schemas, category) from Adapter
    # This enables "Just-in-Time" loading without live Adapter calls.
    cached_metadata = Column(JSONB, default=dict)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    credential = relationship("Credential", foreign_keys=[credential_id])
    
    def __repr__(self):
        return f"<WorkspaceToolConfig(workspace={self.workspace_id}, tool={self.tool_id}, enabled={self.enabled})>"


class AgentToolAssignment(Base):
    """
    Agent-tool assignments (like skills, 1:many).
    
    Agents can be assigned multiple tools. Tools must first be enabled
    at the tenant level (TenantToolConfig) before they can be assigned to agents.
    
    This is similar to how skills work - agents have a many-to-many relationship
    with tools through this assignment table.
    """
    __tablename__ = 'agent_tool_assignments'
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey('agents.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Reference to Adapter's tool (not a FK - Adapter is separate DB)
    tool_id = Column(String(255), nullable=False, index=True)  # Clean: 'slack', 'github'
    
    enabled = Column(Boolean, default=True, nullable=False)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    agent = relationship("Agent", back_populates="tool_assignments")
    
    def __repr__(self):
        return f"<AgentToolAssignment(agent={self.agent_id}, tool={self.tool_id}, enabled={self.enabled})>"


# ============================================================================
# Pydantic Models for API
# ============================================================================

# --- Tool Definition (from Adapter) ---

class AdapterToolDefinition(BaseModel):
    """Tool definition from Unified Adapter."""
    adapter_tool_id: str = Field(..., alias="name")
    name: str
    description: Optional[str] = None
    provider: str
    category: str
    adapter_type: str  # 'rest' or 'mcp'
    credential_type: Optional[str] = None
    auth_config: Optional[Dict[str, Any]] = None
    icon: Optional[str] = None
    tags: Optional[List[str]] = None
    capabilities: Optional[Dict[str, Any]] = None
    
    class Config:
        populate_by_name = True


# --- Tool Enablement (Tenant Level) ---

class ToolEnableRequest(BaseModel):
    """Request to enable a tool for a tenant."""
    tool_id: str = Field(..., description="Tool ID (e.g., 'slack', 'github')")
    credentials: Dict[str, Any] = Field(..., description="Credential values for this tool")
    configuration: Optional[Dict[str, Any]] = Field(default={}, description="Optional tool configuration")


class ToolDisableRequest(BaseModel):
    """Request to disable a tool for a tenant."""
    tool_id: str = Field(..., description="Tool ID (e.g., 'slack', 'github')")


class WorkspaceToolConfigResponse(BaseModel):
    """Response for workspace tool configuration."""
    id: int
    workspace_id: UUID
    tool_id: str
    tool_name: str
    enabled: bool
    credential_id: Optional[int] = None
    credential_configured: bool = False
    configuration: Dict[str, Any] = {}
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class AvailableToolResponse(BaseModel):
    """Tool available for enablement (from Adapter + tenant status)."""
    tool_id: str
    name: str
    description: Optional[str] = None
    provider: str
    category: str
    credential_type: Optional[str] = None
    auth_config: Optional[Dict[str, Any]] = None
    icon: Optional[str] = None
    tags: Optional[List[str]] = None
    # Workspace-specific status
    enabled_for_workspace: bool = False
    credential_configured: bool = False
    workspace_config_id: Optional[int] = None


# --- Agent Tool Assignment ---

class AgentToolAssignRequest(BaseModel):
    """Request to assign a tool to an agent."""
    tool_id: str = Field(..., alias='adapter_tool_id', description="Tool ID (e.g., 'slack', 'github')")
    enabled: bool = Field(default=True, description="Whether this assignment is enabled")
    
    class Config:
        populate_by_name = True  # Accept both 'tool_id' and 'adapter_tool_id'


class AgentToolAssignBatchRequest(BaseModel):
    """Request to batch update agent tool assignments."""
    assignments: List[AgentToolAssignRequest]


class AgentToolAssignmentResponse(BaseModel):
    """Response for agent tool assignment."""
    id: int
    agent_id: int
    tool_id: str
    enabled: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class AgentToolResponse(BaseModel):
    """Tool assigned to an agent with full details."""
    id: int
    tool_id: str
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    icon: Optional[str] = None
    enabled: bool
    assigned_at: Optional[datetime] = None


# --- Credential Resolution (for Adapter callback) ---

class CredentialResolveRequest(BaseModel):
    """Request from Adapter to resolve credentials."""
    workspace_id: UUID
    tool_name: str
    credential_type: Optional[str] = None
    service_name: str = Field(default="unified-adapter")
    environment: str = Field(default="production")


class CredentialResolveResponse(BaseModel):
    """Response with resolved credential data."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, str]] = None
