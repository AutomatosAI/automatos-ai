"""
Workspace Integration Configuration
==================================

Lightweight DB model used by `/api/tools/*` to persist per-workspace configuration
for a given integration (e.g., enabled actions for a Composio app).

Note: This table is intentionally *decoupled* from external tool catalogs.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

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
    
    The platform owns enablement and optional credential mapping per workspace.
    """
    __tablename__ = 'workspace_tool_config'
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    
    # Integration identifier (e.g., 'gmail', 'slack')
    tool_id = Column(String(255), nullable=False, index=True)
    # Human display name (DB column kept for backwards compatibility)
    display_name = Column("adapter_tool_name", String(255), nullable=False)
    
    enabled = Column(Boolean, default=True, nullable=False)
    
    # 1:1 credential mapping - set when tool is enabled
    credential_id = Column(Integer, ForeignKey('credentials.id', ondelete='SET NULL'), nullable=True)
    
    # Workspace-specific tool configuration overrides
    configuration = Column(JSONB, default=dict)

    # Optional cached metadata for UI display (no external calls required).
    cached_metadata = Column(JSONB, default=dict)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    credential = relationship("Credential", foreign_keys=[credential_id])
    
    def __repr__(self):
        return f"<WorkspaceToolConfig(workspace={self.workspace_id}, tool={self.tool_id}, enabled={self.enabled})>"
