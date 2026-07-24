
"""
Tools Management Database Models
================================

The ``tools`` registry table backing the marketplace UI metadata and the
PRD-123 tool-tier policy check (``modules/tools/registry/tool_registry.py``
resolves a tool's tier from this model when the in-memory spec doesn't carry
one).

PRD-195 S8 (P2-14): the agent-tool RBAC fossil — the per-agent tool-permission
and permission-audit-log models (vocabulary #5, workspace-blind by schema) and
the orphaned tool-configuration / tool-category models — was deleted with
``api/permissions.py``; their tables are dropped by
``alembic/versions/prd195_drop_authz_fossil_tables.py``.
"""

from enum import Enum

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, ARRAY
from sqlalchemy.sql import func
from .core import Base


class ToolTier(str, Enum):
    """PRD-123 Pattern #4: Tool trust tiers for access policy enforcement."""

    SYSTEM = "system"          # Always available (platform internals like RAG, MEMORY)
    PLATFORM = "platform"      # Available by default, can be disabled per workspace
    MARKETPLACE = "marketplace" # Requires explicit agent_tool_assignments
    CUSTOM = "custom"          # User-created tools, requires explicit assignment

# ====================================
# Tools Models
# ====================================

class Tool(Base):
    """
    Main tool registry storing available tools and their metadata.
    Supports both marketplace tools and custom integrations.
    """
    __tablename__ = 'tools'
    __table_args__ = {'extend_existing': True}  # PRD-17: Prevent duplicate table definition errors

    # Primary identification
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text)
    category = Column(String(100), nullable=False, index=True)  # developer, communication, cloud, etc.
    provider = Column(String(255), nullable=False)
    version = Column(String(100), nullable=False)

    # Display and metadata
    icon = Column(String(50))  # Emoji or icon identifier
    logo = Column(String(255))  # Logo file path, e.g. "/logos/Discord.png"
    pricing = Column(String(100))  # Free, Pro, Pay-per-use, etc.
    rating = Column(Float, default=0.0)
    tags = Column(ARRAY(String), default=list)

    # Status and availability
    status = Column(String(50), default='available', index=True)  # available, deprecated, maintenance, beta
    tier = Column(String(20), default='marketplace', index=True)  # PRD-123 Pattern #4: system|platform|marketplace|custom
    is_installed = Column(Boolean, default=False, index=True)
    is_configured = Column(Boolean, default=False, index=True)

    # Usage and permissions
    usage_count = Column(Integer, default=0)
    permissions = Column(ARRAY(String), default=list)  # Allowed agent types
    required_credentials = Column(ARRAY(String), default=list)  # Required credential keys
    supported_environments = Column(ARRAY(String), default=list)  # dev, staging, prod

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())

    # Note: Usage tracking is implemented separately (not tied to legacy tool tables).
