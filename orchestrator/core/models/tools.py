
"""
Tools Management Database Models
================================

Comprehensive data models for tools, credentials, configurations, and permissions
for the platform tool registry and marketplace UI.
"""

from enum import Enum

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
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
    
    # Relationships
    credentials = relationship("ToolCredentials", back_populates="tool", cascade="all, delete-orphan")
    configurations = relationship("ToolConfiguration", back_populates="tool", cascade="all, delete-orphan")
    permissions_assignments = relationship("AgentToolPermission", back_populates="tool", cascade="all, delete-orphan")
    # Note: Usage tracking is implemented separately (not tied to legacy tool tables).

class ToolCredentials(Base):
    """
    Encrypted storage for tool credentials per environment.
    Supports multiple credential types and automatic rotation.
    """
    __tablename__ = 'tool_credentials'
    __table_args__ = {'extend_existing': True}  # PRD-17: Prevent duplicate table definition errors
    
    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey('tools.id'), nullable=False, index=True)
    
    # Credential identification
    credential_key = Column(String(100), nullable=False)  # api_token, secret_key, etc.
    credential_value = Column(Text, nullable=False)  # Encrypted value
    environment = Column(String(50), nullable=False, index=True)  # development, staging, production
    
    # Metadata
    description = Column(Text)
    expires_at = Column(DateTime)  # For credential rotation
    is_active = Column(Boolean, default=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    tool = relationship("Tool", back_populates="credentials")

class ToolConfiguration(Base):
    """
    Tool-specific configuration settings per environment.
    Stores non-sensitive configuration data and settings.
    """
    __tablename__ = 'tool_configurations'
    __table_args__ = {'extend_existing': True}  # PRD-17: Prevent duplicate table definition errors
    
    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey('tools.id'), nullable=False, index=True)
    environment = Column(String(50), nullable=False, index=True)
    
    # Configuration data
    configuration = Column(JSON, default=dict)  # Tool-specific settings
    is_active = Column(Boolean, default=True, index=True)
    
    # Health and status
    last_health_check = Column(DateTime)
    health_status = Column(String(50), default='unknown')  # healthy, unhealthy, unknown
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    tool = relationship("Tool", back_populates="configurations")

class AgentToolPermission(Base):
    """
    Junction table managing which agents can access which tools.
    Implements role-based access control with environment separation.
    """
    __tablename__ = 'agent_tool_permissions'
    __table_args__ = {'extend_existing': True}  # PRD-17: Prevent duplicate table definition errors
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, ForeignKey('agents.id'), nullable=False, index=True)
    tool_id = Column(Integer, ForeignKey('tools.id'), nullable=False, index=True)
    environment = Column(String(50), nullable=False, index=True)
    
    # Permission details
    permissions = Column(ARRAY(String), default=list)  # Specific permissions within the tool
    is_active = Column(Boolean, default=True, index=True)
    expires_at = Column(DateTime)  # For time-limited access
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    tool = relationship("Tool", back_populates="permissions_assignments")
    agent = relationship("Agent")  # Assumes Agent model exists in main models.py

# This file defines Tool model for the UI tool registry

# ====================================
# Audit and Security Models
# ====================================

# CredentialAuditLog is defined in credentials.py - removed duplicate definition

class PermissionAuditLog(Base):
    """
    Audit log for permission changes and access control events.
    Maintains compliance trail for agent-tool permission modifications.
    """
    __tablename__ = 'permission_audit_logs'
    __table_args__ = {'extend_existing': True}  # PRD-17: Prevent duplicate table definition errors
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, ForeignKey('agents.id'), nullable=False, index=True)
    tool_id = Column(Integer, ForeignKey('tools.id'), nullable=False, index=True)
    
    # Audit details
    action = Column(String(100), nullable=False, index=True)  # assign, revoke, modify, access
    environment = Column(String(50), index=True)
    user_id = Column(String(255), index=True)  # Who performed the action
    
    # Context and metadata
    details = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=func.now(), index=True)
    
    # Relationships
    agent = relationship("Agent")
    tool = relationship("Tool")

# ====================================
# Tool Marketplace Models
# ====================================

class ToolCategory(Base):
    """
    Tool categories for marketplace organization.
    Provides hierarchical categorization of tools.
    """
    __tablename__ = 'tool_categories'
    __table_args__ = {'extend_existing': True}  # PRD-17: Prevent duplicate table definition errors
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    icon = Column(String(50))  # Category icon
    parent_id = Column(Integer, ForeignKey('tool_categories.id'))  # For hierarchical categories
    sort_order = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    parent = relationship("ToolCategory", remote_side=[id])
    children = relationship("ToolCategory", back_populates="parent")

class ToolReview(Base):
    """
    User reviews and ratings for tools.
    Supports marketplace-style feedback system.
    """
    __tablename__ = 'tool_reviews'
    __table_args__ = {'extend_existing': True}  # PRD-17: Prevent duplicate table definition errors
    
    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey('tools.id'), nullable=False, index=True)
    
    # Review content
    rating = Column(Integer, nullable=False)  # 1-5 stars
    title = Column(String(255))
    review_text = Column(Text)
    
    # Reviewer information
    reviewer_id = Column(String(255), index=True)  # User/agent identifier
    reviewer_type = Column(String(50))  # user, agent, system
    
    # Review metadata
    is_verified = Column(Boolean, default=False)  # Verified purchase/usage
    is_featured = Column(Boolean, default=False)  # Featured review
    helpful_votes = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    tool = relationship("Tool")

class ToolInstallationRequest(Base):
    """
    Tracks tool installation requests and their status.
    Manages the installation workflow and approval process.
    """
    __tablename__ = 'tool_installation_requests'
    __table_args__ = {'extend_existing': True}  # PRD-17: Prevent duplicate table definition errors
    
    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey('tools.id'), nullable=False, index=True)
    
    # Request details
    requested_by = Column(String(255), nullable=False, index=True)  # User/agent identifier
    environment = Column(String(50), nullable=False)
    justification = Column(Text)  # Why the tool is needed
    
    # Approval workflow
    status = Column(String(50), default='pending', index=True)  # pending, approved, rejected, installed
    approved_by = Column(String(255), index=True)
    approval_notes = Column(Text)
    
    # Installation details
    installation_config = Column(JSON, default=dict)
    auto_configure = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    approved_at = Column(DateTime)
    installed_at = Column(DateTime)
    
    # Relationships
    tool = relationship("Tool")


