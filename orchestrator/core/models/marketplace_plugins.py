"""
PRD-42: Plugin Marketplace Models
=================================

Database models for the plugin marketplace: categories, plugins, security scans,
sync history, and junction tables for workspace/agent assignments.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    ARRAY, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer,
    String, Text, func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import relationship

from core.database.base import Base


# ============================================================================
# Plugin Categories
# ============================================================================

class PluginCategory(Base):
    """Marketplace plugin category (e.g., Development, DevOps, Marketing)."""
    __tablename__ = "plugin_categories"
    __table_args__ = {"extend_existing": True}

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    slug = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    icon = Column(String(50))
    sort_order = Column(Integer, default=0)
    parent_id = Column(PGUUID(as_uuid=True), ForeignKey("plugin_categories.id"), nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Self-referential relationship
    parent = relationship("PluginCategory", remote_side="PluginCategory.id", backref="children")
    # Plugins in this category
    plugins = relationship("MarketplacePlugin", back_populates="category")


# ============================================================================
# Marketplace Plugin
# ============================================================================

class MarketplacePlugin(Base):
    """A plugin available in the marketplace."""
    __tablename__ = "marketplace_plugins"
    __table_args__ = (
        Index("idx_marketplace_plugins_category", "category_id"),
        Index("idx_marketplace_plugins_approval_active", "approval_status", "is_active"),
        Index("idx_marketplace_plugins_featured", "is_featured", "enable_count"),
        {"extend_existing": True},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    slug = Column(String(100), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)

    # S3 storage
    s3_bucket = Column(String(255), default="automatos-marketplace")
    s3_path = Column(String(500))

    # Descriptions
    description = Column(Text)
    long_description = Column(Text)

    # Classification
    category_id = Column(PGUUID(as_uuid=True), ForeignKey("plugin_categories.id"), nullable=True)
    tags = Column(ARRAY(String), default=list)

    # Content counts
    skills_count = Column(Integer, default=0)
    commands_count = Column(Integer, default=0)
    agents_count = Column(Integer, default=0)
    hooks_count = Column(Integer, default=0)

    # Source information
    source_type = Column(String(50))  # 'upload', 'github', etc.
    source_url = Column(String(500))
    source_repo = Column(String(500))
    original_author = Column(String(255))
    license = Column(String(100))

    # Model recommendations
    token_estimate = Column(Integer, default=0)
    recommended_models = Column(ARRAY(String), default=list)

    # Security
    security_scan_id = Column(PGUUID(as_uuid=True), ForeignKey("plugin_security_scans.id"), nullable=True)
    security_status = Column(String(50), default="pending")  # pending, safe, review_required, blocked

    # Approval workflow
    is_active = Column(Boolean, default=True)
    is_featured = Column(Boolean, default=False)
    approval_status = Column(String(50), default="pending")  # pending, approved, rejected
    approved_by = Column(String(255))
    approved_at = Column(DateTime, nullable=True)
    rejection_reason = Column(Text)

    # Usage
    enable_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    category = relationship("PluginCategory", back_populates="plugins")
    security_scan = relationship("PluginSecurityScan", foreign_keys=[security_scan_id])
    sync_history = relationship("PluginSyncHistory", back_populates="plugin")


# ============================================================================
# Plugin Security Scan
# ============================================================================

class PluginSecurityScan(Base):
    """Results from static + LLM security scanning of a plugin."""
    __tablename__ = "plugin_security_scans"
    __table_args__ = {"extend_existing": True}

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    plugin_slug = Column(String(100), nullable=False)
    plugin_version = Column(String(50), nullable=False)

    # Static scan results
    static_scan_status = Column(String(50))  # passed, flagged
    static_findings = Column(JSONB, default=list)
    blocked_patterns_found = Column(ARRAY(String), default=list)

    # LLM scan results
    llm_scan_status = Column(String(50))  # passed, flagged, failed
    llm_risk_score = Column(Integer)
    llm_findings = Column(JSONB, default=list)
    llm_summary = Column(Text)
    llm_model_used = Column(String(100))
    llm_tokens_used = Column(Integer)

    # Overall
    overall_verdict = Column(String(50))  # safe, review_required, blocked

    # Metadata
    scanned_at = Column(DateTime, server_default=func.now(), nullable=False)
    scanned_by = Column(String(255))


# ============================================================================
# Plugin Sync History
# ============================================================================

class PluginSyncHistory(Base):
    """Audit log for plugin lifecycle events (upload, approve, reject, deactivate)."""
    __tablename__ = "plugin_sync_history"
    __table_args__ = {"extend_existing": True}

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    action = Column(String(50), nullable=False)  # upload, approve, reject, deactivate
    plugin_id = Column(PGUUID(as_uuid=True), ForeignKey("marketplace_plugins.id"), nullable=True)
    plugin_slug = Column(String(100))
    status = Column(String(50))
    started_at = Column(DateTime, server_default=func.now(), nullable=False)
    completed_at = Column(DateTime, nullable=True)
    details = Column(JSONB, default=dict)
    error_message = Column(Text)
    performed_by = Column(String(255))

    # Relationships
    plugin = relationship("MarketplacePlugin", back_populates="sync_history")
