"""
Composio metadata cache models (Redesign schema).

These tables back the new /api/tools/* endpoints:
- composio_apps_cache
- composio_actions_cache
- composio_stats_cache
- composio_sync_jobs

IMPORTANT:
- The physical columns include JSONB columns named "metadata" which clashes with
  SQLAlchemy's reserved attribute name "metadata". We map those columns to
  *_metadata attributes instead.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
    Float,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import relationship

from core.database.base import Base


class ComposioAppCache(Base):
    __tablename__ = "composio_apps_cache"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    app_name = Column(String(100), nullable=False, unique=True)  # Stored uppercase in practice
    app_slug = Column(String(100), nullable=False, unique=True)  # Stored lowercase
    display_name = Column(String(255), nullable=False)
    description = Column(Text)
    logo_url = Column(Text)
    categories = Column(JSONB, nullable=False, default=list)
    auth_schemes = Column(JSONB, nullable=False, default=list)
    action_count = Column(Integer, default=0)
    trigger_count = Column(Integer, default=0)
    status = Column(String(50), default="ACTIVE")

    # DB column is named "metadata"
    app_metadata = Column("metadata", JSONB, default=dict)

    last_synced_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    actions = relationship(
        "ComposioActionCache",
        primaryjoin="ComposioAppCache.app_name==ComposioActionCache.app_name",
        back_populates="app",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "app_name": self.app_name,
            "app_slug": self.app_slug,
            "display_name": self.display_name,
            "description": self.description,
            "logo_url": self.logo_url,
            "categories": self.categories or [],
            "auth_schemes": self.auth_schemes or [],
            "action_count": self.action_count or 0,
            "trigger_count": self.trigger_count or 0,
            "status": self.status,
            "last_synced_at": self.last_synced_at.isoformat() if self.last_synced_at else None,
        }


class ComposioActionCache(Base):
    __tablename__ = "composio_actions_cache"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    app_name = Column(String(100), ForeignKey("composio_apps_cache.app_name", ondelete="CASCADE"), nullable=False)
    action_name = Column(String(255), nullable=False)
    action_slug = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    description = Column(Text)
    parameters = Column(JSONB, default=dict)
    response_schema = Column(JSONB, default=dict)
    tags = Column(JSONB, default=list)
    category = Column(String(100), index=True)
    is_premium = Column(Boolean, default=False)

    # Dynamic tool suggestions (PRD-40)
    app_suggestions = Column(JSONB, default=list)

    # DB column is named "metadata"
    action_metadata = Column("metadata", JSONB, default=dict)

    last_synced_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    app = relationship("ComposioAppCache", back_populates="actions")

    __table_args__ = (
        UniqueConstraint("app_name", "action_name", name="uq_app_action"),
        Index("idx_actions_cache_app_name", "app_name"),
    )


class ComposioStatsCache(Base):
    __tablename__ = "composio_stats_cache"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    stat_key = Column(String(100), nullable=False, unique=True, index=True)
    stat_value = Column(JSONB, nullable=False)
    last_updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ComposioSyncJob(Base):
    __tablename__ = "composio_sync_jobs"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    job_type = Column(String(50), nullable=False)  # full_sync, incremental_sync, app_sync
    status = Column(String(50), nullable=False, default="pending")  # pending, running, completed, failed
    apps_synced = Column(Integer, default=0)
    actions_synced = Column(Integer, default=0)
    errors_count = Column(Integer, default=0)
    error_details = Column(JSONB)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_ms = Column(Integer)

    # DB column is named "metadata"
    job_metadata = Column("metadata", JSONB, default=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_type": self.job_type,
            "status": self.status,
            "apps_synced": self.apps_synced,
            "actions_synced": self.actions_synced,
            "errors_count": self.errors_count,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
        }

# ---------------------------------------------------------------------------
# Tool routing + assignment models (schema v2)
# ---------------------------------------------------------------------------


class AgentAppAssignment(Base):
    """Agent-to-app assignments.
    
    Tracks which apps/tools are assigned to which agents.
    Supports both external (Composio) and internal tools.
    """
    __tablename__ = "agent_app_assignments"
    __table_args__ = {"extend_existing": True}
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    app_name = Column(String(100), nullable=False)
    app_type = Column(String(50), nullable=False, default="EXTERNAL")  # EXTERNAL or INTERNAL
    assigned_at = Column(DateTime, default=datetime.utcnow)
    assigned_by = Column(Integer)  # user_id
    is_active = Column(Boolean, default=True)
    priority = Column(Integer, default=0)  # Higher = preferred
    config = Column(JSONB, default=dict)  # App-specific configuration
    
    __table_args__ = (
        UniqueConstraint("agent_id", "app_name", name="uq_agent_app"),
        Index("idx_agent_assignments_agent", "agent_id"),
        Index("idx_agent_assignments_active", "agent_id", "is_active"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "app_name": self.app_name,
            "app_type": self.app_type,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "is_active": self.is_active,
            "priority": self.priority,
            "config": self.config or {},
        }


class ToolExecutionLog(Base):
    """Audit trail for tool usage.
    
    Logs every tool execution for analytics and debugging.
    """
    __tablename__ = "tool_execution_logs"

    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    app_name = Column(String(100), nullable=False)
    action_name = Column(String(255), nullable=False)
    user_id = Column(Integer)
    workspace_id = Column(PGUUID(as_uuid=True))

    # Request
    input_parameters = Column(JSONB)
    user_query = Column(Text)

    # Response
    output_result = Column(JSONB)
    status = Column(String(50), nullable=False)  # 'success', 'error', 'timeout'
    error_message = Column(Text)
    error_code = Column(String(100))

    # Performance
    execution_time_ms = Column(Integer)
    token_usage = Column(JSONB)  # {"input": 100, "output": 50, "total": 150}

    # PRD-123 Pattern #12: Cost tracking
    estimated_cost = Column(Float, default=0.0)  # Estimated cost in USD
    rate_limit_remaining = Column(Integer, nullable=True)  # Provider rate limit remaining
    # Note: execution_ms column exists in DB (migration prd123_cost_tracking) but is
    # intentionally unmapped — use execution_time_ms instead to avoid ambiguity.

    # PRD-139: Tool routing telemetry columns
    intent_cluster_id = Column(Integer, ForeignKey("intent_classification_cache.id", ondelete="SET NULL"), nullable=True)
    routing_source = Column(String(20), nullable=True)  # 'llm', 'graph', 'keyword', 'cache'
    telemetry_source = Column(String(20), nullable=True, server_default="production")  # 'production', 'eval', 'replay'

    # Metadata
    router_decision = Column(JSONB)  # How the tool was selected
    cache_hit = Column(Boolean, default=False)
    executed_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_tool_logs_agent", "agent_id"),
        Index("idx_tool_logs_status", "status"),
        Index("idx_tool_logs_executed", "executed_at"),
        {"extend_existing": True},
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "app_name": self.app_name,
            "action_name": self.action_name,
            "status": self.status,
            "execution_time_ms": self.execution_time_ms,
            "cache_hit": self.cache_hit,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
        }


class IntentClassificationCache(Base):
    """Cache for LLM intent classification results.
    
    Caches intent classifications to avoid repeated LLM calls for similar queries.
    """
    __tablename__ = "intent_classification_cache"
    __table_args__ = {"extend_existing": True}
    
    id = Column(Integer, primary_key=True)
    query_text = Column(Text, nullable=False)
    classified_category = Column(String(100), nullable=False)  # EMAIL, CALENDAR, CODE, etc.
    classified_action_type = Column(String(100))  # FETCH, CREATE, UPDATE, DELETE
    confidence = Column(Float)
    reasoning = Column(Text)
    cached_at = Column(DateTime, default=datetime.utcnow)
    hit_count = Column(Integer, default=0)
    last_hit_at = Column(DateTime)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.classified_category,
            "action_type": self.classified_action_type,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


# PRD-142 Wave 5 (WS-AA): ToolExecutionCache was verified dead and removed;
# its table is dropped by alembic/versions/prd142_wave5_drop_dead_tables.py.
