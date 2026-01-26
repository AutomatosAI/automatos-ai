"""
Composio ORM models (PRD-36).

This repository previously referenced `core.models.composio` from multiple routers/services,
but the module was missing. These lightweight SQLAlchemy models provide the minimum tables
needed to persist:
- workspace-scoped Composio entities
- app connection status per entity (connected tools)
- optional feature toggles and trigger subscriptions

They are intentionally simple and safe for local/dev use.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from core.database.base import Base


class ComposioEntity(Base):
    __tablename__ = "composio_entities"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    workspace_id = Column(PGUUID(as_uuid=True), nullable=False, unique=True, index=True)

    # In Composio's newer API shape, the "entity" can just be a stable user_id string.
    # We store it explicitly so we can change strategy later without breaking data.
    composio_entity_id = Column(String(255), nullable=False, unique=True, index=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ComposioConnection(Base):
    __tablename__ = "composio_connections"
    __table_args__ = (
        UniqueConstraint("entity_id", "app_name", name="uq_composio_entity_app"),
        {"extend_existing": True},
    )

    id = Column(Integer, primary_key=True)
    entity_id = Column(Integer, ForeignKey("composio_entities.id", ondelete="CASCADE"), nullable=False, index=True)

    app_name = Column(String(100), nullable=False, index=True)  # stored uppercase
    status = Column(String(50), nullable=False, default="pending")  # pending|active|error|disconnected
    connected_at = Column(DateTime)
    expires_at = Column(DateTime)
    connection_id = Column(String(255))
    # DB column is named "connection_metadata" (avoid SQLAlchemy reserved "metadata")
    connection_metadata = Column("connection_metadata", JSONB, default=dict)
    last_synced_at = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class AgentAppFeature(Base):
    __tablename__ = "agent_app_features"
    __table_args__ = (
        UniqueConstraint("agent_id", "app_name", "action_name", name="uq_agent_app_action"),
        {"extend_existing": True},
    )

    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, nullable=False, index=True)
    app_name = Column(String(100), nullable=False, index=True)
    action_name = Column(String(255), nullable=False, index=True)
    enabled = Column(Boolean, default=True, nullable=False)
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class TriggerSubscription(Base):
    __tablename__ = "trigger_subscriptions"
    __table_args__ = (
        UniqueConstraint("entity_id", "trigger_name", name="uq_entity_trigger"),
        {"extend_existing": True},
    )

    id = Column(Integer, primary_key=True)
    entity_id = Column(Integer, ForeignKey("composio_entities.id", ondelete="CASCADE"), nullable=False, index=True)

    trigger_name = Column(String(255), nullable=False, index=True)
    callback_url = Column(String(500), nullable=False)
    composio_subscription_id = Column(String(255))

    agent_id: Optional[int] = Column(Integer)
    workflow_id: Optional[int] = Column(Integer)
    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

