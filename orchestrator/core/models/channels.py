"""
Channel Connection Model (PRD-55 US-019)
==========================================
Stores per-workspace channel connections (Telegram, Slack, Discord, etc.)
with encrypted credentials and optional default agent routing.
"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import JSON, UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base


class ChannelConnection(Base):
    __tablename__ = "channel_connections"
    __table_args__ = {"extend_existing": True}

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    platform = Column(String(50), nullable=False)  # telegram, slack, discord
    config = Column(JSON)  # encrypted credentials
    status = Column(String(20), default="inactive")  # active, inactive, error
    metadata_ = Column("metadata", JSON, default={})
    default_agent_id = Column(Integer, nullable=True)  # US-027: default routing
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return (
            f"<ChannelConnection id={self.id} platform={self.platform!r} "
            f"workspace={self.workspace_id} status={self.status!r}>"
        )
