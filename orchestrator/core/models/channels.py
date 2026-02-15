"""
Channel Connection model (PRD-55: Autonomous Assistant Platform).

Stores connection configuration for external messaging platforms
(Telegram, Slack, Discord) linked to a workspace.
"""

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func
from core.database.base import Base
import uuid


class ChannelConnection(Base):
    __tablename__ = "channel_connections"
    __table_args__ = {"extend_existing": True}

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    platform = Column(String(50), nullable=False)  # telegram, slack, discord
    config = Column(JSONB, nullable=False, default=dict)
    status = Column(String(20), nullable=False, default="disconnected")
    metadata_ = Column("metadata", JSONB, default=dict)
    default_agent_id = Column(Integer, nullable=True)
    message_count = Column(Integer, default=0)
    last_activity_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
