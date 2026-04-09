"""
SDK API Key ORM model.

Stores hashed API keys issued to workspaces for SDK / widget embedding access.
Each key has a type (public or server), optional domain/IP restrictions, and
per-key rate-limit overrides.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID as PGUUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database.base import Base


class SdkApiKey(Base):
    __tablename__ = "sdk_api_keys"
    __table_args__ = (
        Index("idx_sdk_api_keys_workspace_id", "workspace_id"),
        {"extend_existing": True},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True)

    workspace_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    )

    name = Column(String(200), nullable=False)
    key_prefix = Column(String(16), nullable=False)
    key_hash = Column(String(64), nullable=False, unique=True, index=True)
    key_type = Column(String(20), nullable=False)  # "public" | "server"

    # Permissions & rate limits
    permissions = Column(ARRAY(String), nullable=True)
    rate_limit_requests = Column(Integer, nullable=True)
    rate_limit_tokens = Column(Integer, nullable=True)

    # Agent lock — force all widget chats to use this agent
    default_agent_id = Column(Integer, nullable=True)

    # PRD-124: Team lock — scope all requests through this key to a specific team
    team = Column(String(100), nullable=True)

    # Access restrictions
    allowed_domains = Column(ARRAY(String), nullable=True)
    allowed_ips = Column(ARRAY(String), nullable=True)

    # Status
    is_active = Column(Boolean, server_default="true", nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    workspace = relationship("Workspace", backref="sdk_api_keys", lazy="select")
