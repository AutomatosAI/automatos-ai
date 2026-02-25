"""
Workspace ORM model (PRD-37).

Several models (agents, workflows, messages, etc.) reference `workspaces.id` via
ForeignKey, but this model was missing from SQLAlchemy metadata, causing:

NoReferencedTableError: Foreign key ... could not find table 'workspaces'
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database.base import Base


class Workspace(Base):
    __tablename__ = "workspaces"
    __table_args__ = (
        Index(
            "idx_workspaces_template",
            "is_template",
            postgresql_where="is_template = TRUE",
        ),
        {"extend_existing": True},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True)

    name = Column(String(255), nullable=False)
    slug = Column(String(255))
    owner_id = Column(Integer)
    clerk_org_id = Column(String(255))

    plan = Column(String(50), default="starter")
    plan_limits = Column(JSONB, default=dict)
    settings = Column(JSONB, default=dict)

    is_personal = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)

    # General workspace webhook key (URL-as-secret pattern)
    webhook_key = Column(String(64), unique=True, nullable=True)

    # -- Widget layout persistence (US-001) --
    layout = Column(JSONB, server_default='{"columns":12,"rowHeight":100}', nullable=False)
    layout_mode = Column(String(20), server_default="grid", nullable=False)
    widgets = Column(JSONB, server_default="[]", nullable=False)
    description = Column(Text, nullable=True)

    # -- Template fields --
    is_template = Column(Boolean, server_default="false", nullable=False)
    template_category = Column(String(50), nullable=True)
    template_icon = Column(String(10), nullable=True)

    # -- Visibility --
    visibility = Column(String(20), server_default="private", nullable=False)

    # -- Timestamps --
    last_opened_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships (US-003)
    shares = relationship(
        "WorkspaceShare",
        back_populates="workspace",
        cascade="all, delete-orphan",
    )

