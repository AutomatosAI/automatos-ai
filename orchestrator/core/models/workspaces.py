"""
Workspace ORM model (PRD-37).

Several models (agents, workflows, messages, etc.) reference `workspaces.id` via
ForeignKey, but this model was missing from SQLAlchemy metadata, causing:

NoReferencedTableError: Foreign key ... could not find table 'workspaces'
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base


class Workspace(Base):
    __tablename__ = "workspaces"
    __table_args__ = {"extend_existing": True}

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

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

