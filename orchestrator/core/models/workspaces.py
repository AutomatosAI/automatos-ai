"""
Workspace ORM model (PRD-37).

Several models (agents, workflows, messages, etc.) reference `workspaces.id` via
ForeignKey, but this model was missing from SQLAlchemy metadata, causing:

NoReferencedTableError: Foreign key ... could not find table 'workspaces'
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, text
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

    # PRD-222 W1S1: server-side Auto-led onboarding state machine + funnel record.
    # The ONE new column for the whole PRD-222 Wave-1 branch — trial ledger,
    # segment answers, and per-stage timestamps all live inside this JSONB doc.
    # Owned/written exclusively by services/onboarding_state.py (rebuild-don't-mutate).
    onboarding = Column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb"), default=dict
    )

    is_personal = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)

    # General workspace webhook key (URL-as-secret pattern)
    webhook_key = Column(String(64), unique=True, nullable=True)

    # Admin lifecycle (pause / soft-delete) — see add_workspace_admin_lifecycle_fields migration
    paused_at = Column(DateTime, nullable=True)
    paused_reason = Column(Text, nullable=True)
    deleted_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
