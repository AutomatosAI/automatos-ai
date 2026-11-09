"""Memory profiles: what distillation keeps as a fact, by domain (PRD-274, "Automatos for products", §5.6).

A profile is the part of the distillation prompt that says what counts as a durable fact here:
its ``focus`` (one paragraph on what to keep), what to ``keep`` and what to ``skip``, and worked
``examples`` (``{"fact", "type"}``, the types being the platform's own, ``MEMORY_FACT_TYPES``).
``general`` is today's prompt unchanged; the others are named by domain.

Seeded profiles are the platform's (``workspace_id`` NULL, ``core/seeds/seed_memory_profiles.py``).
The column is there for a workspace's own profiles; whether owners may define them is the
PRD's §8-7 decision, so nothing writes one yet. A workspace's profile and a seeded one may share
a slug; the workspace's wins for that workspace.

The Alembic revision ``memory_profiles_20261109`` declares the same shape, and
``tests/test_afp_w4a_memory_profiles.py`` holds the two together.
"""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, Index, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base

__all__ = ["MemoryProfile", "SEEDED_WHERE", "WORKSPACE_WHERE"]

# Exactly the revision's index predicates.
SEEDED_WHERE = "workspace_id IS NULL"
WORKSPACE_WHERE = "workspace_id IS NOT NULL"


class MemoryProfile(Base):
    """One memory profile: a seeded one (no workspace) or a workspace's own."""

    __tablename__ = "memory_profiles"
    __table_args__ = (
        Index("uq_memory_profiles_seeded_slug", "slug", unique=True, postgresql_where=text(SEEDED_WHERE)),
        Index("uq_memory_profiles_workspace_slug", "workspace_id", "slug", unique=True,
              postgresql_where=text(WORKSPACE_WHERE)),
        {"extend_existing": True},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(PGUUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"))
    slug = Column(String(64), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False, server_default=text("''"))
    focus = Column(Text, nullable=False, server_default=text("''"))  # empty for general: today's prompt
    keep = Column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    skip = Column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    examples = Column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
