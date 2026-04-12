"""
PRD-130: Business Profile Model
================================

ORM model for the business_profiles table created by
prd130_business_profile.py migration.

One row per workspace per wizard run. Updated in-place as the wizard
progresses through scan → scrape → profile → plan.
"""

from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID

from core.database.base import Base


class BusinessProfile(Base):
    """A scraped + enriched view of a workspace's business, used for Mission Zero."""

    __tablename__ = "business_profiles"
    __table_args__ = (
        Index("ix_business_profiles_workspace_status", "workspace_id", "status"),
        {"extend_existing": True},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    )

    domain = Column(Text, nullable=False)
    archetype = Column(Text, nullable=True)
    company_name = Column(Text, nullable=True)

    sectors = Column(JSONB, nullable=True)
    brands = Column(JSONB, nullable=True)
    standards = Column(JSONB, nullable=True)
    voice_notes = Column(Text, nullable=True)
    goals = Column(JSONB, nullable=True)

    raw_map_urls = Column(JSONB, nullable=True)
    selected_urls = Column(JSONB, nullable=True)
    quality_findings = Column(JSONB, nullable=True)
    draft_plan = Column(JSONB, nullable=True)

    # status flow:
    #   started → scanning → scanned → scraping → profiled → planned
    status = Column(Text, nullable=False, server_default="started")

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
