"""
PRD-42: Persona Models
======================

Database model for agent personas — predefined and workspace-custom personality
profiles that agents can adopt.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    ARRAY, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer,
    String, Text, func,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from core.database.base import Base


class Persona(Base):
    """Agent persona profile (global predefined or workspace-custom)."""
    __tablename__ = "personas"
    __table_args__ = (
        Index("idx_personas_scope", "scope", "workspace_id"),
        Index("idx_personas_category", "category"),
        {"extend_existing": True},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    slug = Column(String(100), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    system_prompt = Column(Text)
    voice_description = Column(String(500))
    category = Column(String(100))
    tags = Column(ARRAY(String), default=list)
    suggested_temperature = Column(Float, default=0.7)
    suggested_models = Column(ARRAY(String), default=list)
    source = Column(String(100))  # 'manual', 'imported', etc.
    source_url = Column(String(500))
    scope = Column(String(20), default="global")  # 'global' or 'workspace'
    workspace_id = Column(PGUUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
