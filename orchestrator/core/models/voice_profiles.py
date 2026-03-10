"""
PRD-74 Phase 2: Voice Profile Models
=====================================

Database model for workspace voice profiles — predefined and cloned voice
configurations that can be assigned to agents for TTS synthesis.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Index, String, Text, func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import relationship

from core.database.base import Base


class VoiceProfile(Base):
    """Workspace voice profile for TTS synthesis."""
    __tablename__ = "voice_profiles"
    __table_args__ = (
        Index("idx_voice_profiles_workspace", "workspace_id"),
        Index("idx_voice_profiles_provider", "provider"),
        {"extend_existing": True},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("workspaces.id"),
        nullable=False,
    )
    name = Column(Text, nullable=False)
    provider = Column(Text, nullable=False, server_default="kokoro")
    voice_id = Column(Text, nullable=False)
    reference_audio = Column(Text, nullable=True)  # S3 key for cloned voice
    settings = Column(JSONB, server_default="{}", default=dict)
    is_default = Column(Boolean, default=False, server_default="false")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
