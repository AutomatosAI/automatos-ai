"""
SQLAlchemy ORM model for the memory_short_term table (L2 layer).

Used by UnifiedMemoryService for L2 CRUD operations.
Table created by migration: alembic/versions/prd79_memory_short_term.py
"""

from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func
from uuid import uuid4

from core.database.base import Base


class MemoryShortTerm(Base):
    """L2 short-term memory — raw exchanges with Ebbinghaus decay scoring."""

    __tablename__ = "memory_short_term"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agent_id = Column(
        Integer,
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Content
    content = Column(Text, nullable=False)
    content_type = Column(
        String(30),
        nullable=False,
        default="exchange",
    )
    # Valid content_type values:
    #   exchange, recipe_summary, heartbeat_log, tool_result, session_decision

    # Scoring
    importance = Column(Float, nullable=False, default=0.5)
    decay_score = Column(Float, nullable=False, default=1.0)
    access_count = Column(Integer, nullable=False, default=0)

    # Metadata
    metadata_ = Column("metadata", JSONB, nullable=False, default=dict)

    # Promotion tracking
    promoted_to_l3 = Column(Boolean, nullable=False, default=False)
    promoted_at = Column(DateTime(timezone=True), nullable=True)
    archived_at = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_accessed_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Indexes defined in migration; declared here for documentation only.
    __table_args__ = (
        Index("ix_mem_st_ws_created", "workspace_id", created_at.desc()),
        Index(
            "ix_mem_st_ws_decay",
            "workspace_id",
            "decay_score",
            postgresql_where=(archived_at.is_(None)),
        ),
        Index(
            "ix_mem_st_ws_promote",
            "workspace_id",
            "promoted_to_l3",
            postgresql_where=(
                (promoted_to_l3 == False) & (archived_at.is_(None))  # noqa: E712
            ),
        ),
        Index("ix_mem_st_ws_agent", "workspace_id", "agent_id", created_at.desc()),
        Index("ix_mem_st_ws_type", "workspace_id", "content_type", created_at.desc()),
    )

    def __repr__(self) -> str:
        return (
            f"<MemoryShortTerm(id={self.id}, workspace_id={self.workspace_id}, "
            f"type={self.content_type}, importance={self.importance}, "
            f"decay={self.decay_score})>"
        )
