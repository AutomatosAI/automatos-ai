"""
WorkspaceShare ORM model (US-003).

Tracks per-user sharing permissions on a workspace.
"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database.base import Base


class WorkspaceShare(Base):
    __tablename__ = "workspace_shares"
    __table_args__ = {"extend_existing": True}

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    permission = Column(String(50), nullable=False, default="viewer")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    workspace = relationship("Workspace", back_populates="shares")

    def __repr__(self) -> str:
        return (
            f"<WorkspaceShare id={self.id} workspace={self.workspace_id} "
            f"user={self.user_id} permission={self.permission!r}>"
        )
