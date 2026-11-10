"""Signed completion callbacks for runs an API caller starts (PRD-274 Wave 1, §8-8).

* ``WorkspaceSigningSecret``: the one secret a workspace's callbacks are signed with (HMAC-SHA256),
  encrypted at rest with the platform's credential encryption. Made on first use; a workspace's
  managers can read and rotate it.
* ``RunCallback``: the URL a run's ending is POSTed to, one per ticket, and how its delivery went.

``alembic/versions/run_callbacks_20261109.py`` creates both; ``services/run_callbacks.py`` signs
and delivers.
"""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base

PENDING = "pending"
DELIVERED = "delivered"
FAILED = "failed"


class WorkspaceSigningSecret(Base):
    """A workspace's callback signing secret, encrypted."""

    __tablename__ = "workspace_signing_secrets"

    workspace_id = Column(PGUUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True)
    secret_encrypted = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    rotated_at = Column(DateTime(timezone=True), nullable=True)


class RunCallback(Base):
    """Where a ticket's ending is POSTed, and the outcome of the last delivery."""

    __tablename__ = "run_callbacks"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(PGUUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False,
                          index=True)
    task_id = Column(Integer, ForeignKey("board_tasks.id", ondelete="CASCADE"), nullable=False, unique=True)
    url = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default=PENDING, server_default=text(f"'{PENDING}'"))
    attempts = Column(Integer, nullable=False, default=0, server_default=text("0"))
    last_status_code = Column(Integer, nullable=True)
    last_error = Column(Text, nullable=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
