"""PRD-234 S1a — CLI hosts: the paired local processes that run session-mode tickets.

A CLI host is the user's own machine running ``make cli-host``. It pairs ONCE
with a one-time code the operator reads from Settings → Session mode, receives
a host token (only its SHA-256 is stored), and from then on claims the tickets
of ``runtime: cli`` agents, streams their events and posts their results.

Why a table (CLAUDE.md "no new tables" rule): the pairing state and the token
digest must survive restarts on both sides and be revocable; nothing existing
holds a per-machine credential. One small table, one purpose.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base


class CliHostStatus(str, Enum):
    """``PENDING`` — a pairing code was issued, no host has claimed it yet.
    ``PAIRED`` — the host holds a token and may claim work.
    ``REVOKED`` — the token is dead; the host must pair again."""

    PENDING = "pending"
    PAIRED = "paired"
    REVOKED = "revoked"


class CliHost(Base):
    __tablename__ = "cli_hosts"
    __table_args__ = (
        Index("ix_cli_hosts_workspace_status", "workspace_id", "status"),
        {"extend_existing": True},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(120), nullable=False, default="cli-host")
    status = Column(
        String(16), nullable=False,
        default=CliHostStatus.PENDING.value, server_default="pending",
    )
    # SHA-256 of the one-time pairing code; cleared the moment the host pairs.
    pairing_code_hash = Column(String(64), nullable=True)
    pairing_expires_at = Column(DateTime(timezone=True), nullable=True)
    # SHA-256 of the host token. The token itself is returned exactly once.
    token_hash = Column(String(64), nullable=True, unique=True)
    # What the host announced: installed CLIs, versions, login state, platform.
    capabilities = Column(JSONB, nullable=True)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    paired_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def is_online(self, *, window_seconds: int = 90) -> bool:
        if self.status != CliHostStatus.PAIRED.value or self.last_seen_at is None:
            return False
        seen = self.last_seen_at
        if seen.tzinfo is None:
            seen = seen.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - seen).total_seconds() <= window_seconds

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "name": self.name,
            "status": self.status,
            "online": self.is_online(),
            "capabilities": self.capabilities or {},
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "paired_at": self.paired_at.isoformat() if self.paired_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
