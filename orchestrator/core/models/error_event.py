"""
ErrorEvent ORM model (PRD-142 Wave 0 US-001).

Append-only queryable sink for ``record_error`` records. The
``automatos.errors`` logger emit is the source of truth for *log* sinks;
this table is the source of truth for *dashboard* rollups — specifically
the "error rate by subsystem" tile defined in PRD-142 Wave 0.

Pattern follows PRD-008-A ``WidgetEventLog`` / PRD-139 ``ToolExecutionLog``:
single table, JSONB payload, fire-and-forget writer that never propagates
failures (see ``core/utils/exception_telemetry.py``).
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base


class ErrorEvent(Base):
    __tablename__ = "error_events"
    __table_args__ = (
        Index("idx_error_events_subsystem_created", "subsystem", "created_at"),
        Index("idx_error_events_workspace_created", "workspace_id", "created_at"),
        {"extend_existing": True},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Coarse origin of the failure (e.g. "memory", "tools", "harness").
    # Dashboard groups by this column.
    subsystem = Column(String(64), nullable=False)

    # The specific operation that failed (e.g. "add_memory").
    operation = Column(String(128), nullable=False)

    # Python exception class name. Nullable because record_error tolerates
    # arbitrary objects in `error` and we never want a sink write to be the
    # thing that fails persistence.
    error_type = Column(String(128), nullable=True)

    # Truncated exception message — VARCHAR(500) to bound storage and
    # protect downstream JSON parsers from pathological strings.
    error_message = Column(String(500), nullable=True)

    # Owning workspace, if known. NULL for system-level errors raised
    # before workspace context is established (startup, cron, etc.).
    workspace_id = Column(PGUUID(as_uuid=True), nullable=True)

    # Owning agent, if known.
    agent_id = Column(Integer, nullable=True)

    # Platform action involved, if any.
    action_name = Column(String(128), nullable=True)

    # Caller-supplied extras (correlation IDs, tool/plumbing context, ...).
    # Keep small (<2KB typical). Schema is per-subsystem and intentionally
    # loose so this table can outlive any one PRD's metadata choices.
    event_data = Column(JSONB, nullable=False, server_default="{}")

    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<ErrorEvent id={self.id} subsystem={self.subsystem!r} "
            f"operation={self.operation!r} type={self.error_type!r}>"
        )
