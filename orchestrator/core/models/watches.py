"""
Watch Models — PRD-204 Auto Watcher
====================================

SQLAlchemy models for the watch registry:
- Watch: one row supervising one launched unit of work to a verdict
- WatchEvent: append-only observations per watch (idempotent via event_key)

Style mirrors ``core/models/orchestration.py`` (PRD-82A): UUID PKs with
DB-generated defaults, workspace FK CASCADE, ``version_id`` optimistic
locking on the mutable row, CHECK constraint built from the enum, append-only
event table without version_id.

Source: PRD-204 Section 4 S1.
"""

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from core.database.base import Base
from core.models.watch_enums import (
    WatchPolicy,
    WatchStatus,
    terminal_watch_statuses_sql,
)

# PRD-204 defaults (Section 8 Q2/Q1 build defaults)
DEFAULT_QUALITY_THRESHOLD = 0.8
DEFAULT_CHECK_INTERVAL_SECONDS = 300
DEFAULT_ACTION_BUDGET = 2


class Watch(Base):
    """A workspace-scoped supervisor for one launched unit of work.

    The watch follows the work (Section 8 Q9): corrective reruns/replans
    append to ``lineage`` and repoint ``target_type``/``target_id`` — one
    watch, one verdict. At most ONE non-terminal watch may exist per
    (workspace_id, target_type, target_id), enforced by a partial unique
    index.
    """

    __tablename__ = "watches"

    # Primary key — UUID with DB-generated default (house pattern)
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )

    # Tenant isolation
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Ownership: Clerk user id string (same convention as
    # orchestration_runs.created_by); owner_agent_id for Auto-created watches.
    created_by = Column(String(255), nullable=True)
    owner_agent_id = Column(
        Integer,
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
    )

    # What is being watched
    watch_type = Column(String(32), nullable=False)   # WatchType
    target_type = Column(String(32), nullable=False)  # WatchTargetType (current target)
    target_id = Column(String(255), nullable=False)   # current target row id, as string
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)

    # Lifecycle
    status = Column(
        String(32),
        nullable=False,
        default=WatchStatus.WATCHING.value,
        server_default=WatchStatus.WATCHING.value,
    )

    # Intent snapshot — what "good" means for this launch
    success_criteria = Column(Text, nullable=True)
    failure_criteria = Column(Text, nullable=True)
    quality_threshold = Column(
        Float,
        nullable=False,
        default=DEFAULT_QUALITY_THRESHOLD,
        server_default=text(str(DEFAULT_QUALITY_THRESHOLD)),
    )

    # Heartbeat sweep scheduling
    check_interval_seconds = Column(
        Integer,
        nullable=False,
        default=DEFAULT_CHECK_INTERVAL_SECONDS,
        server_default=text(str(DEFAULT_CHECK_INTERVAL_SECONDS)),
    )
    last_checked_at = Column(DateTime(timezone=True), nullable=True)
    next_check_at = Column(DateTime(timezone=True), nullable=True)
    deadline_at = Column(DateTime(timezone=True), nullable=True)

    # Decision profile + bounded corrective actions
    policy = Column(
        String(32),
        nullable=False,
        default=WatchPolicy.RUN_AND_REPORT.value,
        server_default=WatchPolicy.RUN_AND_REPORT.value,
    )
    allowed_actions = Column(JSONB, nullable=True)  # e.g. ["rerun", "tweak", "replan"]
    action_budget = Column(
        Integer,
        nullable=False,
        default=DEFAULT_ACTION_BUDGET,
        server_default=text(str(DEFAULT_ACTION_BUDGET)),
    )
    actions_taken = Column(Integer, nullable=False, default=0, server_default="0")

    # Verdict (S6 writes score; v1 writes outcome text)
    final_score = Column(Float, nullable=True)
    final_verdict = Column(Text, nullable=True)

    # Ordered target chain — reruns/replans append {target_type, target_id,
    # followed_at, reason}; the columns above always hold the LIVE target.
    lineage = Column(JSONB, nullable=False, server_default=text("'[]'"))

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    closed_at = Column(DateTime(timezone=True), nullable=True)

    # Optimistic locking (house pattern — PRD-82A Section 5 principle 7)
    version_id = Column(Integer, nullable=False, server_default="1")

    __mapper_args__ = {"version_id_col": version_id}

    __table_args__ = (
        CheckConstraint(
            f"status IN ({', '.join(repr(s.value) for s in WatchStatus)})",
            name="ck_watches_status",
        ),
        # One non-terminal watch per target (PRD-204 S1)
        Index(
            "uq_watches_live_target",
            "workspace_id",
            "target_type",
            "target_id",
            unique=True,
            postgresql_where=text(
                f"status NOT IN ({terminal_watch_statuses_sql()})"
            ),
        ),
        # Tick claim: due watches by next_check_at, live statuses only
        Index(
            "ix_watches_due",
            "next_check_at",
            postgresql_where=text("status IN ('watching', 'acting')"),
        ),
        {"extend_existing": True},
    )

    def __repr__(self) -> str:
        return (
            f"<Watch id={self.id} status={self.status} "
            f"target={self.target_type}:{self.target_id} "
            f"title={(self.title[:40] if self.title else None)!r}>"
        )


class WatchEvent(Base):
    """Append-only observation log for a watch.

    ``UNIQUE(watch_id, event_key)`` makes ingest idempotent: producers and
    the sweep tick can both report the same observation and exactly one row
    lands. This table is never updated — no version_id column (same shape as
    OrchestrationEvent).
    """

    __tablename__ = "watch_events"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )

    watch_id = Column(
        UUID(as_uuid=True),
        ForeignKey("watches.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    event_type = Column(String(50), nullable=False)  # WatchEventType
    summary = Column(Text, nullable=True)
    snapshot = Column(JSONB, nullable=True)  # cost snapshot / output pointer / context
    score = Column(Float, nullable=True)
    action_taken = Column(String(100), nullable=True)
    requires_attention = Column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )

    # Idempotency key — semantic identity of the observation
    event_key = Column(String(255), nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint("watch_id", "event_key", name="uq_watch_events_key"),
        Index("ix_watch_events_watch_created", "watch_id", "created_at"),
        {"extend_existing": True},
    )

    def __repr__(self) -> str:
        return (
            f"<WatchEvent id={self.id} watch={self.watch_id} "
            f"type={self.event_type} key={self.event_key!r}>"
        )
