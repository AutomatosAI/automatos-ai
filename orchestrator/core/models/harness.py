"""HARNESS state in the database: the task ledger (F156) and the run table (PRD-263).

``HarnessTaskLedger``: which done [HARNESS] board tasks self-management has
applied (with the value each change replaced, for US-022 rollback), and which
wait for an owner's or admin's /approve. It is the idempotency key for the
weekly tick and the /approve command. It lives here, written only by
services.harness_service, instead of in a file on the workspace volume that a
workspace tool could write.

``HarnessRun`` (PRD-263 US-009): one HARNESS run of one workspace, for the ISO
week it judges. Every worker reads the same row, so run state no longer lives in
last_run/trace JSON files on the volume. Written only through services.harness_runs.
"""
from sqlalchemy import (
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
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from core.database.base import Base

LEDGER_APPLIED = "applied"
LEDGER_HELD = "held"

# harness_runs.status: a run in flight, the two ends a run that passes the
# dormancy check reaches, and the dormant ends of one that does not.
# ck_harness_runs_status allows exactly these.
RUN_RUNNING = "running"
RUN_COMPLETED = "completed"
RUN_FAILED = "failed"
RUN_DORMANT_AGENTS = "dormant_insufficient_agents"
RUN_DORMANT_DATA = "dormant_insufficient_data"


class HarnessTaskLedger(Base):
    __tablename__ = "harness_task_ledger"

    id = Column(Integer, primary_key=True)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    board_task_id = Column(Integer, nullable=False)
    state = Column(String(20), nullable=False)
    entry = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("workspace_id", "board_task_id", name="uq_harness_task_ledger_task"),
        CheckConstraint("state IN ('applied', 'held')", name="ck_harness_task_ledger_state"),
        {"extend_existing": True},
    )


class HarnessRun(Base):
    """One HARNESS run: its judged week, outcome, counts and audit report.

    ``cur_window`` is the last complete ISO week when the run started (``2026-W42``).
    ``iteration`` is the number of completed runs of the workspace up to and
    including this one. ``status`` is one of the ``RUN_*`` values above (a CHECK
    constraint). ``report_status`` is ``ok`` or ``failed: <reason>``. The Alembic
    revision ``prd263_harness_runs`` declares the same names.
    """

    __tablename__ = "harness_runs"

    id = Column(Integer, primary_key=True)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    cur_window = Column(String(10), nullable=False)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(40), nullable=False)
    iteration = Column(Integer, nullable=False, server_default="0")
    coverage = Column(Float, nullable=True)
    counts = Column(JSONB, nullable=True)
    verdicts = Column(JSONB, nullable=True)
    prescriptions = Column(JSONB, nullable=True)
    report_md = Column(Text, nullable=True)
    report_status = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_harness_runs_ws_window", "workspace_id", "cur_window"),
        CheckConstraint(
            "status IN ('running', 'completed', 'failed', "
            "'dormant_insufficient_agents', 'dormant_insufficient_data')",
            name="ck_harness_runs_status",
        ),
        {"extend_existing": True},
    )
