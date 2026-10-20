"""HARNESS state in the database: the task ledger (F156) and the run table (PRD-263).

``HarnessTaskLedger``: one row per HARNESS change proposal and its outcome. F156
kept which done [HARNESS] board tasks self-management applied, and which waited
for an owner's or admin's /approve, here instead of in a file on the workspace
volume that a workspace tool could write. PRD-263 deleted both paths (US-016,
US-047) and the code that wrote it (US-048). US-051 makes it the governed apply
rail's ledger: each row carries its prescription (``rx_id``, unique per
workspace), the run and approval grant behind it, the change's kind and target
agent, the value before and the value proposed, and, once applied, when and over
what value, with the frozen reference the post-change check reads. ``state`` is
one of the ``LEDGER_*`` values below. ``board_task_id`` is the [HARNESS] card a
row is keyed by (still unique per workspace), NULL for a row with none. DIAGNOSE
reads its applied rows (services.harness_diagnose).

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

# harness_task_ledger.state: a proposal waiting for the owner's click, one being
# applied, and its ends. ck_harness_task_ledger_state allows exactly these; the
# F156 ``held`` state became ``expired`` (US-051).
LEDGER_PENDING = "pending"
LEDGER_APPLYING = "applying"
LEDGER_APPLIED = "applied"
LEDGER_FAILED = "failed"
LEDGER_DENIED = "denied"
LEDGER_EXPIRED = "expired"
LEDGER_DRIFTED = "drifted"
LEDGER_WITHDRAWN = "withdrawn"
LEDGER_RAISE_FAILED = "raise_failed"
LEDGER_ROLLED_BACK = "rolled_back"

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
    board_task_id = Column(Integer, nullable=True)
    state = Column(String(20), nullable=False)
    entry = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    # PRD-263 US-051: the governed apply rail. The Alembic revision
    # prd263_harness_ledger_grants adds the same columns; F156 rows leave them NULL.
    rx_id = Column(String(64), nullable=True)
    run_id = Column(Integer, nullable=True)
    grant_id = Column(Integer, nullable=True)
    kind = Column(String(40), nullable=True)
    target_agent_id = Column(Integer, nullable=True)
    prior_value = Column(JSONB, nullable=True)
    proposed_value = Column(JSONB, nullable=True)
    applied_at = Column(DateTime(timezone=True), nullable=True)
    value_at_apply = Column(JSONB, nullable=True)
    frozen_ref_start = Column(DateTime(timezone=True), nullable=True)
    frozen_ref_end = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("workspace_id", "board_task_id", name="uq_harness_task_ledger_task"),
        Index("uq_harness_task_ledger_rx", "workspace_id", "rx_id", unique=True),
        CheckConstraint(
            "state IN ('pending', 'applying', 'applied', 'failed', 'denied', 'expired', "
            "'drifted', 'withdrawn', 'raise_failed', 'rolled_back')",
            name="ck_harness_task_ledger_state",
        ),
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
