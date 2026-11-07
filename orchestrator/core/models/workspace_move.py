"""PRD-273 — ``workspace_move_jobs``: one row per export (and, in wave 2, import) of a workspace.

An export is a background job (``services/workspace_move/jobs.py``): ``queued``, then
``running``, then ``ready`` (its file is in object storage under ``storage_key`` until
``expires_at``) or ``failed`` (``error`` says why, with nothing internal in it). The download
is single-use: it moves the job to ``downloaded`` and the file is deleted; a ready file
nobody downloads is deleted at ``expires_at`` and the job becomes ``expired``.

One job of a kind runs per workspace at a time: a partial unique index over the
``queued``/``running`` rows refuses a second one, so two clicks cannot start two exports.
The passphrase is never stored; ``encrypted`` only says the file has one.

No existing table fits: ``cloud_sync_jobs`` is a cloud-drive connection's sync log. The
Alembic revision ``prd273_w1_workspace_move_jobs`` declares the same shape, and
``tests/test_prd273_w1_migration.py`` holds the two together.
"""
from __future__ import annotations

from sqlalchemy import BigInteger, Boolean, CheckConstraint, Column, DateTime, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base

__all__ = [
    "ACTIVE_STATUSES", "JOB_KINDS", "JOB_STATUSES", "WorkspaceMoveJob", "EXPORT", "IMPORT",
    "QUEUED", "RUNNING", "READY", "DOWNLOADED", "EXPIRED", "FAILED",
]

EXPORT, IMPORT = "export", "import"
JOB_KINDS = (EXPORT, IMPORT)
QUEUED, RUNNING, READY, DOWNLOADED, EXPIRED, FAILED = "queued", "running", "ready", "downloaded", "expired", "failed"
JOB_STATUSES = (QUEUED, RUNNING, READY, DOWNLOADED, EXPIRED, FAILED)
ACTIVE_STATUSES = (QUEUED, RUNNING)
# Exactly the revision's DDL text.
KIND_CHECK = "kind IN ('export', 'import')"
STATUS_CHECK = "status IN ('queued', 'running', 'ready', 'downloaded', 'expired', 'failed')"
ACTIVE_WHERE = "status IN ('queued', 'running')"


class WorkspaceMoveJob(Base):
    """One export or import of a workspace: its status, file and what it held."""

    __tablename__ = "workspace_move_jobs"
    __table_args__ = (
        CheckConstraint(KIND_CHECK, name="ck_workspace_move_jobs_kind"),
        CheckConstraint(STATUS_CHECK, name="ck_workspace_move_jobs_status"),
        Index("uq_workspace_move_jobs_active", "workspace_id", "kind", unique=True,
              postgresql_where=text(ACTIVE_WHERE)),
        Index("ix_workspace_move_jobs_workspace", "workspace_id", "created_at"),
        {"extend_existing": True},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True)
    workspace_id = Column(PGUUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    kind = Column(String(16), nullable=False)  # one of JOB_KINDS
    status = Column(String(16), nullable=False)  # one of JOB_STATUSES
    categories = Column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    encrypted = Column(Boolean, nullable=False, default=False, server_default=text("false"))
    requested_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))  # users.id of the owner or admin
    storage_key = Column(Text)  # the file in object storage; NULL once it is deleted
    file_name = Column(String(255))
    size_bytes = Column(BigInteger)
    summary = Column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    error = Column(Text)  # a message for the owner; the detail is in the server log
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    downloaded_at = Column(DateTime(timezone=True))
