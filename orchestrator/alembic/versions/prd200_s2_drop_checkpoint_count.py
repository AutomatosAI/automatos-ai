"""PRD-200 S2 (P2-19) — drop the decorative checkpoint_count column.

The session-checkpoint apparatus (``services/checkpoint_service.py``, the
``SessionCheckpoint`` S3-blob dataclass, ``GET /{id}/checkpoints`` and this
counter) shipped fully but did NO work at runtime: ``write_checkpoint`` had
zero callers, so ``checkpoint_count`` was never incremented off its ``0``
default and the endpoint always returned ``[]``. The apparatus advertised
crash-recovery it never performed (a support-ticket factory), and its
per-verified-task S3-snapshot shape only ever resumes from the last COMPLETED
task — which the in-DB stall-recovery already does for free. All of it is
deleted in this PR (honest-OFF over silent placebo; true in-flight resume is a
distinct executor-touching build — PRD-200 Q1). This migration removes the
orphaned column.

No data migration: the column was always ``0`` (nothing ever wrote it), and the
checkpoints themselves were S3 blobs that were never written.

HUMAN-GATED by convention (Gerard applies prior DROPs) — NOTE the deploy
entrypoint runs ``alembic upgrade heads`` on boot, so this migration
self-applies on the first post-merge deploy.

Revision ID: prd200_s2_drop_checkpoint_count
Revises: prd196_audit_logs_ws_created_idx
"""

import sqlalchemy as sa

from alembic import op

revision = "prd200_s2_drop_checkpoint_count"
down_revision = "prd196_audit_logs_ws_created_idx"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # IF EXISTS: some environments materialised orchestration_runs via the
    # legacy Base.metadata.create_all path at different times, so the column may
    # already be absent. The drop must be idempotent either way.
    op.execute(
        'ALTER TABLE orchestration_runs DROP COLUMN IF EXISTS checkpoint_count'
    )


def downgrade() -> None:
    op.add_column(
        "orchestration_runs",
        sa.Column(
            "checkpoint_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
    )
