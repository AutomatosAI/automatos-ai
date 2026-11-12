"""The durable job queue's table (11 Nov 2026).

PRD-276, "Durable background work", Wave 1a (§6.2). One new table, ``jobs``:

* a job's ``kind``, tenant (``workspace_id``, NULL for a platform job; ON DELETE CASCADE, a backstop
  to the purge's barrier, §6.8), ``payload`` (ids, never blobs), ``status`` (queued, running, done,
  dead), ``run_at``, ``attempts`` and ``max_attempts``, the lease (``lease_until``, ``claimed_by``,
  ``claim_id``), ``idempotency_key``, ``last_error`` and its times;
* ``uq_jobs_kind_workspace_key``: unique (kind, workspace_id, idempotency_key) NULLS NOT DISTINCT
  where a key is set (Postgres 15+), so a platform job's NULL workspace counts as one value;
* ``ix_jobs_claim`` (status, run_at) for the worker's claim, ``ix_jobs_workspace`` for erasure.

``core/models/jobs.py`` declares the same shape. Create_all-first safe: ``CREATE TABLE IF NOT EXISTS``
and ``CREATE ... INDEX IF NOT EXISTS``; it changes no row. The downgrade drops the table.

Revision ID: jobs_20261111
Revises: end_users_erasing_20261111
Create Date: 2026-11-11
"""
from alembic import op

revision = "jobs_20261111"
down_revision = "end_users_erasing_20261111"
branch_labels = None
depends_on = None

_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY,
    kind VARCHAR(100) NOT NULL,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(16) NOT NULL DEFAULT 'queued',
    run_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL,
    lease_until TIMESTAMP WITH TIME ZONE,
    claimed_by VARCHAR(255),
    claim_id UUID,
    idempotency_key VARCHAR(255),
    last_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    finished_at TIMESTAMP WITH TIME ZONE
);
"""

_INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_kind_workspace_key ON jobs (kind, workspace_id, idempotency_key) "
    "NULLS NOT DISTINCT WHERE idempotency_key IS NOT NULL;",
    "CREATE INDEX IF NOT EXISTS ix_jobs_claim ON jobs (status, run_at);",
    "CREATE INDEX IF NOT EXISTS ix_jobs_workspace ON jobs (workspace_id);",
)


def upgrade() -> None:
    op.execute(_TABLE)
    for statement in _INDEXES:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS jobs;")
