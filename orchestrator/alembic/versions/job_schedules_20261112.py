"""Each schedule's cursor (12 Nov 2026).

PRD-276, "Durable background work", Wave 2b (§6.4). One new table, ``jobs.job_schedules``: a
schedule's ``kind`` and ``schedule_id`` (the key), its tenant (``workspace_id``, NULL for a platform
schedule; ON DELETE CASCADE, a backstop to the purge, §6.8) and ``last_slot``, the last slot it
queued. Each due slot is queued in the transaction that moves ``last_slot`` past it.

``core/models/jobs.py`` (``JobSchedule``) declares the same shape. Create_all-first safe:
``CREATE SCHEMA IF NOT EXISTS``, ``CREATE TABLE IF NOT EXISTS`` and ``CREATE INDEX IF NOT EXISTS``;
it changes no row. The downgrade drops the table.

Revision ID: job_schedules_20261112
Revises: jobs_schema_20261112
Create Date: 2026-11-12
"""
from alembic import op

revision = "job_schedules_20261112"
down_revision = "jobs_schema_20261112"
branch_labels = None
depends_on = None

_TABLE = """
CREATE TABLE IF NOT EXISTS jobs.job_schedules (
    kind VARCHAR(100) NOT NULL,
    schedule_id VARCHAR(255) NOT NULL,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    last_slot TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    PRIMARY KEY (kind, schedule_id)
);
"""


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS jobs")
    op.execute(_TABLE)
    op.execute("CREATE INDEX IF NOT EXISTS ix_job_schedules_workspace ON jobs.job_schedules (workspace_id);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS jobs.job_schedules;")
