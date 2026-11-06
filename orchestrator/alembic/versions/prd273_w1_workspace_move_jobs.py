"""PRD-273 W1: a workspace's export (and, in wave 2, import) jobs (6 Nov 2026).

The ONE migration of wave 1 (export).

* ``workspace_move_jobs``: one row per export or import of a workspace: its kind, status,
  categories, whether the file has a passphrase (never the passphrase), who asked, the file
  in object storage while it waits for its single download, its size, the summary of what it
  holds, a message when it failed, and its times. Two CHECKs keep ``kind`` and ``status`` to
  their vocabularies; a partial unique index over the ``queued`` and ``running`` rows keeps
  one job of a kind per workspace at a time; one index serves a workspace's newest jobs.
* The workspace's foreign key is ON DELETE CASCADE (a purged workspace takes its jobs);
  ``requested_by`` is ON DELETE SET NULL (a deleted person leaves the job's record).

``core/models/workspace_move.py`` (``WorkspaceMoveJob``) declares the same shape;
``tests/test_prd273_w1_migration.py`` holds the two together.

Create_all-first safe: the table is ``CREATE TABLE IF NOT EXISTS`` and each index ``CREATE
... INDEX IF NOT EXISTS``, so a database ``create_all`` built is left as it is, and running it
twice changes nothing. It alters no table and changes no row. The downgrade drops the table.

Revision ID: prd273_w1_workspace_move_jobs
Revises: prd270_w2_cloud_bundles
Create Date: 2026-11-06
"""

from alembic import op


revision = "prd273_w1_workspace_move_jobs"
down_revision = "prd270_w2_cloud_bundles"
branch_labels = None
depends_on = None

_TABLES = (
    ("workspace_move_jobs", """
CREATE TABLE IF NOT EXISTS workspace_move_jobs (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    kind VARCHAR(16) NOT NULL,
    status VARCHAR(16) NOT NULL,
    categories JSONB NOT NULL DEFAULT '[]'::jsonb,
    encrypted BOOLEAN NOT NULL DEFAULT false,
    requested_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    storage_key TEXT,
    file_name VARCHAR(255),
    size_bytes BIGINT,
    summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    downloaded_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT ck_workspace_move_jobs_kind CHECK (kind IN ('export', 'import')),
    CONSTRAINT ck_workspace_move_jobs_status CHECK (status IN ('queued', 'running', 'ready', 'downloaded', 'expired', 'failed'))
);
"""),
)

# As core/models/workspace_move.py declares them.
_INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_workspace_move_jobs_active ON workspace_move_jobs "
    "(workspace_id, kind) WHERE status IN ('queued', 'running');",
    "CREATE INDEX IF NOT EXISTS ix_workspace_move_jobs_workspace ON workspace_move_jobs (workspace_id, created_at);",
)

_DOWNGRADE = (
    "DROP TABLE IF EXISTS workspace_move_jobs;",
)


def upgrade() -> None:
    for _table, statement in _TABLES:
        op.execute(statement)
    for statement in _INDEXES:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
