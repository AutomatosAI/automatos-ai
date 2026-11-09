"""Signed completion callbacks: a workspace's signing secret, and each run's callback (9 Nov 2026).

PRD-274, "Automatos for products", Wave 1 (§8-8). Two new tables:

* ``workspace_signing_secrets``: one per workspace (its primary key), the secret its callbacks are
  signed with, encrypted. ON DELETE CASCADE with the workspace.
* ``run_callbacks``: the URL a ticket's ending is POSTed to, one per ticket (unique), and how the
  delivery went. ON DELETE CASCADE with the workspace and with the ticket.

``core/models/run_callbacks.py`` declares the same shape; ``tests/test_afp_w1_run_callbacks.py``
holds the two together.

Create_all-first safe: each table is ``CREATE TABLE IF NOT EXISTS`` and the index ``CREATE INDEX IF
NOT EXISTS``, so a database ``create_all`` built is left as it is, and running it twice changes
nothing. It alters no table and changes no row. The downgrade drops both tables.

Revision ID: run_callbacks_20261109
Revises: memory_profiles_20261109
Create Date: 2026-11-09
"""
from alembic import op

revision = "run_callbacks_20261109"
down_revision = "memory_profiles_20261109"
branch_labels = None
depends_on = None

_SECRETS = """
CREATE TABLE IF NOT EXISTS workspace_signing_secrets (
    workspace_id UUID PRIMARY KEY REFERENCES workspaces(id) ON DELETE CASCADE,
    secret_encrypted TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    rotated_at TIMESTAMP WITH TIME ZONE
);
"""

_CALLBACKS = """
CREATE TABLE IF NOT EXISTS run_callbacks (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    task_id INTEGER NOT NULL UNIQUE REFERENCES board_tasks(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    last_status_code INTEGER,
    last_error TEXT,
    delivered_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);
"""

# As core/models/run_callbacks.py declares it (index=True on workspace_id).
_INDEX = "CREATE INDEX IF NOT EXISTS ix_run_callbacks_workspace_id ON run_callbacks (workspace_id);"


def upgrade() -> None:
    op.execute(_SECRETS)
    op.execute(_CALLBACKS)
    op.execute(_INDEX)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS run_callbacks;")
    op.execute("DROP TABLE IF EXISTS workspace_signing_secrets;")
