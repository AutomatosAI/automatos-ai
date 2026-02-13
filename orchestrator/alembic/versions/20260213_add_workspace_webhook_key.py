"""Add webhook_key column to workspaces for general workspace webhooks

Revision ID: 20260213_ws_webhook_key
Revises: 20260205_personas_plugins
Create Date: 2026-02-13

Adds:
  - webhook_key column (String(64), unique, nullable) on workspaces table
  - Auto-populates existing workspaces with a random hex key
"""

from alembic import op
import sqlalchemy as sa
from uuid import uuid4

# revision identifiers
revision = '20260213_ws_webhook_key'
down_revision = '20260205_personas_plugins'
branch_labels = None
depends_on = None


def upgrade():
    # Add the column as nullable first
    op.add_column('workspaces', sa.Column('webhook_key', sa.String(64), nullable=True))

    # Populate existing workspaces with unique keys
    conn = op.get_bind()
    workspaces = conn.execute(sa.text("SELECT id FROM workspaces")).fetchall()
    for (ws_id,) in workspaces:
        key = uuid4().hex
        conn.execute(
            sa.text("UPDATE workspaces SET webhook_key = :key WHERE id = :id"),
            {"key": key, "id": str(ws_id)},
        )

    # Add unique index
    op.create_index('ix_workspaces_webhook_key', 'workspaces', ['webhook_key'], unique=True)


def downgrade():
    op.drop_index('ix_workspaces_webhook_key', table_name='workspaces')
    op.drop_column('workspaces', 'webhook_key')
