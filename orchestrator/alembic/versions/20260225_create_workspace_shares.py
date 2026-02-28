"""Create workspace_shares table (US-002)

Revision ID: 20260225_ws_shares
Revises: 20260225_ws_persistence
Create Date: 2026-02-25

Creates workspace_shares table:
- id            UUID PK
- workspace_id  UUID FK workspaces ON DELETE CASCADE
- user_id       UUID FK users
- permission    VARCHAR(20) DEFAULT 'view' CHECK IN ('view','edit','admin')
- created_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
- UNIQUE(workspace_id, user_id)
- INDEX idx_workspace_shares_user ON user_id
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260225_ws_shares'
down_revision = '20260225_ws_persistence'
branch_labels = None
depends_on = None

TABLE = 'workspace_shares'


def upgrade() -> None:
    op.create_table(
        TABLE,
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id'), nullable=False),
        sa.Column('permission', sa.String(20), server_default='view', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("permission IN ('view', 'edit', 'admin')", name='ck_workspace_shares_permission'),
        sa.UniqueConstraint('workspace_id', 'user_id', name='uq_workspace_shares_workspace_user'),
    )

    op.create_index('idx_workspace_shares_user', TABLE, ['user_id'])


def downgrade() -> None:
    op.drop_table(TABLE)
