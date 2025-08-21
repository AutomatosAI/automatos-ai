"""add playbooks table

Revision ID: add_playbooks
Revises: add_code_graph
Create Date: 2025-08-12 14:20:03

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = 'add_playbooks'
down_revision = 'add_code_graph'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'playbooks',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('tenant_id', sa.String(length=128), nullable=True),
        sa.Column('pattern', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('support', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
    )
    op.create_index('ix_playbooks_name', 'playbooks', ['name'])
    op.create_index('ix_playbooks_tenant', 'playbooks', ['tenant_id'])


def downgrade() -> None:
    op.drop_index('ix_playbooks_tenant', table_name='playbooks')
    op.drop_index('ix_playbooks_name', table_name='playbooks')
    op.drop_table('playbooks')


