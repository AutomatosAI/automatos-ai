"""add context_policies table

Revision ID: add_context_policies
Revises: 6203026dbac0
Create Date: 2025-08-12 14:13:06

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = 'add_context_policies'
down_revision = '6203026dbac0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'context_policies',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('policy_id', sa.String(length=128), nullable=False),
        sa.Column('domain', sa.String(length=128), nullable=True),
        sa.Column('agent_id', sa.String(length=128), nullable=True),
        sa.Column('tenant_id', sa.String(length=128), nullable=True),
        sa.Column('slots', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('max_total_chars', sa.Integer(), nullable=False, server_default='12000'),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
    )
    op.create_index('ix_context_policies_policy_id', 'context_policies', ['policy_id'])
    op.create_index('ix_context_policies_domain', 'context_policies', ['domain'])
    op.create_index('ix_context_policies_agent_id', 'context_policies', ['agent_id'])
    op.create_index('ix_context_policies_tenant_id', 'context_policies', ['tenant_id'])


def downgrade() -> None:
    op.drop_index('ix_context_policies_tenant_id', table_name='context_policies')
    op.drop_index('ix_context_policies_agent_id', table_name='context_policies')
    op.drop_index('ix_context_policies_domain', table_name='context_policies')
    op.drop_index('ix_context_policies_policy_id', table_name='context_policies')
    op.drop_table('context_policies')


