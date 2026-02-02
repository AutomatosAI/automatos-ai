"""Update marketplace_installs table for single-table architecture

Revision ID: 20260131_marketplace_installs
Revises: 20260131_marketplace_agents
Create Date: 2026-01-31

This migration updates the marketplace_installs table to reference
agents instead of marketplace_items, tracking both the marketplace
agent and the cloned workspace agent.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20260131_marketplace_installs'
down_revision = '20260131_marketplace_agents'
branch_labels = None
depends_on = None


def upgrade():
    """Update marketplace_installs to reference agents table."""

    # Add new columns
    op.add_column('marketplace_installs', sa.Column('marketplace_agent_id', sa.Integer(), nullable=True))
    op.add_column('marketplace_installs', sa.Column('cloned_agent_id', sa.Integer(), nullable=True))

    # Add foreign key constraints
    op.create_foreign_key(
        'fk_marketplace_installs_marketplace_agent_id',
        'marketplace_installs', 'agents',
        ['marketplace_agent_id'], ['id'],
        ondelete='CASCADE'
    )

    op.create_foreign_key(
        'fk_marketplace_installs_cloned_agent_id',
        'marketplace_installs', 'agents',
        ['cloned_agent_id'], ['id'],
        ondelete='SET NULL'
    )

    # Create indexes
    op.create_index(
        'idx_marketplace_installs_marketplace_agent',
        'marketplace_installs',
        ['marketplace_agent_id']
    )

    op.create_index(
        'idx_marketplace_installs_cloned_agent',
        'marketplace_installs',
        ['cloned_agent_id']
    )

    # Note: item_id column remains for backward compatibility
    # Can be removed in a future migration after data migration


def downgrade():
    """Revert marketplace_installs changes."""

    # Drop indexes
    op.drop_index('idx_marketplace_installs_cloned_agent', table_name='marketplace_installs')
    op.drop_index('idx_marketplace_installs_marketplace_agent', table_name='marketplace_installs')

    # Drop foreign keys
    op.drop_constraint('fk_marketplace_installs_cloned_agent_id', 'marketplace_installs', type_='foreignkey')
    op.drop_constraint('fk_marketplace_installs_marketplace_agent_id', 'marketplace_installs', type_='foreignkey')

    # Drop columns
    op.drop_column('marketplace_installs', 'cloned_agent_id')
    op.drop_column('marketplace_installs', 'marketplace_agent_id')
