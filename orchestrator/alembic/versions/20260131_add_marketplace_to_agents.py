"""Add marketplace fields to agents table for single-table architecture

Revision ID: 20260131_marketplace_agents
Revises: 20260130_rename_templates_to_recipes
Create Date: 2026-01-31

This migration adds marketplace-related fields to the agents table to enable
a single-table architecture where both workspace and marketplace agents are
stored in the same table, distinguished by owner_type.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260131_marketplace_agents'
down_revision = 'rename_templates_to_recipes'
branch_labels = None
depends_on = None


def upgrade():
    """Add marketplace fields to agents table."""

    # Add ownership columns
    op.add_column('agents', sa.Column('owner_type', sa.String(20), nullable=False, server_default='workspace'))
    op.add_column('agents', sa.Column('owner_id', sa.String(255), nullable=True))

    # Add cloning/tracking columns
    op.add_column('agents', sa.Column('cloned_from_id', sa.Integer(), nullable=True))
    op.add_column('agents', sa.Column('original_creator_id', sa.Integer(), nullable=True))

    # Add marketplace metadata columns
    op.add_column('agents', sa.Column('install_count', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('agents', sa.Column('is_featured', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('agents', sa.Column('is_approved', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('agents', sa.Column('marketplace_category', sa.String(100), nullable=True))
    op.add_column('agents', sa.Column('marketplace_icon', sa.String(500), nullable=True))
    op.add_column('agents', sa.Column('version', sa.String(50), nullable=False, server_default='1.0.0'))

    # Add foreign key constraints
    op.create_foreign_key(
        'fk_agents_cloned_from_id',
        'agents', 'agents',
        ['cloned_from_id'], ['id'],
        ondelete='SET NULL'
    )

    op.create_foreign_key(
        'fk_agents_original_creator_id',
        'agents', 'users',
        ['original_creator_id'], ['id'],
        ondelete='SET NULL'
    )

    # Add check constraint for owner_type
    op.create_check_constraint(
        'ck_agents_owner_type',
        'agents',
        "owner_type IN ('workspace', 'marketplace')"
    )

    # Create indexes for efficient querying
    op.create_index(
        'idx_agents_owner_type',
        'agents',
        ['owner_type', 'is_approved']
    )

    op.create_index(
        'idx_agents_marketplace_featured',
        'agents',
        ['is_featured', 'install_count'],
        postgresql_where=sa.text("owner_type = 'marketplace'")
    )

    op.create_index(
        'idx_agents_marketplace_category',
        'agents',
        ['marketplace_category'],
        postgresql_where=sa.text("owner_type = 'marketplace'")
    )

    op.create_index(
        'idx_agents_cloned_from',
        'agents',
        ['cloned_from_id']
    )

    # Set owner_id to workspace_id for existing agents
    op.execute("""
        UPDATE agents
        SET owner_id = workspace_id::varchar
        WHERE owner_type = 'workspace'
    """)


def downgrade():
    """Remove marketplace fields from agents table."""

    # Drop indexes
    op.drop_index('idx_agents_cloned_from', table_name='agents')
    op.drop_index('idx_agents_marketplace_category', table_name='agents')
    op.drop_index('idx_agents_marketplace_featured', table_name='agents')
    op.drop_index('idx_agents_owner_type', table_name='agents')

    # Drop constraints
    op.drop_constraint('ck_agents_owner_type', 'agents', type_='check')
    op.drop_constraint('fk_agents_original_creator_id', 'agents', type_='foreignkey')
    op.drop_constraint('fk_agents_cloned_from_id', 'agents', type_='foreignkey')

    # Drop columns
    op.drop_column('agents', 'version')
    op.drop_column('agents', 'marketplace_icon')
    op.drop_column('agents', 'marketplace_category')
    op.drop_column('agents', 'is_approved')
    op.drop_column('agents', 'is_featured')
    op.drop_column('agents', 'install_count')
    op.drop_column('agents', 'original_creator_id')
    op.drop_column('agents', 'cloned_from_id')
    op.drop_column('agents', 'owner_id')
    op.drop_column('agents', 'owner_type')
