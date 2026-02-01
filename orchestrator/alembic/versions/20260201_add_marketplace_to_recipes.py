"""Add marketplace fields to recipes table for single-table architecture

Revision ID: 20260201_marketplace_recipes
Revises: 20260131_make_workspace_id_nullable
Create Date: 2026-02-01

This migration adds marketplace-related fields to the workflow_recipes table to enable
a single-table architecture where both workspace and marketplace recipes are
stored in the same table, distinguished by owner_type.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260201_marketplace_recipes'
down_revision = '20260131_workspace_nullable'
branch_labels = None
depends_on = None


def upgrade():
    """Add marketplace fields to workflow_recipes table."""

    # Add workspace_id if it doesn't exist (it should from the rename, but check)
    # Make it nullable to support marketplace recipes (NULL = marketplace)
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='workflow_recipes' AND column_name='workspace_id'
            ) THEN
                ALTER TABLE workflow_recipes ADD COLUMN workspace_id UUID;
                CREATE INDEX IF NOT EXISTS ix_workflow_recipes_workspace_id ON workflow_recipes(workspace_id);
            ELSE
                -- Make workspace_id nullable if it exists
                ALTER TABLE workflow_recipes ALTER COLUMN workspace_id DROP NOT NULL;
            END IF;
        END $$;
    """)

    # Add ownership columns
    op.add_column('workflow_recipes', sa.Column('owner_type', sa.String(20), nullable=False, server_default='workspace'))
    op.add_column('workflow_recipes', sa.Column('owner_id', sa.String(255), nullable=True))

    # Add cloning/tracking columns
    op.add_column('workflow_recipes', sa.Column('cloned_from_id', sa.Integer(), nullable=True))
    op.add_column('workflow_recipes', sa.Column('original_creator_id', sa.Integer(), nullable=True))
    op.add_column('workflow_recipes', sa.Column('created_by_user_id', sa.Integer(), nullable=True))

    # Add marketplace metadata columns
    op.add_column('workflow_recipes', sa.Column('install_count', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('workflow_recipes', sa.Column('is_approved', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('workflow_recipes', sa.Column('marketplace_category', sa.String(100), nullable=True))
    op.add_column('workflow_recipes', sa.Column('marketplace_icon', sa.String(500), nullable=True))

    # Add foreign key constraints
    op.create_foreign_key(
        'fk_recipes_workspace_id',
        'workflow_recipes', 'workspaces',
        ['workspace_id'], ['id'],
        ondelete='CASCADE'
    )

    op.create_foreign_key(
        'fk_recipes_cloned_from_id',
        'workflow_recipes', 'workflow_recipes',
        ['cloned_from_id'], ['id'],
        ondelete='SET NULL'
    )

    op.create_foreign_key(
        'fk_recipes_original_creator_id',
        'workflow_recipes', 'users',
        ['original_creator_id'], ['id'],
        ondelete='SET NULL'
    )

    op.create_foreign_key(
        'fk_recipes_created_by_user_id',
        'workflow_recipes', 'users',
        ['created_by_user_id'], ['id'],
        ondelete='SET NULL'
    )

    # Add check constraint for owner_type
    op.create_check_constraint(
        'ck_recipes_owner_type',
        'workflow_recipes',
        "owner_type IN ('workspace', 'marketplace')"
    )

    # Create indexes for efficient querying
    op.create_index(
        'idx_recipes_owner_type',
        'workflow_recipes',
        ['owner_type', 'is_approved']
    )

    op.create_index(
        'idx_recipes_marketplace_featured',
        'workflow_recipes',
        ['is_featured', 'install_count'],
        postgresql_where=sa.text("owner_type = 'marketplace'")
    )

    op.create_index(
        'idx_recipes_marketplace_category',
        'workflow_recipes',
        ['marketplace_category'],
        postgresql_where=sa.text("owner_type = 'marketplace'")
    )

    op.create_index(
        'idx_recipes_workspace',
        'workflow_recipes',
        ['workspace_id', 'owner_type']
    )

    op.create_index(
        'idx_recipes_cloned_from',
        'workflow_recipes',
        ['cloned_from_id']
    )

    # Data migration: Set owner_id to workspace_id for existing workspace recipes
    # First, set workspace_id to a default workspace if NULL
    op.execute("""
        UPDATE workflow_recipes
        SET workspace_id = (SELECT id FROM workspaces ORDER BY created_at LIMIT 1)
        WHERE workspace_id IS NULL AND owner_type = 'workspace'
    """)

    # Set owner_id to workspace_id for existing recipes
    op.execute("""
        UPDATE workflow_recipes
        SET owner_id = workspace_id::varchar
        WHERE owner_type = 'workspace' AND workspace_id IS NOT NULL
    """)

    # Mark existing recipes as approved (they were already in the system)
    op.execute("""
        UPDATE workflow_recipes
        SET is_approved = true
        WHERE owner_type = 'workspace'
    """)


def downgrade():
    """Remove marketplace fields from workflow_recipes table."""

    # Drop indexes
    op.drop_index('idx_recipes_cloned_from', table_name='workflow_recipes')
    op.drop_index('idx_recipes_workspace', table_name='workflow_recipes')
    op.drop_index('idx_recipes_marketplace_category', table_name='workflow_recipes')
    op.drop_index('idx_recipes_marketplace_featured', table_name='workflow_recipes')
    op.drop_index('idx_recipes_owner_type', table_name='workflow_recipes')

    # Drop constraints
    op.drop_constraint('ck_recipes_owner_type', 'workflow_recipes', type_='check')
    op.drop_constraint('fk_recipes_created_by_user_id', 'workflow_recipes', type_='foreignkey')
    op.drop_constraint('fk_recipes_original_creator_id', 'workflow_recipes', type_='foreignkey')
    op.drop_constraint('fk_recipes_cloned_from_id', 'workflow_recipes', type_='foreignkey')
    op.drop_constraint('fk_recipes_workspace_id', 'workflow_recipes', type_='foreignkey')

    # Drop columns
    op.drop_column('workflow_recipes', 'marketplace_icon')
    op.drop_column('workflow_recipes', 'marketplace_category')
    op.drop_column('workflow_recipes', 'is_approved')
    op.drop_column('workflow_recipes', 'install_count')
    op.drop_column('workflow_recipes', 'created_by_user_id')
    op.drop_column('workflow_recipes', 'original_creator_id')
    op.drop_column('workflow_recipes', 'cloned_from_id')
    op.drop_column('workflow_recipes', 'owner_id')
    op.drop_column('workflow_recipes', 'owner_type')
