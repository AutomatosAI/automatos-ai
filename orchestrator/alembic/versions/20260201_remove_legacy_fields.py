"""Remove legacy fields from workflow_recipes table

Revision ID: 20260201_remove_legacy
Revises: 20260201_self_learning
Create Date: 2026-02-01

This migration removes legacy fields that are no longer needed:
- category: moved to marketplace_category for marketplace items only
- difficulty: not needed - complexity auto-calculated
- estimated_time: not needed - actual execution time tracked
- icon: moved to marketplace_icon for marketplace items only
- recipe_type: all recipes are workflows now

Also drops the index ix_workflow_recipes_category on the category column.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20260201_remove_legacy'
down_revision = '20260201_self_learning'
branch_labels = None
depends_on = None


def upgrade():
    """Remove legacy fields from workflow_recipes table."""

    # Drop the category index first (depends on category column)
    op.drop_index('ix_workflow_recipes_category', table_name='workflow_recipes', if_exists=True)

    # Drop legacy columns
    op.drop_column('workflow_recipes', 'category')
    op.drop_column('workflow_recipes', 'difficulty')
    op.drop_column('workflow_recipes', 'estimated_time')
    op.drop_column('workflow_recipes', 'icon')
    op.drop_column('workflow_recipes', 'recipe_type')


def downgrade():
    """Re-add legacy fields to workflow_recipes table."""

    op.add_column('workflow_recipes', sa.Column('recipe_type', sa.String(20), nullable=False, server_default='complex'))
    op.add_column('workflow_recipes', sa.Column('icon', sa.String(50), nullable=True))
    op.add_column('workflow_recipes', sa.Column('estimated_time', sa.String(50), nullable=True))
    op.add_column('workflow_recipes', sa.Column('difficulty', sa.String(50), server_default='intermediate'))
    op.add_column('workflow_recipes', sa.Column('category', sa.String(100), nullable=False, server_default='General'))

    # Re-create the category index
    op.create_index('ix_workflow_recipes_category', 'workflow_recipes', ['category'])
