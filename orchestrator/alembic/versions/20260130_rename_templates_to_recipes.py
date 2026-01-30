"""rename workflow_templates to workflow_recipes

Revision ID: rename_templates_to_recipes
Revises:
Create Date: 2026-01-30

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'rename_templates_to_recipes'
down_revision = None  # Update this if there are previous migrations
branch_labels = None
depends_on = None


def upgrade():
    """Rename workflow_templates table to workflow_recipes"""
    # Rename the table
    op.rename_table('workflow_templates', 'workflow_recipes')

    # Rename the primary key constraint (if it exists)
    op.execute("""
        ALTER INDEX IF EXISTS workflow_templates_pkey
        RENAME TO workflow_recipes_pkey
    """)

    # Rename any other indexes
    op.execute("""
        ALTER INDEX IF EXISTS ix_workflow_templates_workspace_id
        RENAME TO ix_workflow_recipes_workspace_id
    """)

    op.execute("""
        ALTER INDEX IF EXISTS ix_workflow_templates_category
        RENAME TO ix_workflow_recipes_category
    """)

    op.execute("""
        ALTER INDEX IF EXISTS ix_workflow_templates_is_public
        RENAME TO ix_workflow_recipes_is_public
    """)

    op.execute("""
        ALTER INDEX IF EXISTS ix_workflow_templates_is_featured
        RENAME TO ix_workflow_recipes_is_featured
    """)


def downgrade():
    """Revert workflow_recipes table back to workflow_templates"""
    # Rename indexes back
    op.execute("""
        ALTER INDEX IF EXISTS ix_workflow_recipes_is_featured
        RENAME TO ix_workflow_templates_is_featured
    """)

    op.execute("""
        ALTER INDEX IF EXISTS ix_workflow_recipes_is_public
        RENAME TO ix_workflow_templates_is_public
    """)

    op.execute("""
        ALTER INDEX IF EXISTS ix_workflow_recipes_category
        RENAME TO ix_workflow_templates_category
    """)

    op.execute("""
        ALTER INDEX IF EXISTS ix_workflow_recipes_workspace_id
        RENAME TO ix_workflow_templates_workspace_id
    """)

    # Rename primary key back
    op.execute("""
        ALTER INDEX IF EXISTS workflow_recipes_pkey
        RENAME TO workflow_templates_pkey
    """)

    # Rename the table back
    op.rename_table('workflow_recipes', 'workflow_templates')
