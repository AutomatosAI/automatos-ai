"""add recipe_type field for hybrid recipes

Revision ID: add_recipe_type
Revises: rename_templates_to_recipes
Create Date: 2026-01-30

US-010: Support both simple (prompt-based) and complex (multi-step) recipes
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = 'add_recipe_type'
down_revision = 'rename_templates_to_recipes'
branch_labels = None
depends_on = None


def upgrade():
    """Add recipe_type field and simple recipe fields"""
    # Add recipe_type enum column (default to 'complex' for existing recipes)
    op.execute("""
        ALTER TABLE workflow_recipes
        ADD COLUMN recipe_type VARCHAR(20) DEFAULT 'complex' NOT NULL
    """)

    # Add simple recipe fields (stored in metadata JSONB for flexibility)
    # - For simple recipes: metadata will contain:
    #   - agent_id: int
    #   - prompt: text
    #   - inputs: [{name, type, required, default}]
    #   - schedule: optional cron string
    # - For complex recipes: metadata contains existing template_definition structure

    # Create index on recipe_type for filtering
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_workflow_recipes_recipe_type
        ON workflow_recipes(recipe_type)
    """)


def downgrade():
    """Remove recipe_type field"""
    op.execute("""
        DROP INDEX IF EXISTS ix_workflow_recipes_recipe_type
    """)

    op.execute("""
        ALTER TABLE workflow_recipes
        DROP COLUMN recipe_type
    """)
