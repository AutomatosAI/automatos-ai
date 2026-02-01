"""Merge heads: marketplace agents and recipe type

Revision ID: 20260131_merge
Revises: add_recipe_type, 20260131_marketplace_installs
Create Date: 2026-01-31

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20260131_merge'
down_revision = ('add_recipe_type', '20260131_marketplace_installs')
branch_labels = None
depends_on = None


def upgrade():
    """Merge migrations - no changes needed"""
    pass


def downgrade():
    """Merge migrations - no changes needed"""
    pass
