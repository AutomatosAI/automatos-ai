"""Merge heads: app_suggestions and ownership_columns

Revision ID: 20260129_merge
Revises: 20260129_app_suggestions, c551b7ae7bec
Create Date: 2026-01-29

Merge migration to combine parallel migration branches.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20260129_merge'
down_revision = ('20260129_app_suggestions', 'c551b7ae7bec')
branch_labels = None
depends_on = None


def upgrade():
    """No-op merge migration."""
    pass


def downgrade():
    """No-op merge migration."""
    pass
