
"""merge prd176 + prd181 heads

Revision ID: e773c09189a9
Revises: prd176_merge_heads, prd181_s2_approval_grants
Create Date: 2026-07-03 21:59:48.797434

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e773c09189a9'
down_revision = ('prd176_merge_heads', 'prd181_s2_approval_grants')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
