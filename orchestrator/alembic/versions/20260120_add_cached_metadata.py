"""Add cached_metadata to tenant_tool_config

Revision ID: 20260120_add_cached_metadata
Revises: 20260117_add_tool_assignment_tables
Create Date: 2026-01-20 11:15:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260120_add_cached_metadata'
down_revision = '20260117_tools'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('tenant_tool_config', sa.Column('cached_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade():
    op.drop_column('tenant_tool_config', 'cached_metadata')
