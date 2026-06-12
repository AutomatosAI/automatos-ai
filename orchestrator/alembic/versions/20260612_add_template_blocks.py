"""Add blocks JSONB column to document_templates (PRD-167 S2).

The canonical block-tree body for a template ({"version", "blocks": [...]}). Nullable:
legacy templates keep rendering via template_content until migrated. Chains off the
document_templates table-creation migration (the table being altered).

Revision ID: 20260612_template_blocks
Revises: 20260218_document_templates
Create Date: 2026-06-12
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '20260612_template_blocks'
down_revision = '20260218_document_templates'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'document_templates',
        sa.Column('blocks', JSONB(), nullable=True),
    )


def downgrade():
    op.drop_column('document_templates', 'blocks')
