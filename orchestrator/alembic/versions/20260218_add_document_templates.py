"""Add document_templates table and update artifacts kind constraint (PRD-63)

Revision ID: 20260218_document_templates
Revises: 20260218_codegraph_v2
Create Date: 2026-02-18

Adds:
- document_templates table for storing PDF/DOCX/XLSX templates
- Updates artifacts.kind constraint to include 'document'
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260218_document_templates'
down_revision = '20260218_codegraph_v2'
branch_labels = None
depends_on = None


def upgrade():
    # Create document_templates table
    op.create_table(
        'document_templates',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('format', sa.String(20), nullable=False),
        sa.Column('template_content', sa.Text(), nullable=True),
        sa.Column('template_file_path', sa.String(500), nullable=True),
        sa.Column('data_schema', postgresql.JSONB(), server_default='{}', nullable=False),
        sa.Column('sample_data', postgresql.JSONB(), server_default='{}', nullable=True),
        sa.Column('category', sa.String(100), server_default='general', nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('thumbnail_url', sa.String(500), nullable=True),
        sa.Column('version', sa.Integer(), server_default='1', nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=True),
        sa.Column('created_by', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("format IN ('pdf', 'docx', 'xlsx')", name='check_document_template_format'),
        sa.UniqueConstraint('workspace_id', 'name', 'version', name='uq_template_workspace_name_version'),
    )
    op.create_index('idx_templates_workspace', 'document_templates', ['workspace_id'])
    op.create_index('idx_templates_format', 'document_templates', ['format'])
    op.create_index('idx_templates_category', 'document_templates', ['category'])

    # Update artifacts kind constraint to include 'document'
    op.drop_constraint('artifacts_kind_check', 'artifacts', type_='check')
    op.create_check_constraint(
        'artifacts_kind_check',
        'artifacts',
        "kind IN ('code', 'text', 'image', 'sheet', 'document')"
    )


def downgrade():
    # Restore original artifacts constraint
    op.drop_constraint('artifacts_kind_check', 'artifacts', type_='check')
    op.create_check_constraint(
        'artifacts_kind_check',
        'artifacts',
        "kind IN ('code', 'text', 'image', 'sheet')"
    )

    # Drop indexes
    op.drop_index('idx_templates_category', 'document_templates')
    op.drop_index('idx_templates_format', 'document_templates')
    op.drop_index('idx_templates_workspace', 'document_templates')

    # Drop table
    op.drop_table('document_templates')
