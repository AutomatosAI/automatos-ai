"""add code graph tables

Revision ID: add_code_graph
Revises: add_context_policies
Create Date: 2025-08-12 14:20:03

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = 'add_code_graph'
down_revision = 'add_context_policies'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'code_symbols',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('project', sa.String(length=128), nullable=False),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('symbol_name', sa.String(length=256), nullable=False),
        sa.Column('symbol_type', sa.String(length=32), nullable=False),
        sa.Column('signature', sa.Text(), nullable=True),
        sa.Column('docstring', sa.Text(), nullable=True),
        sa.Column('start_line', sa.Integer(), nullable=True),
        sa.Column('end_line', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
    )
    op.create_index('ix_code_symbols_project_name', 'code_symbols', ['project', 'symbol_name'])

    op.create_table(
        'code_edges',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('project', sa.String(length=128), nullable=False),
        sa.Column('src_symbol_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('code_symbols.id', ondelete='CASCADE'), nullable=False),
        sa.Column('dst_symbol_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('code_symbols.id', ondelete='CASCADE'), nullable=False),
        sa.Column('edge_type', sa.String(length=32), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
    )
    op.create_index('ix_code_edges_project_type', 'code_edges', ['project', 'edge_type'])


def downgrade() -> None:
    op.drop_index('ix_code_edges_project_type', table_name='code_edges')
    op.drop_table('code_edges')
    op.drop_index('ix_code_symbols_project_name', table_name='code_symbols')
    op.drop_table('code_symbols')


