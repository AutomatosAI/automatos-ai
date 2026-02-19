"""fix codegraph schema — replace legacy code_symbols/code_edges with
proper codegraph_* tables matching codegraph_service.py

Revision ID: 20260218_codegraph_v2
Revises: 20260215_heartbeat_channels
Create Date: 2026-02-18

PRD-62: CodeGraph v2 Competitive Upgrade
- Fixes schema mismatch (Bug #1 from code audit)
- Adds codegraph_files table with file_hash for incremental indexing
- Proper FK relationships and indexes
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = '20260218_codegraph_v2'
down_revision = '20260215_heartbeat_channels'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop legacy tables from original PRD-11 migration (if they exist)
    op.execute("DROP TABLE IF EXISTS code_edges CASCADE")
    op.execute("DROP TABLE IF EXISTS code_symbols CASCADE")

    # ── codegraph_projects ─────────────────────────────────────────
    op.create_table(
        'codegraph_projects',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('source_type', sa.String(50), server_default='github'),
        sa.Column('source_url', sa.Text()),
        sa.Column('branch', sa.String(255), server_default='main'),
        sa.Column('status', sa.String(50), server_default='pending'),
        sa.Column('total_files', sa.Integer(), server_default='0'),
        sa.Column('total_symbols', sa.Integer(), server_default='0'),
        sa.Column('total_relationships', sa.Integer(), server_default='0'),
        sa.Column('language', sa.String(50)),
        sa.Column('last_indexed', sa.DateTime(timezone=True)),
        sa.Column('index_duration_seconds', sa.Float()),
        sa.Column('exclude_patterns', postgresql.ARRAY(sa.Text()), server_default='{}'),
        sa.Column('auto_reindex', sa.Boolean(), server_default='false'),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )
    op.create_index('idx_codegraph_projects_workspace', 'codegraph_projects', ['workspace_id'])

    # ── codegraph_files ────────────────────────────────────────────
    op.create_table(
        'codegraph_files',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('project_id', sa.Integer(),
                  sa.ForeignKey('codegraph_projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('file_hash', sa.String(64)),  # SHA-256 for incremental indexing
        sa.Column('file_size', sa.Integer()),
        sa.Column('lines_of_code', sa.Integer()),
        sa.Column('language', sa.String(50)),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('indexed_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )
    op.create_index('idx_codegraph_files_project', 'codegraph_files', ['project_id'])
    op.create_index('idx_codegraph_files_hash', 'codegraph_files', ['file_hash'])
    op.create_index('idx_codegraph_files_path', 'codegraph_files', ['project_id', 'file_path'])

    # ── codegraph_symbols ──────────────────────────────────────────
    op.create_table(
        'codegraph_symbols',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('project_id', sa.Integer(),
                  sa.ForeignKey('codegraph_projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('symbol_type', sa.String(50), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('qualified_name', sa.Text()),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('line_number', sa.Integer()),
        sa.Column('signature', sa.Text()),
        sa.Column('docstring', sa.Text()),
        sa.Column('code_snippet', sa.Text()),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}'),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )
    op.create_index('idx_codegraph_symbols_project', 'codegraph_symbols', ['project_id'])
    op.create_index('idx_codegraph_symbols_name', 'codegraph_symbols', ['name'])
    op.create_index('idx_codegraph_symbols_qualified', 'codegraph_symbols', ['qualified_name'])
    op.create_index('idx_codegraph_symbols_file', 'codegraph_symbols', ['project_id', 'file_path'])

    # Note: embedding vector column is managed dynamically by codegraph_service.py
    # via _ensure_embedding_dimension() to match the configured embedding model size.

    # ── codegraph_relationships ────────────────────────────────────
    op.create_table(
        'codegraph_relationships',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('project_id', sa.Integer(),
                  sa.ForeignKey('codegraph_projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('from_symbol_id', sa.Integer(),
                  sa.ForeignKey('codegraph_symbols.id', ondelete='CASCADE')),
        sa.Column('to_symbol_id', sa.Integer(),
                  sa.ForeignKey('codegraph_symbols.id', ondelete='CASCADE')),
        sa.Column('relationship_type', sa.String(50), nullable=False),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}'),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )
    op.create_index('idx_codegraph_rels_project', 'codegraph_relationships', ['project_id'])
    op.create_index('idx_codegraph_rels_from', 'codegraph_relationships', ['from_symbol_id'])
    op.create_index('idx_codegraph_rels_to', 'codegraph_relationships', ['to_symbol_id'])
    op.create_index('idx_codegraph_rels_type', 'codegraph_relationships', ['relationship_type'])

    # ── codegraph_query_logs ───────────────────────────────────────
    op.create_table(
        'codegraph_query_logs',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('project_id', sa.Integer(),
                  sa.ForeignKey('codegraph_projects.id', ondelete='CASCADE')),
        sa.Column('query_text', sa.Text(), nullable=False),
        sa.Column('query_type', sa.String(50)),
        sa.Column('results_count', sa.Integer()),
        sa.Column('duration_ms', sa.Float()),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )


def downgrade() -> None:
    op.drop_table('codegraph_query_logs')
    op.drop_table('codegraph_relationships')
    op.drop_table('codegraph_symbols')
    op.drop_table('codegraph_files')
    op.drop_table('codegraph_projects')

    # Restore legacy tables (from original PRD-11 migration)
    op.create_table(
        'code_symbols',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('project', sa.String(128), nullable=False),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('symbol_name', sa.String(256), nullable=False),
        sa.Column('symbol_type', sa.String(32), nullable=False),
        sa.Column('signature', sa.Text()),
        sa.Column('docstring', sa.Text()),
        sa.Column('start_line', sa.Integer()),
        sa.Column('end_line', sa.Integer()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )
    op.create_index('ix_code_symbols_project_name', 'code_symbols', ['project', 'symbol_name'])

    op.create_table(
        'code_edges',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('project', sa.String(128), nullable=False),
        sa.Column('src_symbol_id', postgresql.UUID(as_uuid=False),
                  sa.ForeignKey('code_symbols.id', ondelete='CASCADE'), nullable=False),
        sa.Column('dst_symbol_id', postgresql.UUID(as_uuid=False),
                  sa.ForeignKey('code_symbols.id', ondelete='CASCADE'), nullable=False),
        sa.Column('edge_type', sa.String(32), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )
    op.create_index('ix_code_edges_project_type', 'code_edges', ['project', 'edge_type'])
