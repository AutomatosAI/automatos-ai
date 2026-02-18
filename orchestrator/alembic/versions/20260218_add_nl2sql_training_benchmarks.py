"""Add NL2SQL training examples and benchmark tables (PRD-61)

Revision ID: 20260218_nl2sql_training
Revises: 20260215_heartbeat_channels
Create Date: 2026-02-18 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260218_nl2sql_training'
down_revision = '20260215_heartbeat_channels'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # NL2SQL Training Examples (US-001, US-002)
    op.create_table(
        'nl2sql_training_examples',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('database_source_id', sa.Integer(), nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('sql', sa.Text(), nullable=False),
        sa.Column('tables_used', postgresql.JSONB(), server_default='[]'),
        sa.Column('is_verified', sa.Boolean(), server_default='false'),
        sa.Column('verification_source', sa.String(50)),
        sa.Column('usage_count', sa.Integer(), server_default='0'),
        sa.Column('last_used_at', sa.DateTime()),
        sa.Column('embedding_id', sa.String(255)),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}'),
        sa.Column('created_by', sa.String(255)),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id']),
        sa.ForeignKeyConstraint(['database_source_id'], ['database_knowledge_sources.id']),
    )

    op.create_index('idx_nl2sql_examples_workspace', 'nl2sql_training_examples', ['workspace_id'])
    op.create_index('idx_nl2sql_examples_source', 'nl2sql_training_examples', ['database_source_id'])
    op.create_index(
        'idx_nl2sql_examples_verified',
        'nl2sql_training_examples',
        ['is_verified'],
        postgresql_where=sa.text('is_verified = TRUE')
    )

    # NL2SQL Benchmark Runs (US-017, US-018)
    op.create_table(
        'nl2sql_benchmark_runs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('database_source_id', sa.Integer(), nullable=False),
        sa.Column('total_examples', sa.Integer(), server_default='0'),
        sa.Column('exact_match_rate', sa.Float(), server_default='0.0'),
        sa.Column('execution_match_rate', sa.Float(), server_default='0.0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id']),
        sa.ForeignKeyConstraint(['database_source_id'], ['database_knowledge_sources.id']),
    )

    # NL2SQL Benchmark Results (US-018)
    op.create_table(
        'nl2sql_benchmark_results',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('benchmark_run_id', sa.Integer(), nullable=False),
        sa.Column('question', sa.Text()),
        sa.Column('expected_sql', sa.Text()),
        sa.Column('generated_sql', sa.Text()),
        sa.Column('exact_match', sa.Boolean(), server_default='false'),
        sa.Column('execution_match', sa.Boolean(), server_default='false'),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['benchmark_run_id'], ['nl2sql_benchmark_runs.id']),
    )


def downgrade() -> None:
    op.drop_table('nl2sql_benchmark_results')
    op.drop_table('nl2sql_benchmark_runs')
    op.drop_index('idx_nl2sql_examples_verified', table_name='nl2sql_training_examples')
    op.drop_index('idx_nl2sql_examples_source', table_name='nl2sql_training_examples')
    op.drop_index('idx_nl2sql_examples_workspace', table_name='nl2sql_training_examples')
    op.drop_table('nl2sql_training_examples')
