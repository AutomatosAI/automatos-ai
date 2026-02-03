"""Add recipe_executions table

Revision ID: 20260201_recipe_executions
Revises: 20260201_remove_legacy
Create Date: 2026-02-01

Creates recipe_executions table for tracking recipe execution state.
Each execution tracks: recipe, workspace, input data, output data,
step-level results, status, and timestamps.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision = '20260201_recipe_executions'
down_revision = '20260201_remove_legacy'
branch_labels = None
depends_on = None


def upgrade():
    """Create recipe_executions table."""
    op.create_table(
        'recipe_executions',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('execution_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('recipe_id', sa.Integer, sa.ForeignKey('workflow_recipes.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('workspace_id', UUID(as_uuid=True), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='pending'),  # pending, running, completed, failed, cancelled
        sa.Column('input_data', JSONB, nullable=True),
        sa.Column('output_data', JSONB, nullable=True),
        sa.Column('step_results', JSONB, nullable=True),  # Per-step execution results
        sa.Column('current_stage', sa.Integer, nullable=True),  # Current 9-stage position (1-9)
        sa.Column('current_step', sa.Integer, nullable=True),  # Current workflow step index
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('started_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('completed_at', sa.DateTime, nullable=True),
        sa.Column('triggered_by', sa.String(255), nullable=True),  # user email or 'cron' or 'webhook'
        sa.Column('execution_metadata', JSONB, nullable=True),  # Additional metadata (models used, tokens, costs)
    )

    # Index for querying executions by recipe + status
    op.create_index(
        'idx_recipe_executions_recipe_status',
        'recipe_executions',
        ['recipe_id', 'status'],
    )


def downgrade():
    """Drop recipe_executions table."""
    op.drop_index('idx_recipe_executions_recipe_status', table_name='recipe_executions')
    op.drop_table('recipe_executions')
