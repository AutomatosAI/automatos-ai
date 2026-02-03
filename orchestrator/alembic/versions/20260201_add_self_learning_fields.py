"""Add self-learning fields to workflow_recipes table

Revision ID: 20260201_self_learning
Revises: 20260201_workflow_definition
Create Date: 2026-02-01

This migration adds self-learning fields for the Learn-Quality-Memory stages:
- quality_score: Average quality from Stage 7 assessments
- learning_data: Patterns, suggestions, and performance metrics from Stage 6
- idx_recipes_quality: Index on quality_score DESC for efficient quality-based queries
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260201_self_learning'
down_revision = '20260201_workflow_definition'
branch_labels = None
depends_on = None


def upgrade():
    """Add self-learning fields to workflow_recipes table."""

    # Quality score: Average quality from Stage 7 assessments (0.0 - 1.0)
    op.add_column('workflow_recipes', sa.Column('quality_score', sa.Float(), nullable=True))

    # Learning data: Patterns, suggestions, and performance metrics from Stage 6
    # Example: { "patterns": [...], "suggestions": [...], "performance_metrics": {...} }
    op.add_column('workflow_recipes', sa.Column('learning_data', postgresql.JSONB(), nullable=True))

    # Index for efficient quality-based queries (e.g., "show me top-rated recipes")
    op.create_index('idx_recipes_quality', 'workflow_recipes', [sa.text('quality_score DESC')], if_not_exists=True)


def downgrade():
    """Remove self-learning fields from workflow_recipes table."""

    op.drop_index('idx_recipes_quality', table_name='workflow_recipes', if_exists=True)
    op.drop_column('workflow_recipes', 'learning_data')
    op.drop_column('workflow_recipes', 'quality_score')
