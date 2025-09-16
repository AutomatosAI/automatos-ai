"""Add missing columns to agents table

Revision ID: fix_agents_schema_202509
Revises: 6203026dbac0
Create Date: 2025-09-09 21:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fix_agents_schema_202509'
down_revision = '6203026dbac0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add missing columns to agents table that API expects
    try:
        # Add priority_level column if it doesn't exist
        op.add_column('agents', sa.Column('priority_level', sa.String(length=50), nullable=True, server_default='medium'))
    except Exception:
        pass  # Column might already exist
    
    try:
        # Add max_concurrent_tasks column if it doesn't exist
        op.add_column('agents', sa.Column('max_concurrent_tasks', sa.Integer(), nullable=True, server_default='5'))
    except Exception:
        pass  # Column might already exist
    
    try:
        # Add auto_start column if it doesn't exist
        op.add_column('agents', sa.Column('auto_start', sa.Boolean(), nullable=True, server_default='false'))
    except Exception:
        pass  # Column might already exist
    
    # Ensure configuration column is JSON type, not TEXT
    try:
        op.alter_column('agents', 'configuration', 
                       existing_type=sa.Text(),
                       type_=sa.JSON(),
                       existing_nullable=True)
    except Exception:
        pass  # Column might already be JSON
    
    # Ensure performance_metrics column is JSON type, not TEXT
    try:
        op.alter_column('agents', 'performance_metrics', 
                       existing_type=sa.Text(),
                       type_=sa.JSON(),
                       existing_nullable=True)
    except Exception:
        pass  # Column might already be JSON


def downgrade() -> None:
    # Remove the columns we added
    try:
        op.drop_column('agents', 'auto_start')
    except Exception:
        pass
    
    try:
        op.drop_column('agents', 'max_concurrent_tasks')
    except Exception:
        pass
    
    try:
        op.drop_column('agents', 'priority_level')
    except Exception:
        pass
