"""Tool Registry Schema - PRD-17

Revision ID: 004_tool_registry
Revises: 003_memory_system
Create Date: 2025-10-18

Description:
    Adds database tables for centralized tool registry system:
    - tool_registry: All registered tools with metadata
    - task_tool_mappings: Custom task-to-tool mappings
    - Enhances tool_usage_logs with additional tracking fields
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '004_tool_registry'
down_revision = '003_memory_system'
branch_labels = None
depends_on = None


def upgrade():
    """Create tool registry tables"""
    
    # Create tool_registry table
    op.create_table(
        'tool_registry',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tool_name', sa.String(255), nullable=False, unique=True),
        sa.Column('tool_category', sa.String(50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('executor_class', sa.String(255), nullable=False),
        sa.Column('executor_method', sa.String(255), nullable=False),
        sa.Column('parameters_schema', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('permissions_required', postgresql.JSONB(astext_type=sa.Text()), 
                  nullable=False, server_default='{"read": true}'),
        sa.Column('security_level', sa.String(50), nullable=False, server_default='safe'),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), 
                  nullable=False, server_default='{}'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, 
                  server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('idx_tool_registry_name', 'tool_registry', ['tool_name'])
    op.create_index('idx_tool_registry_category', 'tool_registry', ['tool_category'])
    op.create_index('idx_tool_registry_active', 'tool_registry', ['is_active'])
    
    # Create task_tool_mappings table
    op.create_table(
        'task_tool_mappings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('task_type', sa.String(100), nullable=False),
        sa.Column('tool_categories', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('required_tools', postgresql.JSONB(astext_type=sa.Text()), 
                  nullable=False, server_default='[]'),
        sa.Column('optional_tools', postgresql.JSONB(astext_type=sa.Text()), 
                  nullable=False, server_default='[]'),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('usage_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('success_rate', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, 
                  server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('idx_task_mappings_type', 'task_tool_mappings', ['task_type'])
    op.create_index('idx_task_mappings_confidence', 'task_tool_mappings', ['confidence'])
    
    # Enhance existing tool_usage_logs table (if it exists)
    # Check if table exists first
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    
    if 'tool_usage_logs' in inspector.get_table_names():
        # Add new columns to existing table
        op.add_column('tool_usage_logs', 
                     sa.Column('tool_category', sa.String(50), nullable=True))
        op.add_column('tool_usage_logs', 
                     sa.Column('task_type', sa.String(100), nullable=True))
        op.add_column('tool_usage_logs', 
                     sa.Column('security_validated', sa.Boolean(), 
                              nullable=False, server_default='false'))
        
        # Create indexes on new columns
        op.create_index('idx_tool_usage_category', 'tool_usage_logs', ['tool_category'])
        op.create_index('idx_tool_usage_task_type', 'tool_usage_logs', ['task_type'])


def downgrade():
    """Remove tool registry tables"""
    
    # Check if tool_usage_logs exists and has our columns
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    
    if 'tool_usage_logs' in inspector.get_table_names():
        # Drop indexes
        op.drop_index('idx_tool_usage_task_type', table_name='tool_usage_logs')
        op.drop_index('idx_tool_usage_category', table_name='tool_usage_logs')
        
        # Drop columns
        op.drop_column('tool_usage_logs', 'security_validated')
        op.drop_column('tool_usage_logs', 'task_type')
        op.drop_column('tool_usage_logs', 'tool_category')
    
    # Drop task_tool_mappings table
    op.drop_index('idx_task_mappings_confidence', table_name='task_tool_mappings')
    op.drop_index('idx_task_mappings_type', table_name='task_tool_mappings')
    op.drop_table('task_tool_mappings')
    
    # Drop tool_registry table
    op.drop_index('idx_tool_registry_active', table_name='tool_registry')
    op.drop_index('idx_tool_registry_category', table_name='tool_registry')
    op.drop_index('idx_tool_registry_name', table_name='tool_registry')
    op.drop_table('tool_registry')

