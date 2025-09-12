"""Add telemetry tables for agent run tracking and pattern mining

Revision ID: 20250906_add_telemetry
Revises: 20250812_142003_add_code_graph
Create Date: 2025-09-06 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '20250906_add_telemetry'
down_revision = '20250812_142003_add_code_graph'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create run_traces table for tracking agent executions
    op.create_table('run_traces',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.String(length=255), nullable=False),
        sa.Column('agent_id', sa.String(length=255), nullable=False),
        sa.Column('task_type', sa.String(length=255), nullable=False),
        sa.Column('start_time', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=50), server_default='running'),
        sa.Column('success', sa.Boolean(), nullable=True),
        sa.Column('input_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('output_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('context_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('run_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_run_traces_run_id', 'run_traces', ['run_id'], unique=True)
    op.create_index('idx_run_traces_agent_id', 'run_traces', ['agent_id'])
    op.create_index('idx_run_traces_task_type', 'run_traces', ['task_type'])
    op.create_index('idx_run_traces_start_time', 'run_traces', ['start_time'])
    op.create_index('idx_run_traces_success', 'run_traces', ['success'])
    op.create_index('idx_run_traces_status', 'run_traces', ['status'])

    # Create run_events table for detailed step tracking
    op.create_table('run_events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.String(length=255), nullable=False),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('step_name', sa.String(length=255), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('event_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_run_events_run_id', 'run_events', ['run_id'])
    op.create_index('idx_run_events_event_type', 'run_events', ['event_type'])
    op.create_index('idx_run_events_timestamp', 'run_events', ['timestamp'])

    # Create playbook_executions table
    op.create_table('playbook_executions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('playbook_id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.String(length=255), nullable=False),
        sa.Column('executed_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('success', sa.Boolean(), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('cost', sa.Float(), server_default='0.0'),
        sa.Column('steps_completed', sa.Integer(), server_default='0'),
        sa.Column('steps_total', sa.Integer(), server_default='0'),
        sa.Column('efficiency_score', sa.Float(), nullable=True),
        sa.Column('deviations', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('improvements', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_playbook_executions_playbook_id', 'playbook_executions', ['playbook_id'])
    op.create_index('idx_playbook_executions_run_id', 'playbook_executions', ['run_id'])


def downgrade() -> None:
    op.drop_table('playbook_executions')
    op.drop_table('run_events')
    op.drop_table('run_traces')

