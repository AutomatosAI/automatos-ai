"""Add heartbeat_results and channel_connections tables (PRD-55)

Revision ID: 20260215_heartbeat_channels
Revises: 20260214_merge_heads
Create Date: 2026-02-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260215_heartbeat_channels'
down_revision = '20260214_merge_heads'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Heartbeat results table (US-008)
    op.create_table(
        'heartbeat_results',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('source_type', sa.String(20), nullable=False),  # 'agent' or 'orchestrator'
        sa.Column('source_id', sa.String(255), nullable=False),  # agent_id or workspace_id
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='success'),
        sa.Column('findings', postgresql.JSONB(), server_default='[]'),
        sa.Column('actions_taken', postgresql.JSONB(), server_default='[]'),
        sa.Column('tokens_used', sa.Integer(), server_default='0'),
        sa.Column('cost', sa.Float(), server_default='0.0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_heartbeat_results_workspace_id', 'heartbeat_results', ['workspace_id'])
    op.create_index('ix_heartbeat_results_source', 'heartbeat_results', ['source_type', 'source_id'])
    op.create_index('ix_heartbeat_results_created_at', 'heartbeat_results', ['created_at'])

    # Channel connections table (US-019)
    op.create_table(
        'channel_connections',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('platform', sa.String(50), nullable=False),
        sa.Column('config', postgresql.JSONB(), server_default='{}'),
        sa.Column('status', sa.String(20), server_default='inactive'),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}'),
        sa.Column('default_agent_id', sa.Integer(), nullable=True),
        sa.Column('message_count', sa.Integer(), server_default='0'),
        sa.Column('last_activity_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_channel_connections_workspace_id', 'channel_connections', ['workspace_id'])
    op.create_index('ix_channel_connections_platform', 'channel_connections', ['platform'])


def downgrade() -> None:
    op.drop_index('ix_channel_connections_platform', table_name='channel_connections')
    op.drop_index('ix_channel_connections_workspace_id', table_name='channel_connections')
    op.drop_table('channel_connections')

    op.drop_index('ix_heartbeat_results_created_at', table_name='heartbeat_results')
    op.drop_index('ix_heartbeat_results_source', table_name='heartbeat_results')
    op.drop_index('ix_heartbeat_results_workspace_id', table_name='heartbeat_results')
    op.drop_table('heartbeat_results')
