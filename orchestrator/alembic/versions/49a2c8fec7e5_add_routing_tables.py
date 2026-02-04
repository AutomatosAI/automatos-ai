"""add routing tables

Revision ID: 49a2c8fec7e5
Revises: 20260202_workspace_isolation
Create Date: 2026-02-03 19:57:09.388134

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '49a2c8fec7e5'
down_revision = '20260202_workspace_isolation'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create routing_decisions, routing_rules, and unrouted_events tables."""

    op.create_table('routing_decisions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('request_id', sa.UUID(), nullable=False),
        sa.Column('envelope_hash', sa.String(length=64), nullable=False),
        sa.Column('workspace_id', sa.UUID(), nullable=True),
        sa.Column('source', sa.String(length=50), nullable=False),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('route_type', sa.String(length=50), nullable=False),
        sa.Column('agent_id', sa.Integer(), nullable=True),
        sa.Column('workflow_id', sa.Integer(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('cached', sa.Boolean(), nullable=False),
        sa.Column('was_corrected', sa.Boolean(), nullable=False),
        sa.Column('corrected_agent_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_routing_decisions_envelope_hash', 'routing_decisions', ['envelope_hash'])
    op.create_index('ix_routing_decisions_request_id', 'routing_decisions', ['request_id'])
    op.create_index('ix_routing_decisions_workspace_id', 'routing_decisions', ['workspace_id'])

    op.create_table('routing_rules',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('workspace_id', sa.UUID(), nullable=False),
        sa.Column('source_pattern', sa.String(length=100), nullable=True),
        sa.Column('intent_keywords', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('target_agent_id', sa.Integer(), nullable=True),
        sa.Column('target_workflow_id', sa.Integer(), nullable=True),
        sa.Column('priority', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_routing_rules_workspace_id', 'routing_rules', ['workspace_id'])

    op.create_table('unrouted_events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('workspace_id', sa.UUID(), nullable=False),
        sa.Column('source', sa.String(length=50), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('raw_payload', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('reason', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_unrouted_events_workspace_id', 'unrouted_events', ['workspace_id'])


def downgrade() -> None:
    """Drop routing tables."""
    op.drop_index('ix_unrouted_events_workspace_id', table_name='unrouted_events')
    op.drop_table('unrouted_events')
    op.drop_index('ix_routing_rules_workspace_id', table_name='routing_rules')
    op.drop_table('routing_rules')
    op.drop_index('ix_routing_decisions_workspace_id', table_name='routing_decisions')
    op.drop_index('ix_routing_decisions_request_id', table_name='routing_decisions')
    op.drop_index('ix_routing_decisions_envelope_hash', table_name='routing_decisions')
    op.drop_table('routing_decisions')
