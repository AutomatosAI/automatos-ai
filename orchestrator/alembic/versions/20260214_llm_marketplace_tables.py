"""PRD-54: LLM Provider Marketplace — new tables and fields

Revision ID: 20260214_llm_marketplace
Revises: 20260213_ws_webhook_key
Create Date: 2026-02-14

Adds:
  - New columns on llm_models: tier, tags, category, popularity_score,
    install_count, avg_rating, is_featured, is_default, requires_plan,
    external_id, pricing_updated_at
  - New table: workspace_models (which models a workspace has installed)
  - New table: user_api_keys (BYOK encrypted API keys per workspace)
  - New table: llm_usage (per-request usage tracking for analytics & billing)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers
revision = '20260214_llm_marketplace'
down_revision = '20260213_ws_webhook_key'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Add marketplace columns to llm_models ──────────────────────────
    op.add_column('llm_models', sa.Column('tier', sa.String(50), server_default='direct'))
    op.add_column('llm_models', sa.Column('tags', sa.JSON(), server_default='[]'))
    op.add_column('llm_models', sa.Column('category', sa.String(100)))
    op.add_column('llm_models', sa.Column('popularity_score', sa.Integer(), server_default='0'))
    op.add_column('llm_models', sa.Column('install_count', sa.Integer(), server_default='0'))
    op.add_column('llm_models', sa.Column('avg_rating', sa.Float()))
    op.add_column('llm_models', sa.Column('is_featured', sa.Boolean(), server_default='false'))
    op.add_column('llm_models', sa.Column('is_default', sa.Boolean(), server_default='false'))
    op.add_column('llm_models', sa.Column('requires_plan', sa.String(50)))
    op.add_column('llm_models', sa.Column('external_id', sa.String(255)))
    op.add_column('llm_models', sa.Column('pricing_updated_at', sa.DateTime()))

    # ── workspace_models ───────────────────────────────────────────────
    op.create_table(
        'workspace_models',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', UUID(as_uuid=True), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('model_id', sa.Integer(), sa.ForeignKey('llm_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('installed_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('source', sa.String(50), server_default='marketplace'),
    )
    op.create_unique_constraint('uq_workspace_model', 'workspace_models', ['workspace_id', 'model_id'])

    # ── user_api_keys ──────────────────────────────────────────────────
    op.create_table(
        'user_api_keys',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', UUID(as_uuid=True), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('provider', sa.String(50), nullable=False),
        sa.Column('encrypted_key', sa.Text(), nullable=False),
        sa.Column('display_name', sa.String(255)),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('last_used_at', sa.DateTime()),
        sa.Column('usage_count', sa.Integer(), server_default='0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
    )

    # ── llm_usage ──────────────────────────────────────────────────────
    op.create_table(
        'llm_usage',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('workspace_id', UUID(as_uuid=True), sa.ForeignKey('workspaces.id'), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('provider', sa.String(100), nullable=False),
        sa.Column('tier', sa.String(50), nullable=False),
        # Request details
        sa.Column('agent_id', sa.Integer()),
        sa.Column('execution_id', sa.String(255)),
        sa.Column('request_type', sa.String(50)),
        # Token usage
        sa.Column('input_tokens', sa.Integer(), nullable=False),
        sa.Column('output_tokens', sa.Integer(), nullable=False),
        sa.Column('total_tokens', sa.Integer(), nullable=False),
        # Cost
        sa.Column('input_cost', sa.Float(), nullable=False),
        sa.Column('output_cost', sa.Float(), nullable=False),
        sa.Column('total_cost', sa.Float(), nullable=False),
        sa.Column('is_byok', sa.Boolean(), server_default='false'),
        # Performance
        sa.Column('latency_ms', sa.Integer()),
        sa.Column('status', sa.String(50)),
        sa.Column('error_message', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )

    # Indexes for analytics queries
    op.create_index('idx_llm_usage_workspace_created', 'llm_usage', ['workspace_id', 'created_at'])
    op.create_index('idx_llm_usage_model_created', 'llm_usage', ['model_id', 'created_at'])
    op.create_index('idx_llm_usage_agent', 'llm_usage', ['agent_id', 'created_at'])


def downgrade() -> None:
    op.drop_index('idx_llm_usage_agent', table_name='llm_usage')
    op.drop_index('idx_llm_usage_model_created', table_name='llm_usage')
    op.drop_index('idx_llm_usage_workspace_created', table_name='llm_usage')
    op.drop_table('llm_usage')
    op.drop_table('user_api_keys')
    op.drop_constraint('uq_workspace_model', 'workspace_models', type_='unique')
    op.drop_table('workspace_models')

    op.drop_column('llm_models', 'pricing_updated_at')
    op.drop_column('llm_models', 'external_id')
    op.drop_column('llm_models', 'requires_plan')
    op.drop_column('llm_models', 'is_default')
    op.drop_column('llm_models', 'is_featured')
    op.drop_column('llm_models', 'avg_rating')
    op.drop_column('llm_models', 'install_count')
    op.drop_column('llm_models', 'popularity_score')
    op.drop_column('llm_models', 'category')
    op.drop_column('llm_models', 'tags')
    op.drop_column('llm_models', 'tier')
