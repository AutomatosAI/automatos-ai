"""PRD-42: Add personas table, agent persona columns, and plugin marketplace tables

Revision ID: 20260205_personas_plugins
Revises: 20260202_workspace_isolation
Create Date: 2026-02-05

Creates:
  - personas table (agent personality profiles)
  - persona columns on agents table (persona_id, custom_persona_prompt, use_custom_persona)
  - plugin_categories table
  - plugin_security_scans table (must exist before marketplace_plugins FK)
  - marketplace_plugins table
  - plugin_sync_history table
  - workspace_enabled_plugins junction table
  - agent_assigned_plugins junction table
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '20260205_personas_plugins'
down_revision = '20260202_workspace_isolation'
branch_labels = None
depends_on = None


def upgrade():
    # ── 1. Personas table ──────────────────────────────────────────────
    op.create_table(
        'personas',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('slug', sa.String(100), unique=True, nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('system_prompt', sa.Text),
        sa.Column('voice_description', sa.String(500)),
        sa.Column('category', sa.String(100)),
        sa.Column('tags', postgresql.ARRAY(sa.String), server_default='{}'),
        sa.Column('suggested_temperature', sa.Float, server_default='0.7'),
        sa.Column('suggested_models', postgresql.ARRAY(sa.String), server_default='{}'),
        sa.Column('source', sa.String(100)),
        sa.Column('source_url', sa.String(500)),
        sa.Column('scope', sa.String(20), server_default='global'),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('workspaces.id'), nullable=True),
        sa.Column('is_active', sa.Boolean, server_default='true'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
    )
    op.create_index('idx_personas_scope', 'personas', ['scope', 'workspace_id'])
    op.create_index('idx_personas_category', 'personas', ['category'])

    # ── 2. Agent persona columns ───────────────────────────────────────
    op.add_column('agents', sa.Column('persona_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('agents', sa.Column('custom_persona_prompt', sa.Text, nullable=True))
    op.add_column('agents', sa.Column('use_custom_persona', sa.Boolean, server_default='false', nullable=False))

    op.create_foreign_key('fk_agents_persona_id', 'agents', 'personas', ['persona_id'], ['id'], ondelete='SET NULL')
    op.create_index('idx_agents_persona_id', 'agents', ['persona_id'])

    # ── 3. Plugin categories ───────────────────────────────────────────
    op.create_table(
        'plugin_categories',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('slug', sa.String(50), unique=True, nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('icon', sa.String(50)),
        sa.Column('sort_order', sa.Integer, server_default='0'),
        sa.Column('parent_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('plugin_categories.id'), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
    )

    # ── 4. Plugin security scans (referenced by marketplace_plugins FK)
    op.create_table(
        'plugin_security_scans',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('plugin_slug', sa.String(100), nullable=False),
        sa.Column('plugin_version', sa.String(50), nullable=False),
        sa.Column('static_scan_status', sa.String(50)),
        sa.Column('static_findings', postgresql.JSONB, server_default='[]'),
        sa.Column('blocked_patterns_found', postgresql.ARRAY(sa.String), server_default='{}'),
        sa.Column('llm_scan_status', sa.String(50)),
        sa.Column('llm_risk_score', sa.Integer),
        sa.Column('llm_findings', postgresql.JSONB, server_default='[]'),
        sa.Column('llm_summary', sa.Text),
        sa.Column('llm_model_used', sa.String(100)),
        sa.Column('llm_tokens_used', sa.Integer),
        sa.Column('overall_verdict', sa.String(50)),
        sa.Column('scanned_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('scanned_by', sa.String(255)),
    )

    # ── 5. Marketplace plugins ─────────────────────────────────────────
    op.create_table(
        'marketplace_plugins',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('slug', sa.String(100), unique=True, nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('s3_bucket', sa.String(255), server_default='automatos-marketplace'),
        sa.Column('s3_path', sa.String(500)),
        sa.Column('description', sa.Text),
        sa.Column('long_description', sa.Text),
        sa.Column('category_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('plugin_categories.id'), nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.String), server_default='{}'),
        sa.Column('skills_count', sa.Integer, server_default='0'),
        sa.Column('commands_count', sa.Integer, server_default='0'),
        sa.Column('agents_count', sa.Integer, server_default='0'),
        sa.Column('hooks_count', sa.Integer, server_default='0'),
        sa.Column('source_type', sa.String(50)),
        sa.Column('source_url', sa.String(500)),
        sa.Column('source_repo', sa.String(500)),
        sa.Column('original_author', sa.String(255)),
        sa.Column('license', sa.String(100)),
        sa.Column('token_estimate', sa.Integer, server_default='0'),
        sa.Column('recommended_models', postgresql.ARRAY(sa.String), server_default='{}'),
        sa.Column('security_scan_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('plugin_security_scans.id'), nullable=True),
        sa.Column('security_status', sa.String(50), server_default="'pending'"),
        sa.Column('is_active', sa.Boolean, server_default='true'),
        sa.Column('is_featured', sa.Boolean, server_default='false'),
        sa.Column('approval_status', sa.String(50), server_default="'pending'"),
        sa.Column('approved_by', sa.String(255)),
        sa.Column('approved_at', sa.DateTime, nullable=True),
        sa.Column('rejection_reason', sa.Text),
        sa.Column('enable_count', sa.Integer, server_default='0'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
    )
    op.create_index('idx_marketplace_plugins_category', 'marketplace_plugins', ['category_id'])
    op.create_index('idx_marketplace_plugins_approval_active', 'marketplace_plugins', ['approval_status', 'is_active'])
    op.create_index('idx_marketplace_plugins_featured', 'marketplace_plugins', ['is_featured', 'enable_count'])

    # ── 6. Plugin sync history ─────────────────────────────────────────
    op.create_table(
        'plugin_sync_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('action', sa.String(50), nullable=False),
        sa.Column('plugin_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('marketplace_plugins.id'), nullable=True),
        sa.Column('plugin_slug', sa.String(100)),
        sa.Column('status', sa.String(50)),
        sa.Column('started_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('completed_at', sa.DateTime, nullable=True),
        sa.Column('details', postgresql.JSONB, server_default='{}'),
        sa.Column('error_message', sa.Text),
        sa.Column('performed_by', sa.String(255)),
    )

    # ── 7. Workspace enabled plugins (junction) ───────────────────────
    op.create_table(
        'workspace_enabled_plugins',
        sa.Column('workspace_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('plugin_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('marketplace_plugins.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('enabled_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('enabled_by', sa.Integer, sa.ForeignKey('users.id'), nullable=True),
    )
    op.create_index('idx_workspace_enabled_plugins_workspace', 'workspace_enabled_plugins', ['workspace_id'])
    op.create_index('idx_workspace_enabled_plugins_plugin', 'workspace_enabled_plugins', ['plugin_id'])

    # ── 8. Agent assigned plugins (junction) ──────────────────────────
    op.create_table(
        'agent_assigned_plugins',
        sa.Column('agent_id', sa.Integer, sa.ForeignKey('agents.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('plugin_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('marketplace_plugins.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('priority', sa.Integer, server_default='0'),
        sa.Column('assigned_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
    )
    op.create_index('idx_agent_assigned_plugins_agent', 'agent_assigned_plugins', ['agent_id'])
    op.create_index('idx_agent_assigned_plugins_plugin', 'agent_assigned_plugins', ['plugin_id'])


def downgrade():
    # Drop in reverse order (junction tables first, then dependents)
    op.drop_table('agent_assigned_plugins')
    op.drop_table('workspace_enabled_plugins')
    op.drop_table('plugin_sync_history')
    op.drop_table('marketplace_plugins')
    op.drop_table('plugin_security_scans')
    op.drop_table('plugin_categories')

    # Agent persona columns
    op.drop_index('idx_agents_persona_id', table_name='agents')
    op.drop_constraint('fk_agents_persona_id', 'agents', type_='foreignkey')
    op.drop_column('agents', 'use_custom_persona')
    op.drop_column('agents', 'custom_persona_prompt')
    op.drop_column('agents', 'persona_id')

    # Personas table
    op.drop_table('personas')
