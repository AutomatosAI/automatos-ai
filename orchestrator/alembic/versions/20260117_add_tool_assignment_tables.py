"""Add tenant_tool_config and agent_tool_assignments tables (PRD-35)

Revision ID: 20260117_tools
Revises: 20251006_models
Create Date: 2026-01-17 10:00:00

This migration adds:
- tenant_tool_config: Tool enablement per tenant with 1:1 credential mapping
- agent_tool_assignments: Agent-tool assignments (like skills, 1:many)

Part of PRD-35: Tool Catalog & Registry Architecture
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260117_tools'
down_revision = '20251006_models'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ===================================================================
    # CREATE TENANT_TOOL_CONFIG TABLE
    # ===================================================================
    # Tool enablement per tenant with 1:1 credential mapping
    # When a user enables a tool, they must provide credentials
    op.create_table('tenant_tool_config',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('adapter_tool_id', sa.String(length=255), nullable=False),
        sa.Column('adapter_tool_name', sa.String(length=255), nullable=False),
        sa.Column('enabled', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('credential_id', sa.Integer(), sa.ForeignKey('credentials.id', ondelete='SET NULL'), nullable=True),
        sa.Column('configuration', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id', name='tenant_tool_config_pkey'),
        sa.UniqueConstraint('tenant_id', 'adapter_tool_id', name='uq_tenant_tool_config')
    )
    
    # Create indexes for performance
    op.create_index('idx_tenant_tool_config_tenant', 'tenant_tool_config', ['tenant_id'], unique=False)
    op.create_index('idx_tenant_tool_config_tool', 'tenant_tool_config', ['adapter_tool_id'], unique=False)
    op.create_index('idx_tenant_tool_config_enabled', 'tenant_tool_config', ['tenant_id', 'enabled'], unique=False)
    
    # Add comments
    op.execute("COMMENT ON TABLE tenant_tool_config IS 'Tool enablement per tenant with 1:1 credential mapping (PRD-35)'")
    op.execute("COMMENT ON COLUMN tenant_tool_config.tenant_id IS 'UUID of the tenant organization'")
    op.execute("COMMENT ON COLUMN tenant_tool_config.tool_id IS 'Reference to tool name in Unified Adapter'")
    op.execute("COMMENT ON COLUMN tenant_tool_config.adapter_tool_name IS 'Cached display name from Adapter'")
    op.execute("COMMENT ON COLUMN tenant_tool_config.credential_id IS '1:1 credential mapping - set when tool is enabled'")
    op.execute("COMMENT ON COLUMN tenant_tool_config.configuration IS 'Tenant-specific tool configuration overrides'")
    
    # ===================================================================
    # CREATE AGENT_TOOL_ASSIGNMENTS TABLE
    # ===================================================================
    # Agent-tool assignments (like skills, 1:many)
    op.create_table('agent_tool_assignments_v2',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('agent_id', sa.Integer(), sa.ForeignKey('agents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('adapter_tool_id', sa.String(length=255), nullable=False),
        sa.Column('enabled', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id', name='agent_tool_assignments_v2_pkey'),
        sa.UniqueConstraint('agent_id', 'adapter_tool_id', name='uq_agent_tool_assignment_v2')
    )
    
    # Create indexes for performance
    op.create_index('idx_agent_tool_assignments_v2_agent', 'agent_tool_assignments_v2', ['agent_id'], unique=False)
    op.create_index('idx_agent_tool_assignments_v2_tool', 'agent_tool_assignments_v2', ['adapter_tool_id'], unique=False)
    op.create_index('idx_agent_tool_assignments_v2_enabled', 'agent_tool_assignments_v2', ['agent_id', 'enabled'], unique=False)
    
    # Add comments
    op.execute("COMMENT ON TABLE agent_tool_assignments_v2 IS 'Agent-tool assignments, similar to skills (PRD-35)'")
    op.execute("COMMENT ON COLUMN agent_tool_assignments_v2.agent_id IS 'Reference to agents table'")
    op.execute("COMMENT ON COLUMN agent_tool_assignments_v2.tool_id IS 'Reference to tool name in Unified Adapter'")
    op.execute("COMMENT ON COLUMN agent_tool_assignments_v2.enabled IS 'Whether this tool is enabled for this agent'")
    
    # ===================================================================
    # ADD TENANT_ID TO AGENTS TABLE (if not exists)
    # ===================================================================
    # Check if tenant_id column already exists on agents
    conn = op.get_bind()
    result = conn.execute(sa.text("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'tenant_id'
    """))
    if result.fetchone() is None:
        op.add_column('agents',
            sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True)
        )
        op.create_index('idx_agents_tenant', 'agents', ['tenant_id'], unique=False)
        op.execute("COMMENT ON COLUMN agents.tenant_id IS 'UUID of the tenant organization this agent belongs to'")
        print("   - Added tenant_id column to agents table")
    
    print("✅ Tool assignment tables migration completed successfully!")
    print("   - Created tenant_tool_config table")
    print("   - Created agent_tool_assignments_v2 table")


def downgrade() -> None:
    # Drop agent_tool_assignments_v2 table
    op.drop_index('idx_agent_tool_assignments_v2_enabled', table_name='agent_tool_assignments_v2')
    op.drop_index('idx_agent_tool_assignments_v2_tool', table_name='agent_tool_assignments_v2')
    op.drop_index('idx_agent_tool_assignments_v2_agent', table_name='agent_tool_assignments_v2')
    op.drop_table('agent_tool_assignments_v2')
    
    # Drop tenant_tool_config table
    op.drop_index('idx_tenant_tool_config_enabled', table_name='tenant_tool_config')
    op.drop_index('idx_tenant_tool_config_tool', table_name='tenant_tool_config')
    op.drop_index('idx_tenant_tool_config_tenant', table_name='tenant_tool_config')
    op.drop_table('tenant_tool_config')
    
    # Note: Not removing tenant_id from agents as it may be used elsewhere
    
    print("✅ Tool assignment tables migration rolled back successfully!")
