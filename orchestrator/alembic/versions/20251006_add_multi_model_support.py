"""Add multi-model support for agents (PRD-15)

Revision ID: 20251006_models
Revises: 20250930_tracking
Create Date: 2025-10-06 14:00:00

This migration adds comprehensive multi-model support including:
- llm_models table for model registry
- model_config and model_usage_stats fields to agents table
- models_used field to workflow_executions table
- Seed data for OpenAI and Anthropic models
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from datetime import datetime

# revision identifiers, used by Alembic.
revision = '20251006_models'
down_revision = '128a785a7681'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ===================================================================
    # CREATE LLM_MODELS TABLE
    # ===================================================================
    op.create_table('llm_models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('provider', sa.String(length=50), nullable=False),
        sa.Column('model_id', sa.String(length=255), nullable=False),
        sa.Column('display_name', sa.String(length=255), nullable=False),
        sa.Column('model_family', sa.String(length=100), nullable=True),
        
        # Capabilities
        sa.Column('capabilities', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=True),
        sa.Column('context_window', sa.Integer(), nullable=False),
        sa.Column('max_output_tokens', sa.Integer(), nullable=False),
        sa.Column('supports_functions', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('supports_vision', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('supports_streaming', sa.Boolean(), server_default='true', nullable=True),
        
        # Cost information
        sa.Column('input_cost_per_1k_tokens', sa.Float(), nullable=True),
        sa.Column('output_cost_per_1k_tokens', sa.Float(), nullable=True),
        
        # Metadata
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('release_date', sa.DateTime(), nullable=True),
        sa.Column('deprecation_date', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(length=50), server_default='active', nullable=True),
        sa.Column('recommended_for', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=True),
        
        # Settings
        sa.Column('default_temperature', sa.Float(), server_default='0.7', nullable=True),
        sa.Column('min_temperature', sa.Float(), server_default='0.0', nullable=True),
        sa.Column('max_temperature', sa.Float(), server_default='2.0', nullable=True),
        
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        
        sa.PrimaryKeyConstraint('id', name='llm_models_pkey'),
        sa.UniqueConstraint('model_id', name='llm_models_model_id_key')
    )
    
    # Create indexes for performance
    op.create_index('idx_llm_models_provider', 'llm_models', ['provider'], unique=False)
    op.create_index('idx_llm_models_status', 'llm_models', ['status'], unique=False)
    op.create_index('idx_llm_models_model_family', 'llm_models', ['model_family'], unique=False)
    
    # Add comments
    op.execute("COMMENT ON TABLE llm_models IS 'Registry of available LLM models from different providers'")
    op.execute("COMMENT ON COLUMN llm_models.provider IS 'LLM provider: openai, anthropic, huggingface'")
    op.execute("COMMENT ON COLUMN llm_models.model_id IS 'Unique model identifier used in API calls'")
    op.execute("COMMENT ON COLUMN llm_models.capabilities IS 'JSON of model capabilities: reasoning, coding, analysis, etc.'")
    op.execute("COMMENT ON COLUMN llm_models.recommended_for IS 'JSON array of recommended use cases'")
    
    # ===================================================================
    # UPDATE AGENTS TABLE
    # ===================================================================
    op.add_column('agents', 
        sa.Column('model_config', postgresql.JSONB(astext_type=sa.Text()), 
                  server_default=sa.text("""'{"provider": "openai", "model_id": "gpt-4", "temperature": 0.7, "max_tokens": 2000, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0, "fallback_model_id": null}'::jsonb"""),
                  nullable=True)
    )
    
    op.add_column('agents',
        sa.Column('model_usage_stats', postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("""'{"total_tokens": 0, "total_cost": 0.0, "total_requests": 0, "avg_tokens_per_request": 0, "last_used_at": null}'::jsonb"""),
                  nullable=True)
    )
    
    # Create GIN index for efficient JSON queries
    op.create_index('idx_agents_model_config', 'agents', ['model_config'], unique=False, postgresql_using='gin')
    
    # Add comments
    op.execute("COMMENT ON COLUMN agents.model_config IS 'Agent-specific model configuration including provider, model_id, and parameters'")
    op.execute("COMMENT ON COLUMN agents.model_usage_stats IS 'Real-time tracking of model usage, tokens, and costs'")
    
    # ===================================================================
    # UPDATE WORKFLOW_EXECUTIONS TABLE
    # ===================================================================
    op.add_column('workflow_executions',
        sa.Column('models_used', postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'[]'::jsonb"),
                  nullable=True)
    )
    
    # Create GIN index for JSON queries
    op.create_index('idx_workflow_executions_models', 'workflow_executions', ['models_used'], unique=False, postgresql_using='gin')
    
    # Add comment
    op.execute("COMMENT ON COLUMN workflow_executions.models_used IS 'Array of models used in this execution with tokens and costs'")
    
    # ===================================================================
    # SEED DATA - INSERT DEFAULT MODELS
    # ===================================================================
    
    # Helper to format JSON for PostgreSQL
    def json_val(data):
        import json
        return json.dumps(data).replace("'", "''")
    
    # OpenAI Models
    op.execute(f"""
        INSERT INTO llm_models (
            provider, model_id, display_name, model_family,
            capabilities, context_window, max_output_tokens,
            input_cost_per_1k_tokens, output_cost_per_1k_tokens,
            supports_functions, supports_vision, supports_streaming,
            recommended_for, status, description
        ) VALUES
        -- GPT-4 Turbo
        ('openai', 'gpt-4-turbo-preview', 'GPT-4 Turbo', 'gpt-4',
         '{json_val({"reasoning": "excellent", "coding": "excellent", "analysis": "excellent", "speed": "fast"})}',
         128000, 4096, 0.01, 0.03, true, false, true,
         '{json_val(["code_analysis", "complex_reasoning", "system_design", "architecture"])}',
         'active', 'Latest GPT-4 model with improved performance and 128K context window'),
        
        -- GPT-4
        ('openai', 'gpt-4', 'GPT-4', 'gpt-4',
         '{json_val({"reasoning": "excellent", "coding": "excellent", "analysis": "excellent"})}',
         8192, 4096, 0.03, 0.06, true, false, true,
         '{json_val(["code_review", "security_audit", "architecture", "complex_tasks"])}',
         'active', 'Most capable GPT-4 model for complex reasoning and coding tasks'),
        
        -- GPT-3.5 Turbo
        ('openai', 'gpt-3.5-turbo', 'GPT-3.5 Turbo', 'gpt-3.5',
         '{json_val({"reasoning": "good", "coding": "good", "speed": "very fast", "cost": "low"})}',
         16385, 4096, 0.0005, 0.0015, true, false, true,
         '{json_val(["simple_tasks", "data_processing", "quick_responses", "high_volume"])}',
         'active', 'Fast and cost-effective model for simpler tasks and high-volume operations');
    """)
    
    # Anthropic Models
    op.execute(f"""
        INSERT INTO llm_models (
            provider, model_id, display_name, model_family,
            capabilities, context_window, max_output_tokens,
            input_cost_per_1k_tokens, output_cost_per_1k_tokens,
            supports_functions, supports_vision, supports_streaming,
            recommended_for, status, description
        ) VALUES
        -- Claude 3 Opus
        ('anthropic', 'claude-3-opus-20240229', 'Claude 3 Opus', 'claude-3',
         '{json_val({"reasoning": "excellent", "analysis": "excellent", "creativity": "excellent", "context": "very large"})}',
         200000, 4096, 0.015, 0.075, false, false, true,
         '{json_val(["complex_analysis", "research", "planning", "creative_writing"])}',
         'active', 'Most powerful Claude model with superior reasoning and 200K context window'),
        
        -- Claude 3 Sonnet
        ('anthropic', 'claude-3-sonnet-20240229', 'Claude 3 Sonnet', 'claude-3',
         '{json_val({"reasoning": "excellent", "balance": "optimal", "speed": "fast", "cost": "moderate"})}',
         200000, 4096, 0.003, 0.015, false, false, true,
         '{json_val(["balanced_tasks", "general_purpose", "workflows", "agent_coordination"])}',
         'active', 'Balanced model offering excellent performance at moderate cost'),
        
        -- Claude 3 Haiku
        ('anthropic', 'claude-3-haiku-20240307', 'Claude 3 Haiku', 'claude-3',
         '{json_val({"speed": "fastest", "cost": "lowest", "reasoning": "good", "efficiency": "excellent"})}',
         200000, 4096, 0.00025, 0.00125, false, false, true,
         '{json_val(["high_volume", "simple_tasks", "cost_sensitive", "real_time"])}',
         'active', 'Fastest and most cost-effective Claude model for high-volume operations');
    """)
    
    print("✅ Multi-model support migration completed successfully!")
    print("   - Created llm_models table with 6 models")
    print("   - Added model_config and model_usage_stats to agents")
    print("   - Added models_used to workflow_executions")
    print("   - Seeded OpenAI and Anthropic models")


def downgrade() -> None:
    # Remove workflow_executions changes
    op.drop_index('idx_workflow_executions_models', table_name='workflow_executions')
    op.drop_column('workflow_executions', 'models_used')
    
    # Remove agents table changes
    op.drop_index('idx_agents_model_config', table_name='agents')
    op.drop_column('agents', 'model_usage_stats')
    op.drop_column('agents', 'model_config')
    
    # Drop llm_models table
    op.drop_index('idx_llm_models_model_family', table_name='llm_models')
    op.drop_index('idx_llm_models_status', table_name='llm_models')
    op.drop_index('idx_llm_models_provider', table_name='llm_models')
    op.drop_table('llm_models')
    
    print("✅ Multi-model support migration rolled back successfully!")

