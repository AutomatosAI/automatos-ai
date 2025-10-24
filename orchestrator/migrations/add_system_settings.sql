-- Migration: Add System Settings Table
-- Description: Creates system_settings table for managing system-wide configuration
-- Date: 2025-10-18

-- Create system_settings table
CREATE TABLE IF NOT EXISTS system_settings (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    key VARCHAR(100) NOT NULL,
    value TEXT,
    value_type VARCHAR(20) DEFAULT 'string',
    description TEXT,
    is_sensitive BOOLEAN DEFAULT FALSE,
    is_required BOOLEAN DEFAULT FALSE,
    default_value TEXT,
    validation_rules JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100) DEFAULT 'system'
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_system_settings_category ON system_settings(category);
CREATE INDEX IF NOT EXISTS idx_system_settings_key ON system_settings(key);
CREATE INDEX IF NOT EXISTS idx_system_settings_category_key ON system_settings(category, key);

-- Create unique constraint to prevent duplicate settings
CREATE UNIQUE INDEX IF NOT EXISTS idx_system_settings_unique ON system_settings(category, key);

-- Add updated_at trigger
CREATE OR REPLACE FUNCTION update_system_settings_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_system_settings_updated_at
    BEFORE UPDATE ON system_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_system_settings_updated_at();

-- Insert some initial system settings
INSERT INTO system_settings (category, key, value, value_type, description, is_required, default_value) VALUES
-- General Settings
('general', 'environment', 'development', 'string', 'Application environment (development, staging, production)', true, 'development'),
('general', 'log_level', 'INFO', 'string', 'System logging level', false, 'INFO'),

-- Orchestrator LLM Settings
('orchestrator_llm', 'llm_provider', 'openai', 'string', 'Default LLM provider for orchestrator', true, 'openai'),
('orchestrator_llm', 'llm_model', 'gpt-4', 'string', 'Default LLM model', true, 'gpt-4'),
('orchestrator_llm', 'llm_temperature', '0.7', 'number', 'LLM temperature setting', false, '0.7'),
('orchestrator_llm', 'llm_max_tokens', '2000', 'number', 'Maximum tokens for LLM responses', false, '2000'),

-- CodeGraph Settings
('codegraph', 'enabled', 'true', 'boolean', 'Enable CodeGraph analysis', false, 'true'),
('codegraph', 'max_file_size', '1000000', 'number', 'Maximum file size for analysis (bytes)', false, '1000000'),
('codegraph', 'supported_languages', 'python,typescript,javascript,go,rust', 'string', 'Supported programming languages', false, 'python,typescript,javascript,go,rust'),
('codegraph', 'cache_ttl', '3600', 'number', 'Cache TTL in seconds', false, '3600'),

-- API Rate Limiting Settings
('api_rate_limiting', 'requests_per_window', '100', 'number', 'Maximum requests per time window', false, '100'),
('api_rate_limiting', 'window_seconds', '60', 'number', 'Rate limiting time window in seconds', false, '60'),

-- Backend API Keys
('backend_api_keys', 'api_key', 'test_api_key_for_backend_validation_2025', 'string', 'Backend API authentication key', true, 'test_api_key_for_backend_validation_2025'),
('backend_api_keys', 'api_port', '8000', 'number', 'Backend API port', true, '8000'),
('backend_api_keys', 'api_url', 'api.automatos.app', 'string', 'Backend API URL', true, 'api.automatos.app'),

-- ML/AI Model Configuration
('general', 'embedding_model', 'disabled', 'string', 'Embedding model configuration', false, 'disabled'),
('general', 'embedding_cache_dir', './model_cache', 'string', 'Directory for embedding model cache', false, './model_cache'),
('general', 'embedding_max_seq_length', '256', 'number', 'Maximum sequence length for embeddings', false, '256'),
('general', 'openai_embedding_model', 'text-embedding-3-small', 'string', 'OpenAI embedding model', false, 'text-embedding-3-small'),
('general', 'vector_store_type', 'faiss', 'string', 'Vector store type', false, 'faiss'),
('general', 'vector_store_dimensions', '384', 'number', 'Vector store dimensions', false, '384'),
('general', 'vector_store_index_type', 'IndexFlatIP', 'string', 'Vector store index type', false, 'IndexFlatIP'),
('general', 'chunk_size', '512', 'number', 'Text chunk size for processing', false, '512'),
('general', 'chunk_overlap', '50', 'number', 'Text chunk overlap', false, '50'),
('general', 'max_context_length', '4000', 'number', 'Maximum context length', false, '4000'),

-- SSH Deployment Settings
('general', 'deploy_host', 'mcp.automatos.app', 'string', 'Deployment host', false, 'mcp.automatos.app'),
('general', 'deploy_port', '22', 'number', 'Deployment SSH port', false, '22'),
('general', 'deploy_user', 'root', 'string', 'Deployment SSH user', false, 'root'),
('general', 'deploy_key_path', '/aRpp/keys/deploy_key', 'string', 'Deployment SSH key path', false, '/aRpp/keys/deploy_key'),
('general', 'deploy_enabled', 'false', 'boolean', 'Enable deployment', false, 'false'),

-- NextJS Frontend Settings
('general', 'nextauth_secret', 'your-nextauth-secret-here', 'string', 'NextAuth secret key', false, 'your-nextauth-secret-here'),
('general', 'nextauth_url', 'https://8018adcb5.preview.abacusai.app', 'string', 'NextAuth URL', false, 'https://8018adcb5.preview.abacusai.app'),
('general', 'next_public_api_url', 'https://api.automatos.app', 'string', 'Public API URL for frontend', false, 'https://api.automatos.app')

ON CONFLICT (category, key) DO UPDATE SET
    value = EXCLUDED.value,
    description = EXCLUDED.description,
    is_required = EXCLUDED.is_required,
    default_value = EXCLUDED.default_value,
    updated_at = NOW();

-- Mark sensitive fields
UPDATE system_settings SET is_sensitive = true WHERE key = 'api_key' AND category = 'backend_api_keys';
UPDATE system_settings SET is_sensitive = true WHERE key = 'nextauth_secret' AND category = 'general';

COMMIT;
