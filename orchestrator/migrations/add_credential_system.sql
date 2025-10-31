-- PRD-18: n8n-Style Credential Management System
-- Migration to add credential management tables
-- Run with: psql -U postgres -d orchestrator_db -f add_credential_system.sql

BEGIN;

-- ============================================================================
-- 1. Credential Types Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS credential_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    icon VARCHAR(50),
    description TEXT,
    schema_definition JSON NOT NULL,
    test_endpoint JSON,
    documentation_url VARCHAR(500),
    is_system BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_credential_types_name ON credential_types(name);
CREATE INDEX idx_credential_types_category ON credential_types(category);
CREATE INDEX idx_credential_types_active ON credential_types(is_active);

COMMENT ON TABLE credential_types IS 'Credential type definitions (like n8n). Defines schema for credential types.';
COMMENT ON COLUMN credential_types.schema_definition IS 'JSON array of field definitions for dynamic form generation';
COMMENT ON COLUMN credential_types.test_endpoint IS 'Configuration for testing credential validity';

-- ============================================================================
-- 2. Credentials Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS credentials (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    credential_type_id INTEGER NOT NULL REFERENCES credential_types(id) ON DELETE CASCADE,
    encrypted_data TEXT NOT NULL,
    environment VARCHAR(50) DEFAULT 'production',
    description TEXT,
    tags JSON DEFAULT '[]',
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP,
    last_tested TIMESTAMP,
    test_status VARCHAR(50),
    test_message TEXT,
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_credential_name_env UNIQUE(name, environment)
);

CREATE INDEX idx_credentials_type ON credentials(credential_type_id);
CREATE INDEX idx_credentials_env ON credentials(environment);
CREATE INDEX idx_credentials_active ON credentials(is_active);
CREATE INDEX idx_credentials_name ON credentials(name);

COMMENT ON TABLE credentials IS 'Stored credentials with encrypted values';
COMMENT ON COLUMN credentials.encrypted_data IS 'Fernet-encrypted JSON blob of credential field values';
COMMENT ON COLUMN credentials.environment IS 'Target environment: development, staging, production';

-- ============================================================================
-- 3. Credential Audit Logs Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS credential_audit_logs (
    id SERIAL PRIMARY KEY,
    credential_id INTEGER REFERENCES credentials(id) ON DELETE CASCADE,
    action VARCHAR(50) NOT NULL,
    user_id VARCHAR(255),
    ip_address VARCHAR(45),
    success BOOLEAN,
    error_message TEXT,
    metadata JSON,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_credential_audit_credential ON credential_audit_logs(credential_id);
CREATE INDEX idx_credential_audit_action ON credential_audit_logs(action);
CREATE INDEX idx_credential_audit_created ON credential_audit_logs(created_at);
CREATE INDEX idx_credential_audit_user ON credential_audit_logs(user_id);

COMMENT ON TABLE credential_audit_logs IS 'Audit trail for all credential operations';
COMMENT ON COLUMN credential_audit_logs.action IS 'Action type: created, updated, deleted, accessed, tested';

-- ============================================================================
-- 4. Update AgentToolAssignment to link credentials
-- ============================================================================

-- Add credential_id column to existing table
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agent_tool_assignments' 
        AND column_name = 'credential_id'
    ) THEN
        ALTER TABLE agent_tool_assignments 
        ADD COLUMN credential_id INTEGER REFERENCES credentials(id) ON DELETE SET NULL;
        
        CREATE INDEX idx_agent_tool_credential ON agent_tool_assignments(credential_id);
        
        COMMENT ON COLUMN agent_tool_assignments.credential_id IS 'Linked credential for n8n-style credential injection';
    END IF;
END $$;

-- ============================================================================
-- 5. Seed System Credential Types
-- ============================================================================

-- PostgreSQL Credential Type
INSERT INTO credential_types (name, display_name, category, icon, description, schema_definition, test_endpoint, is_system, is_active)
VALUES (
    'postgres_credentials',
    'PostgreSQL',
    'database',
    'database',
    'PostgreSQL database connection credentials',
    '[
        {
            "displayName": "Host",
            "name": "host",
            "type": "string",
            "required": true,
            "default": "localhost",
            "description": "Database host address"
        },
        {
            "displayName": "Port",
            "name": "port",
            "type": "number",
            "required": true,
            "default": 5432,
            "description": "Database port"
        },
        {
            "displayName": "Database",
            "name": "database",
            "type": "string",
            "required": true,
            "description": "Database name"
        },
        {
            "displayName": "User",
            "name": "user",
            "type": "string",
            "required": true,
            "description": "Database user"
        },
        {
            "displayName": "Password",
            "name": "password",
            "type": "password",
            "required": true,
            "description": "Database password",
            "typeOptions": {"password": true}
        },
        {
            "displayName": "SSL Mode",
            "name": "ssl_mode",
            "type": "options",
            "required": false,
            "default": "prefer",
            "description": "SSL connection mode",
            "typeOptions": {
                "options": [
                    {"name": "Disable", "value": "disable"},
                    {"name": "Prefer", "value": "prefer"},
                    {"name": "Require", "value": "require"}
                ]
            }
        }
    ]'::json,
    '{"method": "connect", "description": "Tests PostgreSQL database connection"}'::json,
    true,
    true
) ON CONFLICT (name) DO NOTHING;

-- Redis Credential Type
INSERT INTO credential_types (name, display_name, category, icon, description, schema_definition, test_endpoint, is_system, is_active)
VALUES (
    'redis_credentials',
    'Redis',
    'database',
    'database',
    'Redis connection credentials',
    '[
        {
            "displayName": "Host",
            "name": "host",
            "type": "string",
            "required": true,
            "default": "localhost",
            "description": "Redis host address"
        },
        {
            "displayName": "Port",
            "name": "port",
            "type": "number",
            "required": true,
            "default": 6379,
            "description": "Redis port"
        },
        {
            "displayName": "Password",
            "name": "password",
            "type": "password",
            "required": false,
            "description": "Redis password (if required)",
            "typeOptions": {"password": true}
        },
        {
            "displayName": "Database",
            "name": "database",
            "type": "number",
            "required": false,
            "default": 0,
            "description": "Redis database number (0-15)"
        }
    ]'::json,
    '{"method": "connect", "description": "Tests Redis connection with PING command"}'::json,
    true,
    true
) ON CONFLICT (name) DO NOTHING;

-- OpenAI Credential Type
INSERT INTO credential_types (name, display_name, category, icon, description, schema_definition, test_endpoint, is_system, is_active)
VALUES (
    'openai_api',
    'OpenAI API',
    'ai',
    'brain',
    'OpenAI API credentials for GPT models',
    '[
        {
            "displayName": "API Key",
            "name": "api_key",
            "type": "password",
            "required": true,
            "description": "Your OpenAI API key (sk-...)",
            "typeOptions": {"password": true},
            "placeholder": "sk-..."
        },
        {
            "displayName": "Organization ID",
            "name": "organization_id",
            "type": "string",
            "required": false,
            "description": "Optional organization ID for team accounts"
        },
        {
            "displayName": "Base URL",
            "name": "base_url",
            "type": "string",
            "required": false,
            "default": "https://api.openai.com/v1",
            "description": "Custom API endpoint (for proxies or Azure OpenAI)"
        }
    ]'::json,
    '{"method": "api_call", "url": "{base_url}/models", "headers": {"Authorization": "Bearer {api_key}"}, "description": "Lists available models"}'::json,
    true,
    true
) ON CONFLICT (name) DO NOTHING;

-- Anthropic (Claude) Credential Type
INSERT INTO credential_types (name, display_name, category, icon, description, schema_definition, test_endpoint, is_system, is_active)
VALUES (
    'anthropic_api',
    'Anthropic API',
    'ai',
    'brain',
    'Anthropic Claude API credentials',
    '[
        {
            "displayName": "API Key",
            "name": "api_key",
            "type": "password",
            "required": true,
            "description": "Your Anthropic API key (sk-ant-...)",
            "typeOptions": {"password": true},
            "placeholder": "sk-ant-..."
        },
        {
            "displayName": "Base URL",
            "name": "base_url",
            "type": "string",
            "required": false,
            "default": "https://api.anthropic.com",
            "description": "Custom API endpoint (for proxies)"
        }
    ]'::json,
    '{"method": "api_call", "url": "{base_url}/v1/messages", "headers": {"x-api-key": "{api_key}", "anthropic-version": "2023-06-01"}, "description": "Tests API access"}'::json,
    true,
    true
) ON CONFLICT (name) DO NOTHING;

-- GitHub Credential Type
INSERT INTO credential_types (name, display_name, category, icon, description, schema_definition, test_endpoint, is_system, is_active)
VALUES (
    'github_api',
    'GitHub API',
    'code',
    'github',
    'GitHub API credentials for repository access',
    '[
        {
            "displayName": "Access Token",
            "name": "access_token",
            "type": "password",
            "required": true,
            "description": "GitHub Personal Access Token (ghp_...)",
            "typeOptions": {"password": true},
            "placeholder": "ghp_..."
        },
        {
            "displayName": "Token Type",
            "name": "token_type",
            "type": "options",
            "required": false,
            "default": "personal",
            "description": "Type of GitHub token",
            "typeOptions": {
                "options": [
                    {"name": "Personal Access Token", "value": "personal"},
                    {"name": "OAuth App Token", "value": "oauth"},
                    {"name": "GitHub App Token", "value": "app"}
                ]
            }
        }
    ]'::json,
    '{"method": "api_call", "url": "https://api.github.com/user", "headers": {"Authorization": "Bearer {access_token}"}, "description": "Validates token and retrieves user info"}'::json,
    true,
    true
) ON CONFLICT (name) DO NOTHING;

-- SSH Credential Type
INSERT INTO credential_types (name, display_name, category, icon, description, schema_definition, test_endpoint, is_system, is_active)
VALUES (
    'ssh_credentials',
    'SSH',
    'infrastructure',
    'server',
    'SSH connection credentials for remote server access',
    '[
        {
            "displayName": "Host",
            "name": "host",
            "type": "string",
            "required": true,
            "description": "SSH server hostname or IP"
        },
        {
            "displayName": "Port",
            "name": "port",
            "type": "number",
            "required": true,
            "default": 22,
            "description": "SSH port"
        },
        {
            "displayName": "Username",
            "name": "username",
            "type": "string",
            "required": true,
            "description": "SSH username"
        },
        {
            "displayName": "Authentication Method",
            "name": "auth_method",
            "type": "options",
            "required": true,
            "default": "password",
            "typeOptions": {
                "options": [
                    {"name": "Password", "value": "password"},
                    {"name": "Private Key", "value": "key"}
                ]
            }
        },
        {
            "displayName": "Password",
            "name": "password",
            "type": "password",
            "required": false,
            "description": "SSH password",
            "typeOptions": {"password": true},
            "displayOptions": {
                "show": {"auth_method": ["password"]}
            }
        },
        {
            "displayName": "Private Key",
            "name": "private_key",
            "type": "password",
            "required": false,
            "description": "SSH private key content",
            "typeOptions": {"password": true, "rows": 8},
            "displayOptions": {
                "show": {"auth_method": ["key"]}
            }
        },
        {
            "displayName": "Passphrase",
            "name": "passphrase",
            "type": "password",
            "required": false,
            "description": "Private key passphrase (if encrypted)",
            "typeOptions": {"password": true},
            "displayOptions": {
                "show": {"auth_method": ["key"]}
            }
        }
    ]'::json,
    '{"method": "connect", "description": "Tests SSH connection and authentication"}'::json,
    true,
    true
) ON CONFLICT (name) DO NOTHING;

-- Generic API Credential Type
INSERT INTO credential_types (name, display_name, category, icon, description, schema_definition, test_endpoint, is_system, is_active)
VALUES (
    'generic_api',
    'Generic API',
    'api',
    'key',
    'Generic API credentials with customizable authentication',
    '[
        {
            "displayName": "API Key",
            "name": "api_key",
            "type": "password",
            "required": true,
            "description": "API authentication key",
            "typeOptions": {"password": true}
        },
        {
            "displayName": "API Secret",
            "name": "api_secret",
            "type": "password",
            "required": false,
            "description": "Optional API secret for HMAC signing",
            "typeOptions": {"password": true}
        },
        {
            "displayName": "Base URL",
            "name": "base_url",
            "type": "string",
            "required": false,
            "description": "API base URL"
        },
        {
            "displayName": "Authentication Method",
            "name": "auth_method",
            "type": "options",
            "required": true,
            "default": "header",
            "typeOptions": {
                "options": [
                    {"name": "Header", "value": "header"},
                    {"name": "Query Parameter", "value": "query"},
                    {"name": "Bearer Token", "value": "bearer"}
                ]
            }
        },
        {
            "displayName": "Header Name",
            "name": "header_name",
            "type": "string",
            "required": false,
            "default": "X-API-Key",
            "displayOptions": {
                "show": {"auth_method": ["header"]}
            }
        }
    ]'::json,
    null,
    true,
    true
) ON CONFLICT (name) DO NOTHING;

-- OAuth2 Token Credential Type (Basic - stores token only)
INSERT INTO credential_types (name, display_name, category, icon, description, schema_definition, test_endpoint, is_system, is_active)
VALUES (
    'oauth2_token',
    'OAuth2 Token',
    'api',
    'key',
    'OAuth2 access token storage (token must be obtained externally)',
    '[
        {
            "displayName": "Access Token",
            "name": "access_token",
            "type": "password",
            "required": true,
            "description": "OAuth2 access token",
            "typeOptions": {"password": true}
        },
        {
            "displayName": "Refresh Token",
            "name": "refresh_token",
            "type": "password",
            "required": false,
            "description": "OAuth2 refresh token (optional)",
            "typeOptions": {"password": true}
        },
        {
            "displayName": "Token Expiry",
            "name": "expires_at",
            "type": "string",
            "required": false,
            "description": "Token expiration timestamp (ISO format)"
        },
        {
            "displayName": "Scope",
            "name": "scope",
            "type": "string",
            "required": false,
            "description": "OAuth2 token scope"
        }
    ]'::json,
    null,
    true,
    true
) ON CONFLICT (name) DO NOTHING;

COMMIT;

-- Display summary
DO $$
DECLARE
    type_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO type_count FROM credential_types;
    RAISE NOTICE '✅ Credential system migration complete!';
    RAISE NOTICE '   - credential_types table created';
    RAISE NOTICE '   - credentials table created';
    RAISE NOTICE '   - credential_audit_logs table created';
    RAISE NOTICE '   - agent_tool_assignments.credential_id column added';
    RAISE NOTICE '   - % credential types seeded', type_count;
    RAISE NOTICE '';
    RAISE NOTICE '🔐 Next steps:';
    RAISE NOTICE '   1. Create credentials via API or UI';
    RAISE NOTICE '   2. Link credentials to tool assignments';
    RAISE NOTICE '   3. Update services to use credential resolver';
END $$;

