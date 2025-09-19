-- Tool Tables Creation Script
-- Creates all missing Tool-related tables for Automatos AI

-- Tools table
CREATE TABLE IF NOT EXISTS tools (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(100) NOT NULL,
    provider VARCHAR(255) NOT NULL,
    version VARCHAR(100) NOT NULL,
    icon VARCHAR(50),
    pricing VARCHAR(100),
    rating REAL DEFAULT 0.0,
    tags TEXT[],
    status VARCHAR(50) DEFAULT 'available',
    is_installed BOOLEAN DEFAULT FALSE,
    is_configured BOOLEAN DEFAULT FALSE,
    usage_count INTEGER DEFAULT 0,
    permissions TEXT[],
    required_credentials TEXT[],
    supported_environments TEXT[],
    mcp_config JSON,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

-- Tool Credentials table
CREATE TABLE IF NOT EXISTS tool_credentials (
    id SERIAL PRIMARY KEY,
    tool_id INTEGER NOT NULL REFERENCES tools(id),
    credential_key VARCHAR(100) NOT NULL,
    credential_value TEXT NOT NULL,
    environment VARCHAR(50) NOT NULL,
    description TEXT,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Tool Configurations table
CREATE TABLE IF NOT EXISTS tool_configurations (
    id SERIAL PRIMARY KEY,
    tool_id INTEGER NOT NULL REFERENCES tools(id),
    environment VARCHAR(50) NOT NULL,
    configuration JSON DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    last_health_check TIMESTAMP,
    health_status VARCHAR(50) DEFAULT 'unknown',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Agent Tool Permissions table
CREATE TABLE IF NOT EXISTS agent_tool_permissions (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER NOT NULL REFERENCES agents(id),
    tool_id INTEGER NOT NULL REFERENCES tools(id),
    environment VARCHAR(50) NOT NULL,
    permissions TEXT[],
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Tool Usage Logs table
CREATE TABLE IF NOT EXISTS tool_usage_logs (
    id SERIAL PRIMARY KEY,
    tool_id INTEGER NOT NULL REFERENCES tools(id),
    action VARCHAR(100) NOT NULL,
    agent_id INTEGER REFERENCES agents(id),
    environment VARCHAR(50),
    response_time_ms INTEGER,
    success BOOLEAN,
    error_message TEXT,
    details JSON DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Credential Audit Logs table
CREATE TABLE IF NOT EXISTS credential_audit_logs (
    id SERIAL PRIMARY KEY,
    tool_id INTEGER NOT NULL REFERENCES tools(id),
    credential_key VARCHAR(100) NOT NULL,
    action VARCHAR(100) NOT NULL,
    environment VARCHAR(50),
    user_id VARCHAR(255),
    details JSON DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Permission Audit Logs table
CREATE TABLE IF NOT EXISTS permission_audit_logs (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER NOT NULL REFERENCES agents(id),
    tool_id INTEGER NOT NULL REFERENCES tools(id),
    action VARCHAR(100) NOT NULL,
    environment VARCHAR(50),
    user_id VARCHAR(255),
    details JSON DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_tools_name ON tools(name);
CREATE INDEX IF NOT EXISTS idx_tools_category ON tools(category);
CREATE INDEX IF NOT EXISTS idx_tools_status ON tools(status);
CREATE INDEX IF NOT EXISTS idx_tool_credentials_tool_id ON tool_credentials(tool_id);
CREATE INDEX IF NOT EXISTS idx_tool_credentials_environment ON tool_credentials(environment);
CREATE INDEX IF NOT EXISTS idx_tool_configurations_tool_id ON tool_configurations(tool_id);
CREATE INDEX IF NOT EXISTS idx_agent_tool_permissions_agent_id ON agent_tool_permissions(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_tool_permissions_tool_id ON agent_tool_permissions(tool_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_logs_tool_id ON tool_usage_logs(tool_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_logs_timestamp ON tool_usage_logs(timestamp);

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Tool tables created successfully';
END $$;
