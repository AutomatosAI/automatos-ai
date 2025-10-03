-- ============================================
-- MIGRATION 004: Agent Tools Integration
-- Phase 3: Skills & Tools
-- ============================================
-- Creates tables for MCP tool management and agent-tool assignments

-- 1. Create MCP Tools Table
CREATE TABLE IF NOT EXISTS mcp_tools (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    mcp_server_url VARCHAR(500),
    capabilities JSONB DEFAULT '{}',
    credentials_schema JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active',
    provider VARCHAR(255),
    version VARCHAR(50),
    icon VARCHAR(100),
    category VARCHAR(100),
    tags TEXT[],
    metadata JSONB DEFAULT '{}',
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 2. Create Agent-Tool Assignment Table (Many-to-Many with metadata)
CREATE TABLE IF NOT EXISTS agent_tool_assignments (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    tool_id INTEGER NOT NULL REFERENCES mcp_tools(id) ON DELETE CASCADE,
    enabled BOOLEAN DEFAULT true,
    permissions JSONB DEFAULT '{}',
    configuration JSONB DEFAULT '{}',
    assigned_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(agent_id, tool_id)
);

-- 3. Create Tool Credentials Table (Encrypted Storage)
CREATE TABLE IF NOT EXISTS tool_credentials (
    id SERIAL PRIMARY KEY,
    tool_id INTEGER NOT NULL REFERENCES mcp_tools(id) ON DELETE CASCADE,
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    environment VARCHAR(50) NOT NULL,
    credentials JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tool_id, agent_id, environment)
);

-- 4. Create Tool Usage Logs (Tracking & Analytics)
CREATE TABLE IF NOT EXISTS tool_usage_logs (
    id SERIAL PRIMARY KEY,
    execution_id INTEGER REFERENCES workflow_executions(id),
    agent_id INTEGER NOT NULL REFERENCES agents(id),
    tool_id INTEGER NOT NULL REFERENCES mcp_tools(id),
    method_called VARCHAR(255),
    input_data JSONB,
    output_data JSONB,
    success BOOLEAN,
    execution_time_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_agent_tools_agent ON agent_tool_assignments(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_tools_tool ON agent_tool_assignments(tool_id);
CREATE INDEX IF NOT EXISTS idx_agent_tools_enabled ON agent_tool_assignments(enabled);
CREATE INDEX IF NOT EXISTS idx_tool_credentials_tool ON tool_credentials(tool_id);
CREATE INDEX IF NOT EXISTS idx_tool_credentials_agent ON tool_credentials(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_credentials_env ON tool_credentials(environment);
CREATE INDEX IF NOT EXISTS idx_tool_usage_agent ON tool_usage_logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_tool ON tool_usage_logs(tool_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_execution ON tool_usage_logs(execution_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_success ON tool_usage_logs(success);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_status ON mcp_tools(status);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_category ON mcp_tools(category);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_provider ON mcp_tools(provider);

-- Trigger for updated_at on agent_tool_assignments
DROP TRIGGER IF EXISTS update_agent_tool_assignments_updated_at ON agent_tool_assignments;
CREATE TRIGGER update_agent_tool_assignments_updated_at 
    BEFORE UPDATE ON agent_tool_assignments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger for updated_at on tool_credentials
DROP TRIGGER IF EXISTS update_tool_credentials_updated_at ON tool_credentials;
CREATE TRIGGER update_tool_credentials_updated_at 
    BEFORE UPDATE ON tool_credentials
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger for updated_at on mcp_tools
DROP TRIGGER IF EXISTS update_mcp_tools_updated_at ON mcp_tools;
CREATE TRIGGER update_mcp_tools_updated_at 
    BEFORE UPDATE ON mcp_tools
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert Sample MCP Tools
INSERT INTO mcp_tools (name, description, mcp_server_url, status, provider, category, icon, capabilities, credentials_schema, metadata)
VALUES 
    (
        'GitHub MCP',
        'GitHub repository management via Model Context Protocol',
        'http://localhost:3000/mcp/github',
        'active',
        'GitHub',
        'developer',
        '⚡',
        '{"methods": ["repos.list", "repos.create", "issues.list", "issues.create", "pull_requests.list"]}'::jsonb,
        '{"required": ["api_token"], "optional": ["webhook_url"]}'::jsonb,
        '{"version": "1.0.0", "documentation": "https://github.com/mcp-docs"}'::jsonb
    ),
    (
        'Slack MCP',
        'Slack messaging and notifications via Model Context Protocol',
        'http://localhost:3000/mcp/slack',
        'active',
        'Slack Technologies',
        'communication',
        '💬',
        '{"methods": ["chat.postMessage", "channels.list", "users.list"]}'::jsonb,
        '{"required": ["bot_token"], "optional": ["webhook_url"]}'::jsonb,
        '{"version": "1.0.0", "rate_limit": "100/minute"}'::jsonb
    ),
    (
        'AWS S3 MCP',
        'AWS S3 file storage via Model Context Protocol',
        'http://localhost:3000/mcp/aws-s3',
        'active',
        'Amazon Web Services',
        'cloud',
        '☁️',
        '{"methods": ["s3.listBuckets", "s3.putObject", "s3.getObject", "s3.deleteObject"]}'::jsonb,
        '{"required": ["access_key", "secret_key", "region"]}'::jsonb,
        '{"version": "1.0.0", "regions": ["us-east-1", "eu-west-1"]}'::jsonb
    ),
    (
        'Docker MCP',
        'Docker container management via Model Context Protocol',
        'http://localhost:3000/mcp/docker',
        'active',
        'Docker Inc.',
        'cloud',
        '🐳',
        '{"methods": ["containers.list", "containers.start", "containers.stop", "images.list"]}'::jsonb,
        '{"required": ["registry_token"], "optional": ["registry_url"]}'::jsonb,
        '{"version": "1.0.0"}'::jsonb
    ),
    (
        'Jira MCP',
        'Jira project management via Model Context Protocol',
        'http://localhost:3000/mcp/jira',
        'active',
        'Atlassian',
        'developer',
        '📋',
        '{"methods": ["issues.list", "issues.create", "issues.update", "projects.list"]}'::jsonb,
        '{"required": ["api_token", "domain"]}'::jsonb,
        '{"version": "1.0.0"}'::jsonb
    )
ON CONFLICT (name) DO NOTHING;

-- Comments
COMMENT ON TABLE mcp_tools IS 'MCP (Model Context Protocol) tools that agents can use';
COMMENT ON TABLE agent_tool_assignments IS 'Many-to-many relationship between agents and tools with permissions';
COMMENT ON TABLE tool_credentials IS 'Encrypted credentials for tools per agent and environment';
COMMENT ON TABLE tool_usage_logs IS 'Logs of all tool executions for analytics and debugging';

COMMENT ON COLUMN mcp_tools.capabilities IS 'JSON object describing available MCP methods/tools';
COMMENT ON COLUMN mcp_tools.credentials_schema IS 'JSON schema defining required and optional credentials';
COMMENT ON COLUMN agent_tool_assignments.permissions IS 'JSON object with read, write, execute permissions';
COMMENT ON COLUMN agent_tool_assignments.configuration IS 'Tool-specific configuration for this agent';
COMMENT ON COLUMN tool_credentials.credentials IS 'Encrypted credential data (should be encrypted at application level)';

