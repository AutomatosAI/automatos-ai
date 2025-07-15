
-- Initialize database for Enhanced Orchestrator
-- This script creates the necessary tables and indexes

-- Create database if it doesn't exist (PostgreSQL)
-- Note: This is handled by docker-compose environment variables

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create audit logging table
CREATE TABLE IF NOT EXISTS security_events (
    event_id VARCHAR(255) PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    security_level VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    user_id VARCHAR(255),
    source_ip INET,
    resource TEXT,
    action VARCHAR(255),
    result VARCHAR(100),
    details JSONB,
    risk_score INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create threat indicators table
CREATE TABLE IF NOT EXISTS threat_indicators (
    id SERIAL PRIMARY KEY,
    indicator_type VARCHAR(100) NOT NULL,
    value TEXT NOT NULL,
    severity VARCHAR(50) NOT NULL,
    description TEXT,
    first_seen TIMESTAMP WITH TIME ZONE NOT NULL,
    last_seen TIMESTAMP WITH TIME ZONE NOT NULL,
    count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(indicator_type, value)
);

-- Create command executions table
CREATE TABLE IF NOT EXISTS command_executions (
    id VARCHAR(255) PRIMARY KEY,
    command TEXT NOT NULL,
    user_id VARCHAR(255),
    source_ip INET,
    execution_time FLOAT,
    exit_code INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    success BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create workflows table
CREATE TABLE IF NOT EXISTS workflows (
    workflow_id VARCHAR(255) PRIMARY KEY,
    workflow_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    repository_url TEXT NOT NULL,
    target_host VARCHAR(255),
    project_path TEXT,
    config JSONB,
    task_prompt TEXT,
    environment_variables JSONB,
    deployment_logs JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_events_user ON security_events(user_id);
CREATE INDEX IF NOT EXISTS idx_security_events_ip ON security_events(source_ip);
CREATE INDEX IF NOT EXISTS idx_security_events_type ON security_events(event_type);
CREATE INDEX IF NOT EXISTS idx_security_events_level ON security_events(security_level);

CREATE INDEX IF NOT EXISTS idx_threat_indicators_type ON threat_indicators(indicator_type);
CREATE INDEX IF NOT EXISTS idx_threat_indicators_severity ON threat_indicators(severity);
CREATE INDEX IF NOT EXISTS idx_threat_indicators_last_seen ON threat_indicators(last_seen);

CREATE INDEX IF NOT EXISTS idx_command_executions_timestamp ON command_executions(timestamp);
CREATE INDEX IF NOT EXISTS idx_command_executions_user ON command_executions(user_id);
CREATE INDEX IF NOT EXISTS idx_command_executions_ip ON command_executions(source_ip);

CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflows_type ON workflows(workflow_type);
CREATE INDEX IF NOT EXISTS idx_workflows_created ON workflows(created_at);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for workflows table
CREATE TRIGGER update_workflows_updated_at 
    BEFORE UPDATE ON workflows 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Insert initial data
INSERT INTO security_events (
    event_id, event_type, security_level, timestamp, user_id, 
    source_ip, resource, action, result, details
) VALUES (
    'init-001', 'SYSTEM_ERROR', 'low', NOW(), 'system', 
    '127.0.0.1', 'database', 'initialization', 'success',
    '{"message": "Database initialized successfully"}'
) ON CONFLICT (event_id) DO NOTHING;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

COMMIT;
