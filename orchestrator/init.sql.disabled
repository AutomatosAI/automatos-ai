
-- Initialize Automotas AI Database
-- This script sets up the basic database structure

-- Create database if it doesn't exist (handled by Docker)
-- CREATE DATABASE IF NOT EXISTS orchestrator_db;

-- Use the database
-- \c orchestrator_db;

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create workflows table
CREATE TABLE IF NOT EXISTS workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255) UNIQUE NOT NULL,
    workflow_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    repository_url TEXT,
    target_host VARCHAR(255),
    project_path TEXT,
    task_prompt TEXT,
    config JSONB,
    environment_variables JSONB,
    deployment_logs JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create workflow_steps table for detailed step tracking
CREATE TABLE IF NOT EXISTS workflow_steps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255) NOT NULL,
    step_id VARCHAR(255) NOT NULL,
    step_name VARCHAR(255) NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_seconds DECIMAL(10,3),
    progress_percentage DECIMAL(5,2) DEFAULT 0.0,
    metadata JSONB,
    error_details TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
);

-- Create agent_communications table
CREATE TABLE IF NOT EXISTS agent_communications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255),
    message_id VARCHAR(255) UNIQUE NOT NULL,
    from_agent VARCHAR(255) NOT NULL,
    to_agent VARCHAR(255) NOT NULL,
    message_type VARCHAR(50) NOT NULL,
    priority INTEGER DEFAULT 2,
    content JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    requires_response BOOLEAN DEFAULT FALSE,
    correlation_id VARCHAR(255),
    metadata JSONB,
    FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
);

-- Create performance_metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255),
    operation_name VARCHAR(255) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_seconds DECIMAL(10,3),
    cpu_usage_percent DECIMAL(5,2),
    memory_usage_mb DECIMAL(10,2),
    tokens_used INTEGER DEFAULT 0,
    estimated_cost_usd DECIMAL(10,6) DEFAULT 0.0,
    success BOOLEAN DEFAULT TRUE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
);

-- Create agent_status table
CREATE TABLE IF NOT EXISTS agent_status (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255),
    agent_id VARCHAR(255) NOT NULL,
    agent_type VARCHAR(100) NOT NULL,
    current_task TEXT,
    status VARCHAR(50) NOT NULL,
    progress_percentage DECIMAL(5,2) DEFAULT 0.0,
    start_time TIMESTAMP WITH TIME ZONE,
    last_update TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
);

-- Create code_generation_logs table
CREATE TABLE IF NOT EXISTS code_generation_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255),
    file_path VARCHAR(500) NOT NULL,
    status VARCHAR(50) NOT NULL,
    lines_generated INTEGER DEFAULT 0,
    file_size_bytes INTEGER DEFAULT 0,
    content_type VARCHAR(100),
    quality_score DECIMAL(4,2),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
);

-- Create git_operations table
CREATE TABLE IF NOT EXISTS git_operations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255),
    operation VARCHAR(100) NOT NULL,
    repository_path TEXT,
    files JSONB,
    commit_message TEXT,
    branch VARCHAR(255),
    success BOOLEAN DEFAULT TRUE,
    operation_results JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_workflows_workflow_id ON workflows(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflows_created_at ON workflows(created_at);

CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow_id ON workflow_steps(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_status ON workflow_steps(status);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_agent_id ON workflow_steps(agent_id);

CREATE INDEX IF NOT EXISTS idx_agent_communications_workflow_id ON agent_communications(workflow_id);
CREATE INDEX IF NOT EXISTS idx_agent_communications_from_agent ON agent_communications(from_agent);
CREATE INDEX IF NOT EXISTS idx_agent_communications_to_agent ON agent_communications(to_agent);
CREATE INDEX IF NOT EXISTS idx_agent_communications_timestamp ON agent_communications(timestamp);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_workflow_id ON performance_metrics(workflow_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_operation_name ON performance_metrics(operation_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_start_time ON performance_metrics(start_time);

CREATE INDEX IF NOT EXISTS idx_agent_status_workflow_id ON agent_status(workflow_id);
CREATE INDEX IF NOT EXISTS idx_agent_status_agent_id ON agent_status(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_status_last_update ON agent_status(last_update);

CREATE INDEX IF NOT EXISTS idx_code_generation_logs_workflow_id ON code_generation_logs(workflow_id);
CREATE INDEX IF NOT EXISTS idx_code_generation_logs_file_path ON code_generation_logs(file_path);

CREATE INDEX IF NOT EXISTS idx_git_operations_workflow_id ON git_operations(workflow_id);
CREATE INDEX IF NOT EXISTS idx_git_operations_operation ON git_operations(operation);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_workflows_updated_at 
    BEFORE UPDATE ON workflows 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Insert initial data (optional)
-- This can be used for testing or default configurations

-- Create a view for workflow summaries
CREATE OR REPLACE VIEW workflow_summaries AS
SELECT 
    w.workflow_id,
    w.workflow_type,
    w.status,
    w.created_at,
    w.updated_at,
    COUNT(DISTINCT ws.id) as total_steps,
    COUNT(DISTINCT CASE WHEN ws.status = 'completed' THEN ws.id END) as completed_steps,
    COUNT(DISTINCT ast.agent_id) as total_agents,
    COALESCE(SUM(pm.tokens_used), 0) as total_tokens,
    COALESCE(SUM(pm.estimated_cost_usd), 0.0) as total_cost,
    COALESCE(AVG(pm.duration_seconds), 0.0) as avg_operation_duration
FROM workflows w
LEFT JOIN workflow_steps ws ON w.workflow_id = ws.workflow_id
LEFT JOIN agent_status ast ON w.workflow_id = ast.workflow_id
LEFT JOIN performance_metrics pm ON w.workflow_id = pm.workflow_id
GROUP BY w.workflow_id, w.workflow_type, w.status, w.created_at, w.updated_at;

-- Grant permissions (adjust as needed for your security requirements)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO orchestrator_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO orchestrator_user;

-- Log successful initialization
INSERT INTO workflows (workflow_id, workflow_type, status, task_prompt, created_at) 
VALUES ('init_db_001', 'system', 'completed', 'Database initialization completed successfully', CURRENT_TIMESTAMP)
ON CONFLICT (workflow_id) DO NOTHING;

-- Display initialization summary
SELECT 
    'Database initialization completed' as message,
    COUNT(*) as tables_created
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN (
    'workflows', 'workflow_steps', 'agent_communications', 
    'performance_metrics', 'agent_status', 'code_generation_logs', 
    'git_operations'
);
