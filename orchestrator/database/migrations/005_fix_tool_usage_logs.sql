-- ============================================
-- MIGRATION 005: Fix tool_usage_logs Schema
-- ============================================
-- Updates tool_usage_logs table to match expected schema

-- First, backup existing data
CREATE TABLE IF NOT EXISTS tool_usage_logs_backup AS 
SELECT * FROM tool_usage_logs;

-- Drop the existing table and recreate with correct schema
DROP TABLE IF EXISTS tool_usage_logs CASCADE;

-- Recreate with correct schema matching migration 004
CREATE TABLE tool_usage_logs (
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

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_tool_usage_agent ON tool_usage_logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_tool ON tool_usage_logs(tool_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_execution ON tool_usage_logs(execution_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_success ON tool_usage_logs(success);

-- Comments
COMMENT ON TABLE tool_usage_logs IS 'Logs of all tool executions for analytics and debugging';
COMMENT ON COLUMN tool_usage_logs.execution_id IS 'Reference to workflow execution (optional)';
COMMENT ON COLUMN tool_usage_logs.method_called IS 'Name of the MCP method/tool that was called';
COMMENT ON COLUMN tool_usage_logs.input_data IS 'Input parameters sent to the tool';
COMMENT ON COLUMN tool_usage_logs.output_data IS 'Output/response data from the tool';
COMMENT ON COLUMN tool_usage_logs.execution_time_ms IS 'Time taken to execute the tool in milliseconds';

