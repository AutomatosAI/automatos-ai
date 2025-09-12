
-- Migration: Add Run Telemetry Tables for Pattern Mining
-- This migration adds the telemetry tables needed for agent execution tracking and pattern discovery

-- Create run_traces table for tracking agent executions
CREATE TABLE IF NOT EXISTS run_traces (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) UNIQUE NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    task_type VARCHAR(255) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    status VARCHAR(50) DEFAULT 'running',
    success BOOLEAN,
    input_data JSONB,
    output_data JSONB,
    context_data JSONB,
    metadata JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create run_events table for detailed step tracking
CREATE TABLE IF NOT EXISTS run_events (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    step_name VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    event_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create playbook_executions table for tracking playbook usage
CREATE TABLE IF NOT EXISTS playbook_executions (
    id SERIAL PRIMARY KEY,
    playbook_id INTEGER NOT NULL,
    run_id VARCHAR(255) NOT NULL,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN,
    duration_ms INTEGER,
    cost DECIMAL(10,6) DEFAULT 0.0,
    steps_completed INTEGER DEFAULT 0,
    steps_total INTEGER DEFAULT 0,
    efficiency_score DECIMAL(5,2),
    deviations JSONB,
    improvements JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_run_traces_run_id ON run_traces(run_id);
CREATE INDEX IF NOT EXISTS idx_run_traces_agent_id ON run_traces(agent_id);
CREATE INDEX IF NOT EXISTS idx_run_traces_task_type ON run_traces(task_type);
CREATE INDEX IF NOT EXISTS idx_run_traces_start_time ON run_traces(start_time);
CREATE INDEX IF NOT EXISTS idx_run_traces_success ON run_traces(success);
CREATE INDEX IF NOT EXISTS idx_run_traces_status ON run_traces(status);
CREATE INDEX IF NOT EXISTS idx_run_traces_agent_task_time ON run_traces(agent_id, task_type, start_time);
CREATE INDEX IF NOT EXISTS idx_run_traces_success_time ON run_traces(success, start_time);
CREATE INDEX IF NOT EXISTS idx_run_traces_task_success ON run_traces(task_type, success);

CREATE INDEX IF NOT EXISTS idx_run_events_run_id ON run_events(run_id);
CREATE INDEX IF NOT EXISTS idx_run_events_event_type ON run_events(event_type);
CREATE INDEX IF NOT EXISTS idx_run_events_step_name ON run_events(step_name);
CREATE INDEX IF NOT EXISTS idx_run_events_timestamp ON run_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_run_events_run_event_time ON run_events(run_id, event_type, timestamp);
CREATE INDEX IF NOT EXISTS idx_run_events_run_step_time ON run_events(run_id, step_name, timestamp);

CREATE INDEX IF NOT EXISTS idx_playbook_executions_playbook_id ON playbook_executions(playbook_id);
CREATE INDEX IF NOT EXISTS idx_playbook_executions_run_id ON playbook_executions(run_id);
CREATE INDEX IF NOT EXISTS idx_playbook_executions_executed_at ON playbook_executions(executed_at);
CREATE INDEX IF NOT EXISTS idx_playbook_executions_success ON playbook_executions(success);
CREATE INDEX IF NOT EXISTS idx_playbook_executions_playbook_success_time ON playbook_executions(playbook_id, success, executed_at);
CREATE INDEX IF NOT EXISTS idx_playbook_executions_playbook_performance ON playbook_executions(playbook_id, efficiency_score);

-- Add foreign key constraints if the referenced tables exist
-- Note: Uncomment these if you have corresponding tables
-- ALTER TABLE run_events ADD CONSTRAINT fk_run_events_run_id FOREIGN KEY (run_id) REFERENCES run_traces(run_id) ON DELETE CASCADE;
-- ALTER TABLE playbook_executions ADD CONSTRAINT fk_playbook_executions_run_id FOREIGN KEY (run_id) REFERENCES run_traces(run_id) ON DELETE CASCADE;

-- Create trigger for updating updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_run_traces_updated_at BEFORE UPDATE ON run_traces
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for testing (optional - remove in production)
-- INSERT INTO run_traces (run_id, agent_id, task_type, start_time, end_time, duration_ms, status, success, input_data, output_data, context_data, metadata)
-- VALUES 
--     ('test-run-001', 'agent-007', 'document_analysis', CURRENT_TIMESTAMP - INTERVAL '1 hour', CURRENT_TIMESTAMP - INTERVAL '59 minutes', 60000, 'completed', true, '{"document_id": "doc123"}', '{"analysis": "complete"}', '{"user_id": "user456"}', '{"cost": 0.05}'),
--     ('test-run-002', 'agent-008', 'text_generation', CURRENT_TIMESTAMP - INTERVAL '30 minutes', CURRENT_TIMESTAMP - INTERVAL '29 minutes', 45000, 'completed', true, '{"prompt": "Generate summary"}', '{"text": "Generated summary"}', '{"project": "legal"}', '{"cost": 0.03}');

-- Insert sample events for testing (optional - remove in production)
-- INSERT INTO run_events (run_id, event_type, step_name, timestamp, event_data)
-- VALUES 
--     ('test-run-001', 'run_start', NULL, CURRENT_TIMESTAMP - INTERVAL '1 hour', '{"agent_id": "agent-007", "task_type": "document_analysis"}'),
--     ('test-run-001', 'step_start', 'load_document', CURRENT_TIMESTAMP - INTERVAL '1 hour', '{}'),
--     ('test-run-001', 'step_end', 'load_document', CURRENT_TIMESTAMP - INTERVAL '59.5 minutes', '{"success": true}'),
--     ('test-run-001', 'tool_usage', NULL, CURRENT_TIMESTAMP - INTERVAL '59.2 minutes', '{"tool_name": "gpt-4", "cost": 0.05}'),
--     ('test-run-001', 'run_end', NULL, CURRENT_TIMESTAMP - INTERVAL '59 minutes', '{"success": true, "duration_ms": 60000}');
