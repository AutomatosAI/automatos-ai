-- =====================================================
-- AUTOMATOS AI - COMPLETE DATABASE SCHEMA
-- Version: 1.0.0
-- Description: Single source of truth for all tables
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- =====================================================
-- CORE TABLES (Existing)
-- =====================================================

-- Agents table
CREATE TABLE IF NOT EXISTS agents (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    agent_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    configuration JSONB,
    performance_metrics JSONB,
    priority_level VARCHAR(50) DEFAULT 'medium',
    max_concurrent_tasks INTEGER DEFAULT 5,
    auto_start BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- Skills table
CREATE TABLE IF NOT EXISTS skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    skill_type VARCHAR(100) NOT NULL,
    category VARCHAR(100) NOT NULL,
    implementation TEXT,
    parameters JSONB,
    performance_data JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- Patterns table
CREATE TABLE IF NOT EXISTS patterns (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    pattern_type VARCHAR(100) NOT NULL,
    pattern_data JSONB,
    usage_count INTEGER DEFAULT 0,
    effectiveness_score FLOAT DEFAULT 0.0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- Workflows table
CREATE TABLE IF NOT EXISTS workflows (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    workflow_definition JSONB,
    status VARCHAR(50) DEFAULT 'draft',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- Workflow executions table
CREATE TABLE IF NOT EXISTS workflow_executions (
    id SERIAL PRIMARY KEY,
    workflow_id INTEGER REFERENCES workflows(id),
    agent_id INTEGER REFERENCES agents(id),
    status VARCHAR(50) DEFAULT 'pending',
    input_data JSONB,
    output_data JSONB,
    execution_log TEXT,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    error_message TEXT
);

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255),
    file_type VARCHAR(100),
    file_size INTEGER,
    file_path VARCHAR(500),
    content_hash VARCHAR(255),
    status VARCHAR(50) DEFAULT 'uploaded',
    chunk_count INTEGER DEFAULT 0,
    tags JSONB,
    description TEXT,
    doc_metadata JSONB,
    upload_date TIMESTAMP DEFAULT NOW(),
    processed_date TIMESTAMP,
    created_by VARCHAR(255)
);

-- System configurations table
CREATE TABLE IF NOT EXISTS system_configurations (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSONB,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    updated_by VARCHAR(255)
);

-- RAG configurations table
CREATE TABLE IF NOT EXISTS rag_configurations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    embedding_model VARCHAR(255),
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 200,
    retrieval_strategy VARCHAR(100) DEFAULT 'similarity',
    top_k INTEGER DEFAULT 5,
    similarity_threshold FLOAT DEFAULT 0.7,
    configuration JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- =====================================================
-- ASSOCIATION TABLES
-- =====================================================

CREATE TABLE IF NOT EXISTS agent_skills (
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    skill_id INTEGER REFERENCES skills(id) ON DELETE CASCADE,
    PRIMARY KEY (agent_id, skill_id)
);

CREATE TABLE IF NOT EXISTS workflow_agents (
    workflow_id INTEGER REFERENCES workflows(id) ON DELETE CASCADE,
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    PRIMARY KEY (workflow_id, agent_id)
);

-- =====================================================
-- ORCHESTRATION TABLES (PRD-01)
-- =====================================================

CREATE TABLE IF NOT EXISTS task_decompositions (
    id SERIAL PRIMARY KEY,
    parent_task_id INTEGER,
    subtask_id INTEGER,
    dependency_type VARCHAR(50),
    execution_order INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS task_assignments (
    id SERIAL PRIMARY KEY,
    task_id INTEGER,
    agent_id INTEGER REFERENCES agents(id),
    assignment_score FLOAT,
    assignment_reason TEXT,
    status VARCHAR(50),
    assigned_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS execution_contexts (
    id SERIAL PRIMARY KEY,
    workflow_id INTEGER REFERENCES workflows(id),
    context_data JSONB,
    field_state JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- AGENT RUNTIME TABLES (PRD-02)
-- =====================================================

CREATE TABLE IF NOT EXISTS agent_runtimes (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    llm_provider VARCHAR(50),
    model_name VARCHAR(100),
    temperature FLOAT,
    max_tokens INTEGER,
    context_window INTEGER,
    api_key_ref VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_tools (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    tool_id INTEGER,
    configuration JSONB,
    access_level VARCHAR(50),
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_performance (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    task_id INTEGER,
    execution_time FLOAT,
    token_usage JSONB,
    quality_score FLOAT,
    error_count INTEGER,
    success BOOLEAN,
    recorded_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- CONTEXT ENGINEERING TABLES (PRD-03)
-- =====================================================

CREATE TABLE IF NOT EXISTS context_templates (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    template_type VARCHAR(50),
    template_text TEXT,
    parameters JSONB,
    usage_count INTEGER DEFAULT 0,
    success_rate FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS context_examples (
    id SERIAL PRIMARY KEY,
    category VARCHAR(100),
    input_text TEXT,
    output_text TEXT,
    embedding vector(1536),
    metadata JSONB,
    quality_score FLOAT,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS context_patterns (
    id SERIAL PRIMARY KEY,
    pattern_name VARCHAR(255),
    pattern_type VARCHAR(50),
    pattern_structure JSONB,
    applicable_tasks JSONB,
    effectiveness_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS context_optimizations (
    id SERIAL PRIMARY KEY,
    task_id INTEGER,
    original_tokens INTEGER,
    optimized_tokens INTEGER,
    information_gain FLOAT,
    optimization_strategy VARCHAR(100),
    metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- COMMUNICATION TABLES (PRD-04)
-- =====================================================

CREATE TABLE IF NOT EXISTS agent_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_agent_id INTEGER REFERENCES agents(id),
    to_agent_id INTEGER REFERENCES agents(id),
    message_type VARCHAR(50),
    content JSONB,
    priority INTEGER,
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    delivered_at TIMESTAMP,
    read_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS shared_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255),
    team_id UUID,
    context_data JSONB,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS context_permissions (
    id SERIAL PRIMARY KEY,
    context_id UUID REFERENCES shared_contexts(id),
    agent_id INTEGER REFERENCES agents(id),
    permission_level VARCHAR(50),
    granted_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS collaboration_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_id INTEGER,
    team_agents INTEGER[],
    strategy VARCHAR(50),
    shared_context_id UUID REFERENCES shared_contexts(id),
    status VARCHAR(50),
    result JSONB,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- =====================================================
-- MEMORY TABLES (PRD-05)
-- =====================================================

CREATE TABLE IF NOT EXISTS memory_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id INTEGER REFERENCES agents(id),
    content TEXT,
    memory_type VARCHAR(50),
    memory_level VARCHAR(50),
    importance FLOAT,
    embedding vector(1536),
    access_count INTEGER DEFAULT 0,
    last_access TIMESTAMP,
    decay_rate FLOAT DEFAULT 0.1,
    associations UUID[],
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id INTEGER REFERENCES agents(id),
    concept VARCHAR(255),
    description TEXT,
    node_type VARCHAR(50),
    embedding vector(1536),
    importance FLOAT,
    confidence FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_node_id UUID REFERENCES knowledge_nodes(id),
    to_node_id UUID REFERENCES knowledge_nodes(id),
    relationship VARCHAR(100),
    strength FLOAT,
    evidence_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS learning_outcomes (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    task_id INTEGER,
    learned_pattern TEXT,
    success_rate FLOAT,
    confidence FLOAT,
    application_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- MONITORING TABLES (PRD-06)
-- =====================================================

CREATE TABLE IF NOT EXISTS dashboard_configs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    config_name VARCHAR(255),
    layout JSONB,
    widgets JSONB,
    refresh_rate INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analytics_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_type VARCHAR(50),
    metrics JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS alert_configs (
    id SERIAL PRIMARY KEY,
    metric_type VARCHAR(100),
    threshold_value FLOAT,
    comparison_operator VARCHAR(10),
    alert_channel VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS custom_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255),
    calculation_query TEXT,
    visualization_type VARCHAR(50),
    refresh_interval INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(agent_type);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(status);
CREATE INDEX IF NOT EXISTS idx_memory_items_agent ON memory_items(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_items_embedding ON memory_items USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_context_examples_embedding ON context_examples USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_embedding ON knowledge_nodes USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_agent_messages_to ON agent_messages(to_agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_messages_from ON agent_messages(from_agent_id);

-- =====================================================
-- DEFAULT DATA
-- =====================================================

-- System configurations
INSERT INTO system_configurations (config_key, config_value, description) 
VALUES 
    ('default_llm_provider', '{"provider": "openai", "model": "gpt-4"}', 'Default LLM configuration'),
    ('max_context_window', '{"tokens": 8192}', 'Maximum context window size'),
    ('memory_retention_days', '{"days": 30}', 'Default memory retention period')
ON CONFLICT (config_key) DO NOTHING;

-- Default skills
INSERT INTO skills (name, description, skill_type, category) 
VALUES 
    ('code_analysis', 'Analyze and review code', 'technical', 'development'),
    ('task_decomposition', 'Break down complex tasks', 'cognitive', 'planning'),
    ('pattern_recognition', 'Identify patterns in data', 'cognitive', 'analytics')
ON CONFLICT DO NOTHING;

-- =====================================================
-- FUNCTIONS
-- =====================================================

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update timestamp trigger to tables
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_skills_updated_at BEFORE UPDATE ON skills
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_workflows_updated_at BEFORE UPDATE ON workflows
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- PERMISSIONS
-- =====================================================

-- Grant necessary permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO automatos_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO automatos_user;

-- =====================================================
-- VERSION TRACKING
-- =====================================================

CREATE TABLE IF NOT EXISTS schema_versions (
    version VARCHAR(20) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT NOW(),
    description TEXT
);

INSERT INTO schema_versions (version, description) 
VALUES ('1.0.0', 'Complete initial schema with all PRD tables')
ON CONFLICT DO NOTHING;
