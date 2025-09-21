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

-- =====================================================
-- MEMORY & KNOWLEDGE SYSTEM TABLES (PRD-05)
-- =====================================================

-- Enhanced memory items table with vector embeddings
CREATE TABLE IF NOT EXISTS memory_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    memory_type VARCHAR(50) NOT NULL CHECK (memory_type IN ('experience', 'knowledge', 'skill', 'pattern', 'feedback')),
    memory_level VARCHAR(50) NOT NULL DEFAULT 'short_term' CHECK (memory_level IN ('working', 'short_term', 'long_term', 'collective')),
    importance FLOAT DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    embedding vector(384),  -- OpenAI text-embedding-3-small dimension
    access_count INTEGER DEFAULT 0,
    last_access TIMESTAMP DEFAULT NOW(),
    decay_rate FLOAT DEFAULT 0.1,
    associations UUID[],  -- Array of related memory IDs
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    -- Performance tracking fields
    success_rate FLOAT DEFAULT 0.0,
    usage_in_solutions INTEGER DEFAULT 0,
    average_retrieval_time FLOAT DEFAULT 0.0
);

-- Create indexes for efficient retrieval
CREATE INDEX IF NOT EXISTS idx_memory_items_agent_id ON memory_items(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_items_memory_level ON memory_items(memory_level);
CREATE INDEX IF NOT EXISTS idx_memory_items_memory_type ON memory_items(memory_type);
CREATE INDEX IF NOT EXISTS idx_memory_items_importance ON memory_items(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memory_items_created_at ON memory_items(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_items_embedding ON memory_items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Knowledge graph nodes
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,  -- NULL for collective knowledge
    concept VARCHAR(255) NOT NULL,
    description TEXT,
    node_type VARCHAR(50) NOT NULL,
    embedding vector(384),
    importance FLOAT DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    confidence FLOAT DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for knowledge nodes
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_agent_id ON knowledge_nodes(agent_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_concept ON knowledge_nodes(concept);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_node_type ON knowledge_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_embedding ON knowledge_nodes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Knowledge graph edges
CREATE TABLE IF NOT EXISTS knowledge_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_node_id UUID NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    to_node_id UUID NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    relationship VARCHAR(100) NOT NULL,
    strength FLOAT DEFAULT 0.5 CHECK (strength >= 0 AND strength <= 1),
    evidence_count INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(from_node_id, to_node_id, relationship)
);

-- Create indexes for knowledge edges
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_from_node ON knowledge_edges(from_node_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_to_node ON knowledge_edges(to_node_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_relationship ON knowledge_edges(relationship);

-- Learning outcomes tracking
CREATE TABLE IF NOT EXISTS learning_outcomes (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    task_type VARCHAR(255),
    learned_pattern TEXT,
    success_rate_before FLOAT,
    success_rate_after FLOAT,
    execution_time_before FLOAT,
    execution_time_after FLOAT,
    confidence FLOAT DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
    application_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for learning outcomes
CREATE INDEX IF NOT EXISTS idx_learning_outcomes_agent_id ON learning_outcomes(agent_id);
CREATE INDEX IF NOT EXISTS idx_learning_outcomes_task_type ON learning_outcomes(task_type);
CREATE INDEX IF NOT EXISTS idx_learning_outcomes_created_at ON learning_outcomes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_learning_outcomes_confidence ON learning_outcomes(confidence DESC);

-- =====================================================
-- MEMORY SYSTEM HELPER FUNCTIONS
-- =====================================================

-- Function to calculate cosine similarity between vectors
CREATE OR REPLACE FUNCTION cosine_similarity(a vector(384), b vector(384))
RETURNS FLOAT AS $$
BEGIN
    RETURN 1 - (a <=> b);  -- pgvector's <=> operator returns cosine distance
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to get memory statistics for an agent
CREATE OR REPLACE FUNCTION get_memory_stats(p_agent_id INTEGER)
RETURNS TABLE (
    memory_level VARCHAR(50),
    count BIGINT,
    avg_importance FLOAT,
    avg_access_count FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.memory_level,
        COUNT(*) as count,
        AVG(m.importance) as avg_importance,
        AVG(m.access_count::FLOAT) as avg_access_count
    FROM memory_items m
    WHERE m.agent_id = p_agent_id
    GROUP BY m.memory_level;
END;
$$ LANGUAGE plpgsql;

-- Function to prune old working memories (should be called periodically)
CREATE OR REPLACE FUNCTION prune_old_memories()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM memory_items
    WHERE memory_level = 'working'
    AND created_at < NOW() - INTERVAL '5 minutes';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Initial system-level knowledge nodes for collective learning
INSERT INTO knowledge_nodes (agent_id, concept, node_type, description, importance, confidence)
VALUES 
    (NULL, 'task_decomposition', 'system_pattern', 'Breaking complex tasks into manageable subtasks', 0.9, 0.95),
    (NULL, 'parallel_execution', 'system_pattern', 'Executing independent tasks simultaneously', 0.85, 0.9),
    (NULL, 'error_recovery', 'system_pattern', 'Strategies for recovering from failures', 0.95, 0.85),
    (NULL, 'context_optimization', 'system_pattern', 'Optimizing prompt context for better results', 0.8, 0.9)
ON CONFLICT DO NOTHING;

-- Create initial knowledge relationships
INSERT INTO knowledge_edges (from_node_id, to_node_id, relationship, strength)
SELECT 
    (SELECT id FROM knowledge_nodes WHERE concept = 'task_decomposition'),
    (SELECT id FROM knowledge_nodes WHERE concept = 'parallel_execution'),
    'enables',
    0.9
WHERE EXISTS (SELECT 1 FROM knowledge_nodes WHERE concept = 'task_decomposition')
  AND EXISTS (SELECT 1 FROM knowledge_nodes WHERE concept = 'parallel_execution')
ON CONFLICT DO NOTHING;

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

INSERT INTO schema_versions (version, description) 
VALUES ('1.1.0', 'Added PRD-05 Memory & Knowledge System tables')
ON CONFLICT DO NOTHING;

-- =====================================================
-- PRD-05 MEMORY SYSTEM FIXES
-- Applied during initial setup to prevent issues
-- =====================================================

-- Fix 1: Add missing columns to learning_outcomes table
ALTER TABLE learning_outcomes ADD COLUMN IF NOT EXISTS task_type VARCHAR(100);
ALTER TABLE learning_outcomes ADD COLUMN IF NOT EXISTS success_rate_before FLOAT;
ALTER TABLE learning_outcomes ADD COLUMN IF NOT EXISTS success_rate_after FLOAT;
ALTER TABLE learning_outcomes ADD COLUMN IF NOT EXISTS execution_time_before FLOAT;
ALTER TABLE learning_outcomes ADD COLUMN IF NOT EXISTS execution_time_after FLOAT;

-- Fix 2: Update vector dimensions from 1536 to 384 for OpenAI text-embedding-3-small
-- Only alter if the column exists and has wrong dimensions
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'knowledge_nodes' AND column_name = 'embedding') THEN
        ALTER TABLE knowledge_nodes ALTER COLUMN embedding TYPE vector(384);
    END IF;
END $$;

-- Fix 3: Add embedding column to memory_items table (384 dimensions for OpenAI)
ALTER TABLE memory_items ADD COLUMN IF NOT EXISTS embedding vector(384);
ALTER TABLE memory_items ADD COLUMN IF NOT EXISTS memory_level VARCHAR(50) DEFAULT 'working';
ALTER TABLE memory_items ADD COLUMN IF NOT EXISTS agent_id INTEGER REFERENCES agents(id);

-- Fix 4: Create performance indexes for memory operations
CREATE INDEX IF NOT EXISTS idx_memory_items_embedding ON memory_items USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_embedding ON knowledge_nodes USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_learning_outcomes_task_type ON learning_outcomes(task_type);
CREATE INDEX IF NOT EXISTS idx_memory_items_agent_memory_level ON memory_items(agent_id, memory_level);

-- Update schema version for PRD-05 fixes
INSERT INTO schema_versions (version, description) 
VALUES ('1.1.1', 'PRD-05 Memory System Fixes: columns, vectors, indexes')
ON CONFLICT DO NOTHING;
