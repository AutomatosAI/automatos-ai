-- =====================================
-- AUTOMATOS AI DATABASE SCHEMA FIXES
-- =====================================
-- Based on API testing results - fixes 422 errors and missing columns
-- Generated: Thu Sep 18 20:42:42 WEST 2025

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS uuid-ossp;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- =====================================
-- 1. FIX DOCUMENTS TABLE - Missing Columns
-- =====================================

-- Add missing original_filename column (causing 500 errors)
ALTER TABLE documents ADD COLUMN IF NOT EXISTS original_filename VARCHAR(255);

-- Add other potentially missing document columns
ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_path VARCHAR(500);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_count INTEGER DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS tags TEXT[];
ALTER TABLE documents ADD COLUMN IF NOT EXISTS description TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_metadata JSON DEFAULT '{}';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS created_by INTEGER;

-- =====================================
-- 2. CREATE MISSING CORE TABLES
-- =====================================

-- Users table (referenced by foreign keys)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Agents table (referenced by foreign keys)
CREATE TABLE IF NOT EXISTS agents (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    agent_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    configuration JSON DEFAULT '{}',
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =====================================
-- 3. MEMORY SYSTEM TABLES
-- =====================================

-- Memory Items table
CREATE TABLE IF NOT EXISTS memory_items (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    memory_type VARCHAR(100) NOT NULL,
    importance REAL DEFAULT 0.5,
    metadata JSON DEFAULT '{}',
    tags TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- External Knowledge table
CREATE TABLE IF NOT EXISTS external_knowledge (
    id SERIAL PRIMARY KEY,
    source VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    knowledge_type VARCHAR(100) NOT NULL,
    metadata JSON DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =====================================
-- 4. FIELD THEORY TABLES
-- =====================================

-- Field States table
CREATE TABLE IF NOT EXISTS field_states (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    field_id VARCHAR(255) NOT NULL,
    field_type VARCHAR(100) NOT NULL,
    field_value REAL NOT NULL,
    gradient REAL[] DEFAULT '{}',
    context_data JSON NOT NULL,
    influence_weights REAL[] DEFAULT '{}',
    stability REAL DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Field Interactions table
CREATE TABLE IF NOT EXISTS field_interactions (
    id SERIAL PRIMARY KEY,
    task_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id),
    session_id_1 VARCHAR(255) NOT NULL,
    session_id_2 VARCHAR(255) NOT NULL,
    interaction_type VARCHAR(100) NOT NULL,
    similarity_threshold REAL DEFAULT 0.7,
    max_interactions INTEGER DEFAULT 50,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================
-- 5. MULTI-AGENT SYSTEM TABLES
-- =====================================

-- Agent Coordination table
CREATE TABLE IF NOT EXISTS agent_coordination (
    id SERIAL PRIMARY KEY,
    task_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id),
    agents TEXT[] NOT NULL,
    strategy VARCHAR(100) NOT NULL,
    load_balance BOOLEAN DEFAULT FALSE,
    context JSON DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Agent Behavior Monitoring table
CREATE TABLE IF NOT EXISTS agent_behavior_monitoring (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    agents TEXT[] NOT NULL,
    interactions JSON DEFAULT '[]',
    task_data JSON DEFAULT '{}',
    monitoring_start TIMESTAMP DEFAULT NOW(),
    monitoring_end TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Multi-Agent Reasoning table
CREATE TABLE IF NOT EXISTS multi_agent_reasoning (
    id SERIAL PRIMARY KEY,
    task_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id),
    agents TEXT[] NOT NULL,
    task JSON NOT NULL,
    strategy VARCHAR(100) NOT NULL,
    timeout_seconds INTEGER DEFAULT 300,
    status VARCHAR(50) DEFAULT 'pending',
    result JSON DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =====================================
-- 6. EVALUATION SYSTEM TABLES
-- =====================================

-- Evaluation Results table
CREATE TABLE IF NOT EXISTS evaluation_results (
    id SERIAL PRIMARY KEY,
    test_name VARCHAR(255) NOT NULL,
    score REAL NOT NULL,
    details JSON DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Benchmark Assessments table
CREATE TABLE IF NOT EXISTS benchmark_assessments (
    id SERIAL PRIMARY KEY,
    benchmark_name VARCHAR(255) NOT NULL,
    score REAL NOT NULL,
    metrics JSON DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Component Metrics table
CREATE TABLE IF NOT EXISTS component_metrics (
    id SERIAL PRIMARY KEY,
    component_name VARCHAR(255) NOT NULL,
    metrics JSON NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Integration Analysis table
CREATE TABLE IF NOT EXISTS integration_analysis (
    id SERIAL PRIMARY KEY,
    integration_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    analysis_data JSON DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW()
);

-- =====================================
-- 7. WORKFLOW SYSTEM TABLES
-- =====================================

-- Workflows table (if missing)
CREATE TABLE IF NOT EXISTS workflows (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    workflow_definition JSON NOT NULL,
    status VARCHAR(50) DEFAULT 'draft',
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Workflow Executions table
CREATE TABLE IF NOT EXISTS workflow_executions (
    id SERIAL PRIMARY KEY,
    workflow_id INTEGER NOT NULL REFERENCES workflows(id),
    input_data JSON DEFAULT '{}',
    output_data JSON DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    error_message TEXT,
    execution_time_ms INTEGER
);

-- =====================================
-- 8. CONTEXT ENGINEERING TABLES
-- =====================================

-- Context Queries table
CREATE TABLE IF NOT EXISTS context_queries (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_type VARCHAR(100) NOT NULL,
    results JSON DEFAULT '{}',
    response_time_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Context Sources table
CREATE TABLE IF NOT EXISTS context_sources (
    id SERIAL PRIMARY KEY,
    source_name VARCHAR(255) NOT NULL,
    source_type VARCHAR(100) NOT NULL,
    source_data JSON DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =====================================
-- 9. SYSTEM CONFIGURATION TABLES
-- =====================================

-- System Configurations table (if missing)
CREATE TABLE IF NOT EXISTS system_configurations (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSON NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- RAG Configurations table
CREATE TABLE IF NOT EXISTS rag_configurations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    embedding_model VARCHAR(255) NOT NULL,
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 200,
    retrieval_strategy VARCHAR(100) DEFAULT 'similarity',
    top_k INTEGER DEFAULT 5,
    similarity_threshold REAL DEFAULT 0.7,
    configuration JSON DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =====================================
-- 10. SKILLS AND PATTERNS TABLES
-- =====================================

-- Skills table
CREATE TABLE IF NOT EXISTS skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    skill_type VARCHAR(100) NOT NULL,
    category VARCHAR(100),
    configuration JSON DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Patterns table
CREATE TABLE IF NOT EXISTS patterns (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    pattern_type VARCHAR(100) NOT NULL,
    pattern_data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =====================================
-- 11. TASKS TABLE (Referenced by Foreign Keys)
-- =====================================

-- Tasks table (referenced by existing foreign keys)
CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    priority VARCHAR(50) DEFAULT 'medium',
    assigned_to INTEGER REFERENCES agents(id),
    created_by INTEGER REFERENCES users(id),
    due_date TIMESTAMP,
    completed_at TIMESTAMP,
    task_data JSON DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =====================================
-- 12. CREATE INDEXES FOR PERFORMANCE
-- =====================================

-- Documents indexes
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type);
CREATE INDEX IF NOT EXISTS idx_documents_upload_date ON documents(upload_date);
CREATE INDEX IF NOT EXISTS idx_documents_created_by ON documents(created_by);

-- Memory indexes
CREATE INDEX IF NOT EXISTS idx_memory_items_session_id ON memory_items(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_items_memory_type ON memory_items(memory_type);
CREATE INDEX IF NOT EXISTS idx_memory_items_created_at ON memory_items(created_at);

-- Field theory indexes
CREATE INDEX IF NOT EXISTS idx_field_states_session_id ON field_states(session_id);
CREATE INDEX IF NOT EXISTS idx_field_states_field_id ON field_states(field_id);
CREATE INDEX IF NOT EXISTS idx_field_interactions_task_id ON field_interactions(task_id);

-- Agent indexes
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_agent_type ON agents(agent_type);
CREATE INDEX IF NOT EXISTS idx_agents_created_by ON agents(created_by);

-- Workflow indexes
CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_id ON workflow_executions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(status);

-- =====================================
-- 13. INSERT DEFAULT DATA
-- =====================================

-- Default user (for system operations)
INSERT INTO users (id, username, email, full_name, is_active) 
VALUES (1, 'system', 'system@automatos.app', 'System User', TRUE)
ON CONFLICT (id) DO NOTHING;

-- Default RAG configuration
INSERT INTO rag_configurations (name, embedding_model, chunk_size, chunk_overlap, retrieval_strategy, top_k, similarity_threshold, is_active, created_by)
VALUES ('default', 'sentence-transformers/all-MiniLM-L6-v2', 1000, 200, 'similarity', 5, 0.7, TRUE, 'system')
ON CONFLICT (name) DO NOTHING;

-- Default system configurations
INSERT INTO system_configurations (config_key, config_value, description) VALUES
('system.max_agents', '{value: 100}', 'Maximum number of agents allowed in the system'),
('system.default_timeout', '{value: 300}', 'Default timeout for agent operations (seconds)'),
('rag.default_model', '{value: sentence-transformers/all-MiniLM-L6-v2}', 'Default embedding model for RAG system'),
('workflow.max_concurrent', '{value: 10}', 'Maximum concurrent workflow executions')
ON CONFLICT (config_key) DO NOTHING;

-- =====================================
-- 14. FINAL VALIDATION
-- =====================================

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Database schema fixes completed successfully!';
    RAISE NOTICE 'Fixed issues:';
    RAISE NOTICE '- Added missing documents.original_filename column';
    RAISE NOTICE '- Created missing core tables (users, agents, etc.)';
    RAISE NOTICE '- Added field theory tables for API validation';
    RAISE NOTICE '- Added memory system tables';
    RAISE NOTICE '- Added multi-agent coordination tables';
    RAISE NOTICE '- Added evaluation and workflow tables';
    RAISE NOTICE '- Created performance indexes';
    RAISE NOTICE '- Inserted default data';
END $$;
