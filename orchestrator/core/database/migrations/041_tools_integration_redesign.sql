-- =====================================================
-- Migration 041: Tools Integration Redesign (PRD Master)
-- =====================================================
-- Purpose: Complete schema for metadata caching, agent-tool assignment, and tool routing
-- Date: 2026-01-24
-- Related: automatos_analysis/MASTER_PRD.md
-- =====================================================

-- =====================================================
-- SECTION 1: METADATA CACHING TABLES
-- =====================================================

-- Apps Metadata Cache
CREATE TABLE IF NOT EXISTS composio_apps_cache (
    id SERIAL PRIMARY KEY,
    app_name VARCHAR(100) NOT NULL UNIQUE,
    app_slug VARCHAR(100) NOT NULL UNIQUE,
    display_name VARCHAR(255) NOT NULL,
    description TEXT,
    logo_url TEXT,
    categories JSONB NOT NULL DEFAULT '[]',
    auth_schemes JSONB NOT NULL DEFAULT '[]',
    action_count INTEGER DEFAULT 0,
    trigger_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'ACTIVE',
    metadata JSONB DEFAULT '{}',
    last_synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_composio_apps_name ON composio_apps_cache(app_name);
CREATE INDEX IF NOT EXISTS idx_composio_apps_slug ON composio_apps_cache(app_slug);
CREATE INDEX IF NOT EXISTS idx_composio_apps_categories ON composio_apps_cache USING GIN(categories);
CREATE INDEX IF NOT EXISTS idx_composio_apps_status ON composio_apps_cache(status);
CREATE INDEX IF NOT EXISTS idx_composio_apps_last_synced ON composio_apps_cache(last_synced_at);

-- Actions Metadata Cache
CREATE TABLE IF NOT EXISTS composio_actions_cache (
    id SERIAL PRIMARY KEY,
    app_name VARCHAR(100) NOT NULL,
    action_name VARCHAR(255) NOT NULL,
    action_slug VARCHAR(255) NOT NULL,
    display_name VARCHAR(255) NOT NULL,
    description TEXT,
    parameters JSONB DEFAULT '{}',
    response_schema JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    category VARCHAR(100),
    is_premium BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    last_synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT uq_app_action UNIQUE(app_name, action_name),
    CONSTRAINT fk_app_name FOREIGN KEY (app_name) 
        REFERENCES composio_apps_cache(app_name) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_composio_actions_app ON composio_actions_cache(app_name);
CREATE INDEX IF NOT EXISTS idx_composio_actions_name ON composio_actions_cache(action_name);
CREATE INDEX IF NOT EXISTS idx_composio_actions_category ON composio_actions_cache(category);
CREATE INDEX IF NOT EXISTS idx_composio_actions_tags ON composio_actions_cache USING GIN(tags);

-- Stats Cache
CREATE TABLE IF NOT EXISTS composio_stats_cache (
    id SERIAL PRIMARY KEY,
    stat_key VARCHAR(100) NOT NULL UNIQUE,
    stat_value JSONB NOT NULL,
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_composio_stats_key ON composio_stats_cache(stat_key);

-- Sync Jobs Tracking
CREATE TABLE IF NOT EXISTS composio_sync_jobs (
    id SERIAL PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    apps_synced INTEGER DEFAULT 0,
    actions_synced INTEGER DEFAULT 0,
    errors_count INTEGER DEFAULT 0,
    error_details JSONB,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_ms INTEGER,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sync_jobs_type ON composio_sync_jobs(job_type);
CREATE INDEX IF NOT EXISTS idx_sync_jobs_status ON composio_sync_jobs(status);
CREATE INDEX IF NOT EXISTS idx_sync_jobs_started ON composio_sync_jobs(started_at DESC);

-- =====================================================
-- SECTION 2: AGENT-TOOL ASSIGNMENT
-- =====================================================

CREATE TABLE IF NOT EXISTS agent_app_assignments (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER NOT NULL,
    app_name VARCHAR(100) NOT NULL,
    app_type VARCHAR(50) NOT NULL DEFAULT 'EXTERNAL',
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    assigned_by INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 0,
    config JSONB DEFAULT '{}',
    
    CONSTRAINT fk_agent_assignment FOREIGN KEY (agent_id) 
        REFERENCES agents(id) ON DELETE CASCADE,
    CONSTRAINT uq_agent_app_assignment UNIQUE(agent_id, app_name)
);

CREATE INDEX IF NOT EXISTS idx_agent_assignments_agent ON agent_app_assignments(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_assignments_app ON agent_app_assignments(app_name);
CREATE INDEX IF NOT EXISTS idx_agent_assignments_type ON agent_app_assignments(app_type);
CREATE INDEX IF NOT EXISTS idx_agent_assignments_active ON agent_app_assignments(is_active);
CREATE INDEX IF NOT EXISTS idx_agent_assignments_priority ON agent_app_assignments(priority DESC);

-- =====================================================
-- SECTION 3: TOOL EXECUTION TRACKING
-- =====================================================

CREATE TABLE IF NOT EXISTS tool_execution_logs (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER NOT NULL,
    app_name VARCHAR(100) NOT NULL,
    action_name VARCHAR(255) NOT NULL,
    user_id INTEGER NOT NULL,
    workspace_id UUID NOT NULL,
    
    input_parameters JSONB,
    user_query TEXT,
    output_result JSONB,
    
    status VARCHAR(50) NOT NULL,
    error_message TEXT,
    error_code VARCHAR(100),
    
    execution_time_ms INTEGER,
    token_usage JSONB,
    router_decision JSONB,
    cache_hit BOOLEAN DEFAULT FALSE,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_tool_execution_agent FOREIGN KEY (agent_id) 
        REFERENCES agents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tool_logs_agent ON tool_execution_logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_logs_app ON tool_execution_logs(app_name);
CREATE INDEX IF NOT EXISTS idx_tool_logs_user ON tool_execution_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_tool_logs_workspace ON tool_execution_logs(workspace_id);
CREATE INDEX IF NOT EXISTS idx_tool_logs_status ON tool_execution_logs(status);
CREATE INDEX IF NOT EXISTS idx_tool_logs_executed ON tool_execution_logs(executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_tool_logs_agent_status ON tool_execution_logs(agent_id, status, executed_at DESC);

-- =====================================================
-- SECTION 4: TOOL ROUTING CACHE
-- =====================================================

CREATE TABLE IF NOT EXISTS intent_classification_cache (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    classified_category VARCHAR(100) NOT NULL,
    classified_action_type VARCHAR(100),
    confidence FLOAT,
    reasoning TEXT,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hit_count INTEGER DEFAULT 0,
    last_hit_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_intent_cache_category ON intent_classification_cache(classified_category);
CREATE INDEX IF NOT EXISTS idx_intent_cache_cached ON intent_classification_cache(cached_at DESC);
CREATE INDEX IF NOT EXISTS idx_intent_cache_hits ON intent_classification_cache(hit_count DESC);

CREATE TABLE IF NOT EXISTS tool_execution_cache (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER NOT NULL,
    app_name VARCHAR(100) NOT NULL,
    action_name VARCHAR(255) NOT NULL,
    query_text TEXT NOT NULL,
    input_parameters JSONB,
    cached_result JSONB NOT NULL,
    cache_key VARCHAR(255),
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    hit_count INTEGER DEFAULT 0,
    last_hit_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tool_exec_cache_agent ON tool_execution_cache(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_exec_cache_app ON tool_execution_cache(app_name, action_name);
CREATE INDEX IF NOT EXISTS idx_tool_exec_cache_key ON tool_execution_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_tool_exec_cache_expires ON tool_execution_cache(expires_at);

-- =====================================================
-- SECTION 5: UPDATE EXISTING TABLES
-- =====================================================

-- Add last_synced_at to composio_connections if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'composio_connections' 
        AND column_name = 'last_synced_at'
    ) THEN
        ALTER TABLE composio_connections ADD COLUMN last_synced_at TIMESTAMP;
        CREATE INDEX idx_composio_connections_synced ON composio_connections(last_synced_at);
    END IF;
END $$;

-- Add usage tracking to agent_app_features if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agent_app_features' 
        AND column_name = 'usage_count'
    ) THEN
        ALTER TABLE agent_app_features ADD COLUMN usage_count INTEGER DEFAULT 0;
        ALTER TABLE agent_app_features ADD COLUMN last_used_at TIMESTAMP;
        CREATE INDEX idx_agent_app_features_usage ON agent_app_features(usage_count DESC);
    END IF;
END $$;

-- =====================================================
-- SECTION 6: SEED INTERNAL TOOLS
-- =====================================================

-- Insert internal tools (RAG, CodeGraph, Memory, NL2SQL)
INSERT INTO composio_apps_cache 
    (app_name, app_slug, display_name, description, categories, action_count, status)
VALUES
    ('RAG', 'rag', 'Knowledge Base (RAG)', 
     'Retrieval-Augmented Generation for querying knowledge bases',
     '["internal", "ai", "knowledge"]'::JSONB, 3, 'ACTIVE'),
    
    ('CODEGRAPH', 'codegraph', 'Code Analysis', 
     'Analyze code structure, dependencies, and relationships',
     '["internal", "code", "developer"]'::JSONB, 5, 'ACTIVE'),
    
    ('MEMORY', 'memory', 'Conversation Memory', 
     'Store and retrieve conversation history and context',
     '["internal", "ai", "memory"]'::JSONB, 4, 'ACTIVE'),
    
    ('NL2SQL', 'nl2sql', 'Natural Language to SQL', 
     'Convert natural language queries to SQL and query databases',
     '["internal", "database", "query"]'::JSONB, 2, 'ACTIVE')
ON CONFLICT (app_name) DO NOTHING;

-- Insert internal tool actions
INSERT INTO composio_actions_cache 
    (app_name, action_name, action_slug, display_name, description, category)
VALUES
    -- RAG actions
    ('RAG', 'RAG_SEARCH', 'rag_search', 'Search Knowledge Base', 
     'Search documents and knowledge bases using semantic search', 'knowledge'),
    ('RAG', 'RAG_INDEX', 'rag_index', 'Index Documents', 
     'Add new documents to the knowledge base', 'knowledge'),
    ('RAG', 'RAG_DELETE', 'rag_delete', 'Delete Documents', 
     'Remove documents from the knowledge base', 'knowledge'),
    
    -- CodeGraph actions
    ('CODEGRAPH', 'CODEGRAPH_ANALYZE', 'codegraph_analyze', 'Analyze Code', 
     'Analyze code structure and dependencies', 'code'),
    ('CODEGRAPH', 'CODEGRAPH_SEARCH', 'codegraph_search', 'Search Code', 
     'Search for functions, classes, or patterns in codebase', 'code'),
    ('CODEGRAPH', 'CODEGRAPH_DEPENDENCIES', 'codegraph_dependencies', 'Get Dependencies', 
     'Get dependency graph for a file or module', 'code'),
    ('CODEGRAPH', 'CODEGRAPH_IMPACT', 'codegraph_impact', 'Impact Analysis', 
     'Analyze impact of code changes', 'code'),
    ('CODEGRAPH', 'CODEGRAPH_METRICS', 'codegraph_metrics', 'Code Metrics', 
     'Get code quality metrics and statistics', 'code'),
    
    -- Memory actions
    ('MEMORY', 'MEMORY_STORE', 'memory_store', 'Store Memory', 
     'Store information in conversation memory', 'memory'),
    ('MEMORY', 'MEMORY_RETRIEVE', 'memory_retrieve', 'Retrieve Memory', 
     'Retrieve relevant information from memory', 'memory'),
    ('MEMORY', 'MEMORY_UPDATE', 'memory_update', 'Update Memory', 
     'Update existing memory entries', 'memory'),
    ('MEMORY', 'MEMORY_DELETE', 'memory_delete', 'Delete Memory', 
     'Remove specific memories', 'memory'),
    
    -- NL2SQL actions
    ('NL2SQL', 'NL2SQL_QUERY', 'nl2sql_query', 'Execute Natural Language Query', 
     'Convert natural language to SQL and execute query', 'database'),
    ('NL2SQL', 'NL2SQL_SCHEMA', 'nl2sql_schema', 'Get Database Schema', 
     'Retrieve database schema information', 'database')
ON CONFLICT (app_name, action_name) DO NOTHING;

-- =====================================================
-- SECTION 7: INITIAL STATS
-- =====================================================

INSERT INTO composio_stats_cache (stat_key, stat_value) 
VALUES 
    ('total_apps', '{"count": 4, "type": "internal"}'::JSONB),
    ('total_actions', '{"count": 14, "type": "internal"}'::JSONB),
    ('last_full_sync', '{"timestamp": null, "duration_ms": null}'::JSONB)
ON CONFLICT (stat_key) DO UPDATE SET 
    stat_value = EXCLUDED.stat_value,
    last_updated_at = CURRENT_TIMESTAMP;

-- =====================================================
-- SECTION 8: TRIGGERS FOR AUTO-UPDATE
-- =====================================================

-- Function: Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for composio_apps_cache
DROP TRIGGER IF EXISTS trg_composio_apps_cache_updated_at ON composio_apps_cache;
CREATE TRIGGER trg_composio_apps_cache_updated_at
    BEFORE UPDATE ON composio_apps_cache
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function: Track feature usage
CREATE OR REPLACE FUNCTION increment_feature_usage()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'success' THEN
        UPDATE agent_app_features
        SET 
            usage_count = usage_count + 1,
            last_used_at = NEW.executed_at
        WHERE 
            agent_id = NEW.agent_id 
            AND app_name = NEW.app_name 
            AND action_name = NEW.action_name;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for tracking feature usage
DROP TRIGGER IF EXISTS trg_track_feature_usage ON tool_execution_logs;
CREATE TRIGGER trg_track_feature_usage
    AFTER INSERT ON tool_execution_logs
    FOR EACH ROW
    EXECUTE FUNCTION increment_feature_usage();

-- =====================================================
-- SECTION 9: VIEWS FOR CONVENIENCE
-- =====================================================

-- View: Apps with connection status per workspace
CREATE OR REPLACE VIEW v_workspace_apps AS
SELECT 
    a.app_name,
    a.display_name,
    a.description,
    a.logo_url,
    a.categories,
    a.action_count,
    e.workspace_id,
    c.status AS connection_status,
    c.connected_at,
    COUNT(aaf.id) AS enabled_features_count
FROM composio_apps_cache a
LEFT JOIN composio_connections c ON a.app_name = c.app_name
LEFT JOIN composio_entities e ON c.entity_id = e.id
LEFT JOIN agent_app_features aaf ON a.app_name = aaf.app_name AND aaf.enabled = TRUE
GROUP BY a.app_name, a.display_name, a.description, a.logo_url, a.categories, 
         a.action_count, e.workspace_id, c.status, c.connected_at;

-- View: Agent tools summary
CREATE OR REPLACE VIEW v_agent_tools AS
SELECT 
    a.id AS agent_id,
    a.name AS agent_name,
    aal.app_name,
    aal.app_type,
    COUNT(aaf.id) AS enabled_features_count,
    aal.priority
FROM agents a
LEFT JOIN agent_app_assignments aal ON a.id = aal.agent_id AND aal.is_active = TRUE
LEFT JOIN agent_app_features aaf ON a.id = aaf.agent_id 
    AND aal.app_name = aaf.app_name 
    AND aaf.enabled = TRUE
GROUP BY a.id, a.name, aal.app_name, aal.app_type, aal.priority
ORDER BY a.id, aal.priority DESC;

-- View: Tool usage analytics (last 30 days)
CREATE OR REPLACE VIEW v_tool_usage_stats AS
SELECT 
    app_name,
    action_name,
    COUNT(*) AS total_executions,
    COUNT(CASE WHEN status = 'success' THEN 1 END) AS successful_executions,
    COUNT(CASE WHEN status = 'error' THEN 1 END) AS failed_executions,
    AVG(execution_time_ms) AS avg_execution_time_ms,
    MAX(execution_time_ms) AS max_execution_time_ms,
    MIN(execution_time_ms) AS min_execution_time_ms,
    COUNT(CASE WHEN cache_hit = TRUE THEN 1 END) AS cache_hits,
    COUNT(*) - COUNT(CASE WHEN cache_hit = TRUE THEN 1 END) AS cache_misses
FROM tool_execution_logs
WHERE executed_at >= NOW() - INTERVAL '30 days'
GROUP BY app_name, action_name
ORDER BY total_executions DESC;

-- =====================================================
-- MIGRATION COMPLETE
-- =====================================================

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE 'Migration 041 completed successfully';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Run initial metadata sync';
    RAISE NOTICE '2. Update API endpoints to use cache';
    RAISE NOTICE '3. Create scheduler for daily sync';
END $$;
