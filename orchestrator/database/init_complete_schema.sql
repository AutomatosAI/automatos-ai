-- ================================================================
-- Workflow Enhancements for PRD-10 (October 1, 2025)
-- ================================================================
-- Add missing fields to workflows table for frontend Edit/Delete functionality

ALTER TABLE workflows 
ADD COLUMN IF NOT EXISTS owner VARCHAR(255),
ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS default_policy_id VARCHAR(255);

-- Index for owner-based filtering
CREATE INDEX IF NOT EXISTS idx_workflows_owner ON workflows(owner);

-- GIN index for tag-based filtering  
CREATE INDEX IF NOT EXISTS idx_workflows_tags ON workflows USING GIN (tags);

COMMENT ON COLUMN workflows.owner IS 'Owner of the workflow (email, username, or team identifier)';
COMMENT ON COLUMN workflows.tags IS 'Array of tags for categorization and filtering';
COMMENT ON COLUMN workflows.default_policy_id IS 'Default policy configuration for workflow execution';

-- ================================================================
-- Document Usage Tracking Table for Context Engineering Analytics
-- ================================================================
-- This table tracks all document-related events for real-time analytics
-- Used by Context Engineering page to display performance metrics, query history, etc.

CREATE TABLE IF NOT EXISTS document_usage (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,  -- 'document_searched', 'rag_query', 'document_viewed', 'chunk_retrieved'
    document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,  -- Optional: which document was involved
    query TEXT,  -- Search query or RAG query text
    results_count INTEGER DEFAULT 0,  -- Number of results returned
    execution_time_ms INTEGER,  -- Response time in milliseconds
    metadata JSONB DEFAULT '{}',  -- Additional context (config_id, similarity_threshold, etc.)
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- When the event occurred
    user_id VARCHAR(255),  -- Optional: track per-user analytics
    session_id VARCHAR(255)  -- Optional: group events by session
);

-- Performance indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_document_usage_event_type ON document_usage(event_type);
CREATE INDEX IF NOT EXISTS idx_document_usage_timestamp ON document_usage(timestamp DESC);  -- DESC for recent queries
CREATE INDEX IF NOT EXISTS idx_document_usage_event_time ON document_usage(event_type, timestamp DESC);  -- Composite for filtered time-series
CREATE INDEX IF NOT EXISTS idx_document_usage_document ON document_usage(document_id) WHERE document_id IS NOT NULL;

-- GIN index for JSONB metadata queries (if needed for filtering by config_id, etc.)
CREATE INDEX IF NOT EXISTS idx_document_usage_metadata ON document_usage USING GIN (metadata);

-- ================================================================
-- Optional: Materialized view for faster performance dashboards
-- ================================================================
-- Uncomment if you need sub-second query performance for dashboards

-- CREATE MATERIALIZED VIEW IF NOT EXISTS document_usage_hourly AS
-- SELECT 
--     DATE_TRUNC('hour', timestamp) as hour,
--     event_type,
--     COUNT(*) as total_queries,
--     COUNT(CASE WHEN results_count > 0 THEN 1 END) as successful_queries,
--     AVG(execution_time_ms) as avg_execution_time_ms,
--     PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY execution_time_ms) as median_execution_time_ms,
--     PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_execution_time_ms,
--     MAX(execution_time_ms) as max_execution_time_ms
-- FROM document_usage
-- WHERE timestamp >= NOW() - INTERVAL '30 days'
-- GROUP BY DATE_TRUNC('hour', timestamp), event_type;

-- CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_hourly_unique ON document_usage_hourly(hour, event_type);

-- To refresh the materialized view (run periodically via cron or trigger):
-- REFRESH MATERIALIZED VIEW CONCURRENTLY document_usage_hourly;

-- ================================================================
-- Comments for documentation
-- ================================================================
COMMENT ON TABLE document_usage IS 'Tracks all document-related events for analytics and Context Engineering monitoring';
COMMENT ON COLUMN document_usage.event_type IS 'Type of event: document_searched, rag_query, document_viewed, chunk_retrieved';
COMMENT ON COLUMN document_usage.query IS 'Search or RAG query text for semantic analysis';
COMMENT ON COLUMN document_usage.results_count IS 'Number of results returned (0 indicates no results)';
COMMENT ON COLUMN document_usage.execution_time_ms IS 'Query execution time in milliseconds for performance tracking';
COMMENT ON COLUMN document_usage.metadata IS 'JSON metadata: config_id, similarity_threshold, agent_id, etc.';

-- ================================================================
-- Verification queries (for testing after migration)
-- ================================================================
-- SELECT COUNT(*) FROM document_usage;
-- SELECT event_type, COUNT(*), AVG(execution_time_ms) FROM document_usage GROUP BY event_type;
-- SELECT MAX(timestamp) as last_event FROM document_usage;

-- ================================================================
-- PRD-11: CodeGraph Implementation (October 2, 2025)
-- ================================================================
-- CodeGraph system for indexing and searching client codebases
-- Enables AI agents to access precise code context instead of entire repositories

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. CodeGraph Projects Table
-- Stores metadata about indexed code projects
CREATE TABLE IF NOT EXISTS codegraph_projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    source_type VARCHAR(50) NOT NULL,  -- 'github', 'gitlab', 'local'
    source_url TEXT,
    branch VARCHAR(255) DEFAULT 'main',
    language VARCHAR(50),  -- Primary language: 'python', 'typescript', etc.
    total_files INTEGER DEFAULT 0,
    total_symbols INTEGER DEFAULT 0,
    total_relationships INTEGER DEFAULT 0,
    last_indexed TIMESTAMP,
    index_duration_seconds FLOAT,
    status VARCHAR(50) DEFAULT 'active',  -- 'active', 'indexing', 'failed', 'archived'
    auto_reindex BOOLEAN DEFAULT false,
    exclude_patterns JSONB DEFAULT '[]',  -- Array of glob patterns to exclude
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_codegraph_projects_name ON codegraph_projects(name);
CREATE INDEX IF NOT EXISTS idx_codegraph_projects_status ON codegraph_projects(status);
CREATE INDEX IF NOT EXISTS idx_codegraph_projects_source_type ON codegraph_projects(source_type);

COMMENT ON TABLE codegraph_projects IS 'CodeGraph indexed projects - client codebases for AI agent context';
COMMENT ON COLUMN codegraph_projects.source_type IS 'Source: github, gitlab, or local directory';
COMMENT ON COLUMN codegraph_projects.exclude_patterns IS 'Glob patterns to exclude (node_modules, __pycache__, etc.)';

-- 2. CodeGraph Symbols Table
-- Stores extracted code symbols (functions, classes, etc.) with embeddings
CREATE TABLE IF NOT EXISTS codegraph_symbols (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES codegraph_projects(id) ON DELETE CASCADE,
    symbol_type VARCHAR(50) NOT NULL,  -- 'function', 'class', 'interface', 'variable', 'import'
    name VARCHAR(500) NOT NULL,
    qualified_name VARCHAR(1000),  -- Full path: module.Class.method
    file_path VARCHAR(1000) NOT NULL,
    line_number INTEGER NOT NULL,
    signature TEXT,  -- Function signature or class definition
    docstring TEXT,  -- Documentation string
    code_snippet TEXT,  -- Full code of the symbol
    embedding vector(1536),  -- OpenAI ada-002 embeddings for semantic search
    metadata JSONB DEFAULT '{}',  -- Language-specific metadata
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_codegraph_symbols_project ON codegraph_symbols(project_id);
CREATE INDEX IF NOT EXISTS idx_codegraph_symbols_type ON codegraph_symbols(symbol_type);
CREATE INDEX IF NOT EXISTS idx_codegraph_symbols_name ON codegraph_symbols(name);
CREATE INDEX IF NOT EXISTS idx_codegraph_symbols_file ON codegraph_symbols(file_path);

-- Vector similarity index for semantic search (using cosine distance)
CREATE INDEX IF NOT EXISTS idx_codegraph_symbols_embedding 
ON codegraph_symbols USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

COMMENT ON TABLE codegraph_symbols IS 'Code symbols extracted from projects with semantic embeddings';
COMMENT ON COLUMN codegraph_symbols.embedding IS 'Vector embedding for semantic code search (OpenAI ada-002, 1536 dimensions)';
COMMENT ON COLUMN codegraph_symbols.qualified_name IS 'Fully qualified symbol name including module path';

-- 3. CodeGraph Relationships Table
-- Stores relationships between symbols (calls, imports, inheritance)
CREATE TABLE IF NOT EXISTS codegraph_relationships (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES codegraph_projects(id) ON DELETE CASCADE,
    from_symbol_id INTEGER REFERENCES codegraph_symbols(id) ON DELETE CASCADE,
    to_symbol_id INTEGER REFERENCES codegraph_symbols(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL,  -- 'calls', 'imports', 'extends', 'implements', 'references'
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_codegraph_relationships_project ON codegraph_relationships(project_id);
CREATE INDEX IF NOT EXISTS idx_codegraph_relationships_from ON codegraph_relationships(from_symbol_id);
CREATE INDEX IF NOT EXISTS idx_codegraph_relationships_to ON codegraph_relationships(to_symbol_id);
CREATE INDEX IF NOT EXISTS idx_codegraph_relationships_type ON codegraph_relationships(relationship_type);

COMMENT ON TABLE codegraph_relationships IS 'Relationships between code symbols (call graphs, imports, inheritance)';
COMMENT ON COLUMN codegraph_relationships.relationship_type IS 'Type: calls, imports, extends, implements, references';

-- 4. CodeGraph Files Table
-- Tracks individual files for change detection and re-indexing
CREATE TABLE IF NOT EXISTS codegraph_files (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES codegraph_projects(id) ON DELETE CASCADE,
    file_path VARCHAR(1000) NOT NULL,
    file_hash VARCHAR(64),  -- SHA-256 for change detection
    file_size INTEGER,
    lines_of_code INTEGER,
    complexity_score INTEGER,  -- Cyclomatic complexity
    language VARCHAR(50),
    last_modified TIMESTAMP,
    indexed_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    UNIQUE(project_id, file_path)
);

CREATE INDEX IF NOT EXISTS idx_codegraph_files_project ON codegraph_files(project_id);
CREATE INDEX IF NOT EXISTS idx_codegraph_files_path ON codegraph_files(file_path);
CREATE INDEX IF NOT EXISTS idx_codegraph_files_hash ON codegraph_files(file_hash);

COMMENT ON TABLE codegraph_files IS 'Individual files in projects for change detection and incremental indexing';
COMMENT ON COLUMN codegraph_files.file_hash IS 'SHA-256 hash for detecting file changes';

-- 5. CodeGraph Query Logs Table
-- Tracks all CodeGraph queries for analytics
CREATE TABLE IF NOT EXISTS codegraph_query_logs (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES codegraph_projects(id) ON DELETE CASCADE,
    query_type VARCHAR(50) NOT NULL,  -- 'symbol', 'semantic', 'call_graph', 'dependency'
    query_text TEXT NOT NULL,
    results_count INTEGER,
    execution_time_ms FLOAT,
    user_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_codegraph_query_logs_project ON codegraph_query_logs(project_id);
CREATE INDEX IF NOT EXISTS idx_codegraph_query_logs_type ON codegraph_query_logs(query_type);
CREATE INDEX IF NOT EXISTS idx_codegraph_query_logs_created ON codegraph_query_logs(created_at DESC);

COMMENT ON TABLE codegraph_query_logs IS 'Analytics for CodeGraph queries (search performance, popular queries)';
COMMENT ON COLUMN codegraph_query_logs.query_type IS 'Type: symbol, semantic, call_graph, dependency';

-- ================================================================
-- CodeGraph Verification Queries
-- ================================================================
-- SELECT * FROM codegraph_projects;
-- SELECT project_id, symbol_type, COUNT(*) FROM codegraph_symbols GROUP BY project_id, symbol_type;
-- SELECT relationship_type, COUNT(*) FROM codegraph_relationships GROUP BY relationship_type;
-- SELECT query_type, COUNT(*), AVG(execution_time_ms) FROM codegraph_query_logs GROUP BY query_type;

