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

