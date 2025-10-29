-- ================================================================
-- PRD-21: Database Knowledge Source Schema Migration
-- ================================================================
-- Multi-tenant isolated database knowledge with semantic layer
-- Integrates with PRD-18 (Credentials) and PRD-17 (Dynamic Tools)
-- ================================================================

-- Main database knowledge source table
CREATE TABLE IF NOT EXISTS database_knowledge_sources (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER NOT NULL,  -- Multi-tenant isolation
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Credential reference (uses PRD-18 credential system)
    credential_id INTEGER REFERENCES credentials(id) ON DELETE RESTRICT,
    
    -- Database configuration
    dialect VARCHAR(50) NOT NULL CHECK (dialect IN ('postgresql', 'mysql', 'sqlite', 'mssql', 'snowflake', 'bigquery', 'redshift')),
    connection_pool_size INTEGER DEFAULT 5,
    max_rows_limit INTEGER DEFAULT 10000,
    query_timeout_seconds INTEGER DEFAULT 30,
    
    -- Schema metadata (cached in Redis too)
    schema_metadata JSONB,
    schema_version INTEGER DEFAULT 1,
    schema_hash VARCHAR(64),
    last_introspected TIMESTAMP,
    
    -- Semantic layer configuration
    semantic_layer JSONB,
    
    -- Caching configuration
    schema_cache_ttl INTEGER DEFAULT 3600,  -- 1 hour
    query_cache_ttl INTEGER DEFAULT 300,    -- 5 minutes
    
    -- Statistics
    total_queries_executed INTEGER DEFAULT 0,
    avg_query_time_ms FLOAT,
    last_successful_query TIMESTAMP,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    status VARCHAR(50) DEFAULT 'active',
    error_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Indexes for performance
    UNIQUE(tenant_id, name)  -- Unique name per tenant
);

CREATE INDEX idx_dks_tenant_name ON database_knowledge_sources(tenant_id, name);
CREATE INDEX idx_dks_credential ON database_knowledge_sources(credential_id);
CREATE INDEX idx_dks_status ON database_knowledge_sources(status, is_active);

-- Table relationships for JOIN optimization
CREATE TABLE IF NOT EXISTS database_relationships (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES database_knowledge_sources(id) ON DELETE CASCADE,
    
    from_table VARCHAR(255) NOT NULL,
    from_column VARCHAR(255) NOT NULL,
    to_table VARCHAR(255) NOT NULL,
    to_column VARCHAR(255) NOT NULL,
    
    relationship_type VARCHAR(50), -- one-to-one, one-to-many, many-to-many
    is_inferred BOOLEAN DEFAULT FALSE,
    confidence FLOAT DEFAULT 1.0,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_dr_source ON database_relationships(source_id);
CREATE INDEX idx_dr_tables ON database_relationships(from_table, to_table);

-- Query audit trail (multi-tenant isolated)
CREATE TABLE IF NOT EXISTS database_query_audit (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER NOT NULL,
    source_id INTEGER REFERENCES database_knowledge_sources(id) ON DELETE CASCADE,
    
    user_id INTEGER,
    agent_id VARCHAR(255),
    session_id VARCHAR(255),
    
    -- Query details
    natural_language_query TEXT NOT NULL,
    generated_sql TEXT,
    validated_sql TEXT,
    
    -- Execution metrics
    execution_time_ms INTEGER,
    row_count INTEGER,
    bytes_processed INTEGER,
    
    -- Status
    success BOOLEAN NOT NULL,
    error_message TEXT,
    validation_errors JSONB,
    
    -- Caching
    was_cached BOOLEAN DEFAULT FALSE,
    cache_key VARCHAR(64),
    
    -- Metadata
    visualization_type VARCHAR(50),
    confidence_score FLOAT,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_dqa_tenant_source ON database_query_audit(tenant_id, source_id);
CREATE INDEX idx_dqa_user ON database_query_audit(user_id);
CREATE INDEX idx_dqa_created ON database_query_audit(created_at DESC);

-- Semantic metrics (PROMINENT FEATURE)
CREATE TABLE IF NOT EXISTS semantic_metrics (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES database_knowledge_sources(id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    
    name VARCHAR(255) NOT NULL,
    display_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    
    sql_expression TEXT NOT NULL,
    aggregation VARCHAR(50),
    format VARCHAR(50),
    
    description TEXT,
    business_definition TEXT,
    
    tables_used JSONB,
    drill_down_dimensions JSONB,
    
    supports_time_grain BOOLEAN DEFAULT TRUE,
    default_time_grain VARCHAR(50),
    
    -- Usage tracking
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMP,
    
    -- UI prominence
    is_featured BOOLEAN DEFAULT FALSE,
    is_certified BOOLEAN DEFAULT FALSE,
    
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(source_id, name)
);

CREATE INDEX idx_sm_source ON semantic_metrics(source_id);
CREATE INDEX idx_sm_featured ON semantic_metrics(is_featured, is_certified);

-- Semantic dimensions (PROMINENT FEATURE)
CREATE TABLE IF NOT EXISTS semantic_dimensions (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES database_knowledge_sources(id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    
    name VARCHAR(255) NOT NULL,
    display_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    
    sql_expression TEXT NOT NULL,
    type VARCHAR(50), -- categorical, temporal, geographical, hierarchical
    
    description TEXT,
    
    -- Hierarchy support
    hierarchy_levels JSONB,
    parent_dimension_id INTEGER REFERENCES semantic_dimensions(id),
    
    -- Cached values
    cached_values JSONB,
    total_unique_values INTEGER,
    
    is_featured BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(source_id, name)
);

CREATE INDEX idx_sd_source ON semantic_dimensions(source_id);
CREATE INDEX idx_sd_parent ON semantic_dimensions(parent_dimension_id);

-- Query templates (20+ PER DATABASE TYPE!)
CREATE TABLE IF NOT EXISTS database_query_templates (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES database_knowledge_sources(id) ON DELETE CASCADE,
    
    -- Global templates have NULL source_id
    dialect VARCHAR(50),
    category VARCHAR(100),
    
    name VARCHAR(255) NOT NULL,
    description TEXT,
    natural_language TEXT NOT NULL,
    sql_template TEXT,
    
    parameters JSONB,
    visualization_type VARCHAR(50),
    
    -- Usage tracking
    usage_count INTEGER DEFAULT 0,
    avg_execution_time_ms FLOAT,
    
    tags JSONB,
    
    is_featured BOOLEAN DEFAULT FALSE,
    is_certified BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_dqt_source ON database_query_templates(source_id);
CREATE INDEX idx_dqt_dialect ON database_query_templates(dialect, category);
CREATE INDEX idx_dqt_featured ON database_query_templates(is_featured);

-- ================================================================
-- Sample Templates Data (PostgreSQL)
-- ================================================================
INSERT INTO database_query_templates (
    dialect, category, name, description, natural_language, sql_template, 
    parameters, visualization_type, tags, is_featured, is_certified
) VALUES 
(
    'postgresql', 'analytics', 'Top N by Revenue',
    'Show top customers by revenue with time filter',
    'Show top {n} customers by revenue in the last {days} days',
    'SELECT customer_id, customer_name, SUM(amount) as total_revenue FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL ''{days} days'' GROUP BY customer_id, customer_name ORDER BY total_revenue DESC LIMIT {n}',
    '[{"name": "n", "type": "integer", "default": 10}, {"name": "days", "type": "integer", "default": 30}]'::jsonb,
    'bar',
    '["revenue", "customers", "top-n"]'::jsonb,
    true, true
),
(
    'postgresql', 'analytics', 'Revenue Trend',
    'Revenue trend over time',
    'Show revenue trend over the last {months} months',
    'SELECT DATE_TRUNC(''month'', order_date) as month, SUM(amount) as revenue FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL ''{months} months'' GROUP BY month ORDER BY month',
    '[{"name": "months", "type": "integer", "default": 12}]'::jsonb,
    'line',
    '["revenue", "trend", "time-series"]'::jsonb,
    true, true
),
(
    'postgresql', 'reporting', 'Daily Summary',
    'Daily business metrics summary',
    'Daily business summary for {date}',
    'SELECT COUNT(DISTINCT order_id) as total_orders, COUNT(DISTINCT customer_id) as unique_customers, SUM(amount) as revenue, AVG(amount) as avg_order_value FROM orders WHERE DATE(order_date) = ''{date}''',
    '[{"name": "date", "type": "date", "default": "CURRENT_DATE"}]'::jsonb,
    'table',
    '["daily", "summary", "kpi"]'::jsonb,
    true, true
);

-- ================================================================
-- Trigger for updated_at timestamps
-- ================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_database_knowledge_sources_updated_at 
    BEFORE UPDATE ON database_knowledge_sources 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_semantic_metrics_updated_at 
    BEFORE UPDATE ON semantic_metrics 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_semantic_dimensions_updated_at 
    BEFORE UPDATE ON semantic_dimensions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ================================================================
-- Permissions (adjust based on your user setup)
-- ================================================================
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO automatos_user;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO automatos_user;
