-- Create missing tables for dashboard functionality
-- This fixes the database errors appearing in dashboard logs

-- 1. Context Optimization Metrics table
CREATE TABLE IF NOT EXISTS context_optimization_metrics (
    id SERIAL PRIMARY KEY,
    tokens_saved INTEGER NOT NULL DEFAULT 0,
    compression_ratio FLOAT NOT NULL DEFAULT 1.0,
    optimization_type VARCHAR(50),
    pattern_used VARCHAR(100),
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Memory Items table
CREATE TABLE IF NOT EXISTS memory_items (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER,
    memory_type VARCHAR(50) NOT NULL,
    memory_level VARCHAR(50) NOT NULL DEFAULT 'working', -- working, short_term, long_term
    content TEXT NOT NULL,
    importance FLOAT DEFAULT 0.5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE
);

-- 3. Knowledge Graph Nodes table
CREATE TABLE IF NOT EXISTS knowledge_graph_nodes (
    id SERIAL PRIMARY KEY,
    node_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    embedding_vector FLOAT[] DEFAULT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Collaboration Sessions table
CREATE TABLE IF NOT EXISTS collaboration_sessions (
    id SERIAL PRIMARY KEY,
    session_type VARCHAR(50) NOT NULL,
    initiator_agent_id INTEGER,
    participants JSONB NOT NULL DEFAULT '[]',
    status VARCHAR(20) NOT NULL DEFAULT 'active', -- active, completed, failed
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. Memory Consolidations table
CREATE TABLE IF NOT EXISTS memory_consolidations (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER,
    source_memory_ids INTEGER[] NOT NULL DEFAULT '{}',
    target_memory_id INTEGER,
    consolidation_type VARCHAR(50) NOT NULL,
    improvement_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_memory_items_agent_id ON memory_items(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_items_memory_level ON memory_items(memory_level);
CREATE INDEX IF NOT EXISTS idx_memory_items_created_at ON memory_items(created_at);
CREATE INDEX IF NOT EXISTS idx_context_metrics_created_at ON context_optimization_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_created_at ON knowledge_graph_nodes(created_at);
CREATE INDEX IF NOT EXISTS idx_collaboration_sessions_status ON collaboration_sessions(status);
CREATE INDEX IF NOT EXISTS idx_memory_consolidations_agent_id ON memory_consolidations(agent_id);
