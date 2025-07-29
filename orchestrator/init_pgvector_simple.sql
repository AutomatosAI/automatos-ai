-- Simple pgvector initialization for Automotas AI
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create embeddings table for storing vector representations
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255),
    content_type VARCHAR(100) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    content_text TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for embeddings
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_embeddings_content_type ON embeddings(content_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_content_hash ON embeddings(content_hash);
CREATE INDEX IF NOT EXISTS idx_embeddings_workflow_id ON embeddings(workflow_id);

-- Create knowledge_base table for storing reusable knowledge
CREATE TABLE IF NOT EXISTS knowledge_base (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    knowledge_type VARCHAR(100) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    content TEXT NOT NULL,
    embedding vector(1536),
    tags JSONB,
    usage_count INTEGER DEFAULT 0,
    success_rate DECIMAL(5,2) DEFAULT 0.0,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for knowledge base
CREATE INDEX IF NOT EXISTS idx_knowledge_base_vector ON knowledge_base USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_knowledge_base_type ON knowledge_base(knowledge_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_base_tags ON knowledge_base USING GIN (tags);

-- Function to find similar embeddings
CREATE OR REPLACE FUNCTION find_similar_embeddings(
    query_embedding vector(1536),
    content_type_filter VARCHAR(100) DEFAULT NULL,
    similarity_threshold DECIMAL(3,2) DEFAULT 0.8,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    content_text TEXT,
    similarity DECIMAL(5,4),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id,
        e.content_text,
        (1 - (e.embedding <=> query_embedding))::DECIMAL(5,4) as similarity,
        e.metadata,
        e.created_at
    FROM embeddings e
    WHERE 
        (content_type_filter IS NULL OR e.content_type = content_type_filter)
        AND (1 - (e.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Insert some initial knowledge base entries
INSERT INTO knowledge_base (knowledge_type, title, description, content, tags) VALUES
(
    'pattern',
    'FastAPI JWT Authentication',
    'Complete JWT authentication implementation for FastAPI applications',
    'Implementation pattern for JWT-based authentication in FastAPI with user registration, login, and protected routes.',
    '["fastapi", "jwt", "authentication", "security", "python"]'
),
(
    'pattern',
    'React TypeScript Component Structure',
    'Standard structure for React components with TypeScript',
    'Best practices for organizing React components with TypeScript, including props interfaces, state management, and error handling.',
    '["react", "typescript", "components", "frontend", "javascript"]'
),
(
    'template',
    'Docker Multi-Stage Build',
    'Optimized Docker multi-stage build configuration',
    'Template for creating efficient Docker images using multi-stage builds for production deployments.',
    '["docker", "deployment", "optimization", "devops"]'
)
ON CONFLICT DO NOTHING;

-- Display initialization summary
SELECT 
    'pgvector initialization completed successfully' as message,
    COUNT(*) as knowledge_entries_created
FROM knowledge_base;
