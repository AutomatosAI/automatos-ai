
-- Initialize pgvector extension for vector operations
-- This script sets up vector database capabilities for the Automotas AI system

-- Create the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create embeddings table for storing vector representations
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255),
    content_type VARCHAR(100) NOT NULL, -- 'task', 'code', 'documentation', etc.
    content_hash VARCHAR(64) NOT NULL, -- SHA-256 hash of content for deduplication
    content_text TEXT NOT NULL,
    embedding vector(1536), -- OpenAI ada-002 embedding dimension
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- Note: workflow_id is a string reference, no foreign key constraint for now
);

-- Create vector similarity search index
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_embeddings_content_type ON embeddings(content_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_content_hash ON embeddings(content_hash);
CREATE INDEX IF NOT EXISTS idx_embeddings_workflow_id ON embeddings(workflow_id);

-- Create knowledge_base table for storing reusable knowledge
CREATE TABLE IF NOT EXISTS knowledge_base (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    knowledge_type VARCHAR(100) NOT NULL, -- 'pattern', 'solution', 'template', etc.
    title VARCHAR(255) NOT NULL,
    description TEXT,
    content TEXT NOT NULL,
    embedding vector(1536),
    tags JSONB, -- Array of tags for categorization
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
CREATE INDEX IF NOT EXISTS idx_knowledge_base_usage_count ON knowledge_base(usage_count DESC);

-- Create code_patterns table for storing reusable code patterns
CREATE TABLE IF NOT EXISTS code_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_name VARCHAR(255) NOT NULL,
    programming_language VARCHAR(50) NOT NULL,
    pattern_type VARCHAR(100) NOT NULL, -- 'api_endpoint', 'authentication', 'database_model', etc.
    description TEXT,
    code_template TEXT NOT NULL,
    embedding vector(1536),
    parameters JSONB, -- Template parameters and their descriptions
    dependencies JSONB, -- Required dependencies/imports
    usage_examples JSONB, -- Array of usage examples
    complexity_score INTEGER DEFAULT 5, -- 1-10 complexity rating
    usage_count INTEGER DEFAULT 0,
    success_rate DECIMAL(5,2) DEFAULT 0.0,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for code patterns
CREATE INDEX IF NOT EXISTS idx_code_patterns_vector ON code_patterns USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_code_patterns_language ON code_patterns(programming_language);
CREATE INDEX IF NOT EXISTS idx_code_patterns_type ON code_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_code_patterns_complexity ON code_patterns(complexity_score);
CREATE INDEX IF NOT EXISTS idx_code_patterns_usage ON code_patterns(usage_count DESC);

-- Create task_solutions table for storing successful task solutions
CREATE TABLE IF NOT EXISTS task_solutions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_description TEXT NOT NULL,
    task_embedding vector(1536),
    solution_approach TEXT NOT NULL,
    generated_files JSONB, -- Array of file information
    execution_time_seconds DECIMAL(10,3),
    success_metrics JSONB, -- Quality scores, test results, etc.
    technologies_used JSONB, -- Array of technologies/frameworks used
    complexity_score INTEGER DEFAULT 5,
    reuse_count INTEGER DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for task solutions
CREATE INDEX IF NOT EXISTS idx_task_solutions_vector ON task_solutions USING ivfflat (task_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_task_solutions_complexity ON task_solutions(complexity_score);
CREATE INDEX IF NOT EXISTS idx_task_solutions_reuse_count ON task_solutions(reuse_count DESC);
CREATE INDEX IF NOT EXISTS idx_task_solutions_execution_time ON task_solutions(execution_time_seconds);

-- Create functions for vector similarity search

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

-- Function to find similar code patterns
CREATE OR REPLACE FUNCTION find_similar_code_patterns(
    query_embedding vector(1536),
    language_filter VARCHAR(50) DEFAULT NULL,
    pattern_type_filter VARCHAR(100) DEFAULT NULL,
    similarity_threshold DECIMAL(3,2) DEFAULT 0.7,
    max_results INTEGER DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    pattern_name VARCHAR(255),
    programming_language VARCHAR(50),
    pattern_type VARCHAR(100),
    code_template TEXT,
    similarity DECIMAL(5,4),
    usage_count INTEGER,
    success_rate DECIMAL(5,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cp.id,
        cp.pattern_name,
        cp.programming_language,
        cp.pattern_type,
        cp.code_template,
        (1 - (cp.embedding <=> query_embedding))::DECIMAL(5,4) as similarity,
        cp.usage_count,
        cp.success_rate
    FROM code_patterns cp
    WHERE 
        (language_filter IS NULL OR cp.programming_language = language_filter)
        AND (pattern_type_filter IS NULL OR cp.pattern_type = pattern_type_filter)
        AND (1 - (cp.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY 
        (cp.embedding <=> query_embedding),
        cp.usage_count DESC,
        cp.success_rate DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Function to find similar task solutions
CREATE OR REPLACE FUNCTION find_similar_task_solutions(
    query_embedding vector(1536),
    complexity_max INTEGER DEFAULT 10,
    similarity_threshold DECIMAL(3,2) DEFAULT 0.75,
    max_results INTEGER DEFAULT 3
)
RETURNS TABLE (
    id UUID,
    task_description TEXT,
    solution_approach TEXT,
    similarity DECIMAL(5,4),
    complexity_score INTEGER,
    execution_time_seconds DECIMAL(10,3),
    reuse_count INTEGER,
    technologies_used JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ts.id,
        ts.task_description,
        ts.solution_approach,
        (1 - (ts.task_embedding <=> query_embedding))::DECIMAL(5,4) as similarity,
        ts.complexity_score,
        ts.execution_time_seconds,
        ts.reuse_count,
        ts.technologies_used
    FROM task_solutions ts
    WHERE 
        ts.complexity_score <= complexity_max
        AND (1 - (ts.task_embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY 
        (ts.task_embedding <=> query_embedding),
        ts.reuse_count DESC,
        ts.execution_time_seconds ASC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Create triggers to update usage counts and success rates
CREATE OR REPLACE FUNCTION update_pattern_usage()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE code_patterns 
    SET usage_count = usage_count + 1,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.pattern_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_solution_reuse()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE task_solutions 
    SET reuse_count = reuse_count + 1
    WHERE id = NEW.solution_id;
    RETURN NEW;
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
),
(
    'solution',
    'Database Migration Strategy',
    'Safe database migration approach for production systems',
    'Step-by-step approach for handling database schema changes in production environments with rollback capabilities.',
    '["database", "migration", "production", "safety", "devops"]'
)
ON CONFLICT DO NOTHING;

-- Insert some initial code patterns
INSERT INTO code_patterns (pattern_name, programming_language, pattern_type, description, code_template, parameters, dependencies) VALUES
(
    'FastAPI CRUD Endpoint',
    'python',
    'api_endpoint',
    'Standard CRUD operations endpoint for FastAPI',
    'from fastapi import APIRouter, Depends, HTTPException\nfrom sqlalchemy.orm import Session\n\nrouter = APIRouter()\n\n@router.post("/{resource}/")\nasync def create_{resource}(item: {Model}Create, db: Session = Depends(get_db)):\n    # Implementation here\n    pass',
    '{"resource": "Resource name", "Model": "Pydantic model name"}',
    '["fastapi", "sqlalchemy", "pydantic"]'
),
(
    'React Functional Component',
    'typescript',
    'component',
    'Standard React functional component with TypeScript',
    'import React, { useState, useEffect } from ''react'';\n\ninterface {ComponentName}Props {\n  // Props interface\n}\n\nconst {ComponentName}: React.FC<{ComponentName}Props> = (props) => {\n  // Component implementation\n  return (\n    <div>\n      {/* JSX content */}\n    </div>\n  );\n};\n\nexport default {ComponentName};',
    '{"ComponentName": "Name of the component"}',
    '["react", "@types/react"]'
)
ON CONFLICT DO NOTHING;

-- Create a view for vector search statistics
CREATE OR REPLACE VIEW vector_search_stats AS
SELECT 
    'embeddings' as table_name,
    COUNT(*) as total_records,
    COUNT(DISTINCT content_type) as unique_content_types,
    AVG(LENGTH(content_text)) as avg_content_length
FROM embeddings
UNION ALL
SELECT 
    'knowledge_base' as table_name,
    COUNT(*) as total_records,
    COUNT(DISTINCT knowledge_type) as unique_content_types,
    AVG(LENGTH(content)) as avg_content_length
FROM knowledge_base
UNION ALL
SELECT 
    'code_patterns' as table_name,
    COUNT(*) as total_records,
    COUNT(DISTINCT pattern_type) as unique_content_types,
    AVG(LENGTH(code_template)) as avg_content_length
FROM code_patterns
UNION ALL
SELECT 
    'task_solutions' as table_name,
    COUNT(*) as total_records,
    COUNT(DISTINCT complexity_score) as unique_content_types,
    AVG(LENGTH(task_description)) as avg_content_length
FROM task_solutions;

-- Log successful pgvector initialization (commented out until workflows table exists)
-- INSERT INTO workflows (workflow_id, workflow_type, status, task_prompt, created_at) 
-- VALUES ('init_pgvector_001', 'system', 'completed', 'pgvector extension and vector tables initialized successfully', CURRENT_TIMESTAMP)
-- ON CONFLICT (workflow_id) DO NOTHING;

-- Display initialization summary
SELECT 
    'pgvector initialization completed' as message,
    COUNT(*) as vector_tables_created
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('embeddings', 'knowledge_base', 'code_patterns', 'task_solutions');
