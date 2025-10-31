-- ================================================================
-- PRD-19: Multimodal Knowledge Base Enhancement
-- ================================================================
-- Advanced multimodal knowledge storage and retrieval system
-- Adds support for tables, images, formulas, diagrams, and more
--
-- Research Foundation:
-- Multimodal processing concepts informed by RAG-Anything (MIT License)
-- Implementation is original Automatos code
--
-- Created: October 2025
-- Author: Automatos AI Team
-- ================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text similarity

-- ================================================================
-- 1. Knowledge Base Types Registry
-- ================================================================
-- Central registry of all knowledge base types in the system

CREATE TABLE IF NOT EXISTS kb_types (
    id SERIAL PRIMARY KEY,
    type_name VARCHAR(100) UNIQUE NOT NULL,  -- 'document', 'codegraph', 'table', 'image', 'formula', 'diagram', 'knowledge_graph', 'custom'
    display_name VARCHAR(255) NOT NULL,
    description TEXT,
    icon VARCHAR(50),  -- Icon identifier for UI
    processor_class VARCHAR(255),  -- Python class that handles this type
    storage_strategy VARCHAR(100),  -- 'pgvector', 'json', 'binary', 'hybrid'
    supports_embedding BOOLEAN DEFAULT true,
    supports_search BOOLEAN DEFAULT true,
    supports_relationships BOOLEAN DEFAULT false,
    enabled BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Insert default knowledge base types
INSERT INTO kb_types (type_name, display_name, description, icon, processor_class, storage_strategy, supports_relationships) VALUES
('document', 'Documents', 'Text documents (PDF, DOCX, MD, TXT)', 'file-text', 'DocumentProcessor', 'pgvector', false),
('codegraph', 'Code Graph', 'Source code symbols and relationships', 'code', 'CodeGraphProcessor', 'pgvector', true),
('table', 'Tables', 'Structured tabular data from documents', 'table', 'TableProcessor', 'hybrid', false),
('image', 'Images', 'Images and visual content with descriptions', 'image', 'ImageProcessor', 'hybrid', false),
('formula', 'Formulas', 'Mathematical formulas and equations (LaTeX)', 'function', 'FormulaProcessor', 'pgvector', false),
('diagram', 'Diagrams', 'Diagrams and flowcharts', 'diagram', 'DiagramProcessor', 'hybrid', true),
('knowledge_graph', 'Knowledge Graph', 'Concept relationships and knowledge nodes', 'network', 'KnowledgeGraphProcessor', 'pgvector', true),
('memory', 'Agent Memory', 'Agent experiences and learned patterns', 'brain', 'MemoryProcessor', 'pgvector', true)
ON CONFLICT (type_name) DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_kb_types_name ON kb_types(type_name);
CREATE INDEX IF NOT EXISTS idx_kb_types_enabled ON kb_types(enabled);

-- ================================================================
-- 2. Unified Knowledge Items Table
-- ================================================================
-- Polymorphic table that stores ALL knowledge items with type discrimination

CREATE TABLE IF NOT EXISTS knowledge_items (
    id SERIAL PRIMARY KEY,
    kb_type_id INTEGER REFERENCES kb_types(id) ON DELETE CASCADE,
    parent_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE,  -- For hierarchical relationships
    source_type VARCHAR(100),  -- 'upload', 'extraction', 'generation', 'import'
    source_id VARCHAR(255),  -- Original source identifier
    
    -- Core content
    title VARCHAR(500),
    content TEXT NOT NULL,
    summary TEXT,  -- AI-generated summary
    
    -- Embeddings (for all searchable types)
    embedding vector(1536),
    
    -- Rich metadata
    metadata JSONB DEFAULT '{}',  -- Type-specific metadata
    
    -- Quality metrics
    quality_score FLOAT DEFAULT 0.0,  -- 0.0-1.0
    importance_score FLOAT DEFAULT 0.0,  -- 0.0-1.0
    complexity_score FLOAT DEFAULT 0.0,  -- 0.0-1.0
    confidence_score FLOAT DEFAULT 1.0,  -- 0.0-1.0
    
    -- Access control
    visibility VARCHAR(50) DEFAULT 'system',  -- 'public', 'private', 'system', 'team'
    owner_id VARCHAR(255),
    permissions JSONB DEFAULT '{}',
    
    -- Status
    status VARCHAR(50) DEFAULT 'active',  -- 'active', 'archived', 'deleted', 'processing'
    version INTEGER DEFAULT 1,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    accessed_at TIMESTAMP,
    indexed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_knowledge_items_type ON knowledge_items(kb_type_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_parent ON knowledge_items(parent_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_source ON knowledge_items(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_status ON knowledge_items(status);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_owner ON knowledge_items(owner_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_quality ON knowledge_items(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_importance ON knowledge_items(importance_score DESC);

-- Vector similarity index
CREATE INDEX IF NOT EXISTS idx_knowledge_items_embedding 
ON knowledge_items USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- GIN index for metadata search
CREATE INDEX IF NOT EXISTS idx_knowledge_items_metadata ON knowledge_items USING GIN (metadata);

-- Full-text search on content
CREATE INDEX IF NOT EXISTS idx_knowledge_items_content_fts ON knowledge_items USING GIN (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_knowledge_items_title_fts ON knowledge_items USING GIN (to_tsvector('english', title));

-- ================================================================
-- 3. Multimodal Content Table
-- ================================================================
-- Stores type-specific data for multimodal content

CREATE TABLE IF NOT EXISTS multimodal_content (
    id SERIAL PRIMARY KEY,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE,
    content_modality VARCHAR(50) NOT NULL,  -- 'text', 'image', 'table', 'formula', 'diagram', 'video', 'audio'
    
    -- Original content
    original_format VARCHAR(50),  -- 'png', 'jpg', 'svg', 'latex', 'csv', etc.
    original_size_bytes INTEGER,
    original_data BYTEA,  -- Binary data for images, etc.
    
    -- Processed content
    processed_text TEXT,  -- Text representation
    processed_format VARCHAR(50),  -- 'markdown', 'html', 'json', etc.
    processed_data JSONB,  -- Structured representation
    
    -- Position metadata (for content extracted from documents)
    source_document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    page_number INTEGER,
    bounding_box JSONB,  -- {x, y, width, height}
    position_index INTEGER,  -- Order within document
    
    -- Extraction metadata
    extraction_method VARCHAR(100),  -- 'mineru', 'docling', 'tesseract', 'custom'
    extraction_confidence FLOAT,
    extraction_metadata JSONB DEFAULT '{}',
    
    -- Relationships to surrounding content
    context_before TEXT,
    context_after TEXT,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_multimodal_knowledge_item ON multimodal_content(knowledge_item_id);
CREATE INDEX IF NOT EXISTS idx_multimodal_modality ON multimodal_content(content_modality);
CREATE INDEX IF NOT EXISTS idx_multimodal_source_doc ON multimodal_content(source_document_id);
CREATE INDEX IF NOT EXISTS idx_multimodal_extraction ON multimodal_content(extraction_method);

-- ================================================================
-- 4. Knowledge Relationships Table
-- ================================================================
-- Cross-type relationships between knowledge items

CREATE TABLE IF NOT EXISTS knowledge_relationships (
    id SERIAL PRIMARY KEY,
    from_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE,
    to_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,  -- 'references', 'contains', 'similar_to', 'derived_from', 'depends_on', 'illustrated_by', 'calculated_from'
    strength FLOAT DEFAULT 1.0,  -- 0.0-1.0
    bidirectional BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255),
    
    UNIQUE(from_item_id, to_item_id, relationship_type)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_relationships_from ON knowledge_relationships(from_item_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_relationships_to ON knowledge_relationships(to_item_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_relationships_type ON knowledge_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_relationships_strength ON knowledge_relationships(strength DESC);

-- ================================================================
-- 5. Table-Specific Storage
-- ================================================================
-- Enhanced storage for tabular data

CREATE TABLE IF NOT EXISTS kb_tables (
    id SERIAL PRIMARY KEY,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE UNIQUE,
    
    -- Table structure
    headers JSONB NOT NULL,  -- Array of column names
    data_types JSONB,  -- Inferred data types for each column
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    
    -- Table data (multiple storage options)
    markdown_representation TEXT,  -- Markdown table format
    csv_data TEXT,  -- CSV format
    json_data JSONB,  -- JSON array of objects
    
    -- Statistics
    has_header_row BOOLEAN DEFAULT true,
    is_numeric BOOLEAN DEFAULT false,
    has_totals BOOLEAN DEFAULT false,
    
    -- Context
    caption TEXT,
    footnotes TEXT,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kb_tables_knowledge_item ON kb_tables(knowledge_item_id);
CREATE INDEX IF NOT EXISTS idx_kb_tables_size ON kb_tables(row_count, column_count);

-- ================================================================
-- 6. Image-Specific Storage
-- ================================================================
-- Enhanced storage for images with descriptions

CREATE TABLE IF NOT EXISTS kb_images (
    id SERIAL PRIMARY KEY,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE UNIQUE,
    
    -- Image metadata
    width INTEGER,
    height INTEGER,
    format VARCHAR(50),  -- 'png', 'jpg', 'svg', etc.
    file_size_bytes INTEGER,
    
    -- Visual analysis
    description TEXT,  -- AI-generated description
    caption TEXT,  -- Original caption
    alt_text TEXT,  -- Accessibility text
    detected_objects JSONB,  -- Array of detected objects
    detected_text TEXT,  -- OCR extracted text
    
    -- Image data storage
    image_data BYTEA,  -- Original image
    thumbnail_data BYTEA,  -- Thumbnail (200x200)
    storage_path VARCHAR(500),  -- Path if stored externally (S3, etc.)
    
    -- Embeddings
    visual_embedding vector(512),  -- CLIP or similar visual embedding
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kb_images_knowledge_item ON kb_images(knowledge_item_id);
CREATE INDEX IF NOT EXISTS idx_kb_images_format ON kb_images(format);

-- Visual similarity index
CREATE INDEX IF NOT EXISTS idx_kb_images_visual_embedding 
ON kb_images USING ivfflat (visual_embedding vector_cosine_ops) 
WITH (lists = 50);

-- ================================================================
-- 7. Formula-Specific Storage
-- ================================================================
-- Storage for mathematical formulas

CREATE TABLE IF NOT EXISTS kb_formulas (
    id SERIAL PRIMARY KEY,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE UNIQUE,
    
    -- Formula representations
    latex TEXT NOT NULL,  -- LaTeX source
    mathml TEXT,  -- MathML representation
    ascii_math TEXT,  -- ASCII representation
    
    -- Formula analysis
    variables JSONB,  -- Detected variables
    operators JSONB,  -- Mathematical operators
    complexity_level VARCHAR(50),  -- 'basic', 'intermediate', 'advanced'
    
    -- Context
    formula_type VARCHAR(100),  -- 'equation', 'inequality', 'expression', 'identity'
    domain VARCHAR(100),  -- 'algebra', 'calculus', 'statistics', etc.
    
    -- Rendering
    rendered_svg TEXT,  -- SVG rendering
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kb_formulas_knowledge_item ON kb_formulas(knowledge_item_id);
CREATE INDEX IF NOT EXISTS idx_kb_formulas_type ON kb_formulas(formula_type);
CREATE INDEX IF NOT EXISTS idx_kb_formulas_domain ON kb_formulas(domain);

-- ================================================================
-- 8. Knowledge Usage Analytics
-- ================================================================
-- Track usage of knowledge items for ranking and optimization

CREATE TABLE IF NOT EXISTS knowledge_usage (
    id SERIAL PRIMARY KEY,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,  -- 'retrieved', 'viewed', 'used_in_context', 'cited', 'rated'
    context_type VARCHAR(100),  -- 'rag_query', 'workflow', 'agent_task', 'user_query'
    query_text TEXT,
    relevance_score FLOAT,
    user_rating INTEGER,  -- 1-5 stars
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_usage_item ON knowledge_usage(knowledge_item_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_usage_event ON knowledge_usage(event_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_usage_timestamp ON knowledge_usage(timestamp DESC);

-- ================================================================
-- 9. Knowledge Collections
-- ================================================================
-- Organize knowledge items into collections/categories

CREATE TABLE IF NOT EXISTS knowledge_collections (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    kb_type_id INTEGER REFERENCES kb_types(id),  -- Optional: restrict to specific type
    icon VARCHAR(50),
    color VARCHAR(50),
    visibility VARCHAR(50) DEFAULT 'private',
    owner_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_collection_items (
    collection_id INTEGER REFERENCES knowledge_collections(id) ON DELETE CASCADE,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE,
    position INTEGER,  -- Order within collection
    added_at TIMESTAMP DEFAULT NOW(),
    added_by VARCHAR(255),
    PRIMARY KEY (collection_id, knowledge_item_id)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_collections_owner ON knowledge_collections(owner_id);
CREATE INDEX IF NOT EXISTS idx_collection_items_collection ON knowledge_collection_items(collection_id);
CREATE INDEX IF NOT EXISTS idx_collection_items_item ON knowledge_collection_items(knowledge_item_id);

-- ================================================================
-- 10. View: Unified Knowledge Search
-- ================================================================
-- Materialized view for fast cross-type search

CREATE MATERIALIZED VIEW IF NOT EXISTS knowledge_search_view AS
SELECT 
    ki.id,
    ki.kb_type_id,
    kt.type_name,
    kt.display_name as kb_type_display,
    ki.title,
    ki.content,
    ki.summary,
    ki.embedding,
    ki.quality_score,
    ki.importance_score,
    ki.complexity_score,
    ki.metadata,
    ki.status,
    ki.created_at,
    ki.updated_at,
    -- Aggregated usage stats
    COUNT(DISTINCT ku.id) as usage_count,
    AVG(ku.relevance_score) as avg_relevance,
    MAX(ku.timestamp) as last_used
FROM knowledge_items ki
JOIN kb_types kt ON ki.kb_type_id = kt.id
LEFT JOIN knowledge_usage ku ON ki.id = ku.knowledge_item_id
WHERE ki.status = 'active' AND kt.enabled = true
GROUP BY ki.id, kt.id, kt.type_name, kt.display_name;

CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_search_view_id ON knowledge_search_view(id);
CREATE INDEX IF NOT EXISTS idx_knowledge_search_view_type ON knowledge_search_view(kb_type_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_search_view_quality ON knowledge_search_view(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_search_view_usage ON knowledge_search_view(usage_count DESC);

-- Refresh function (call periodically)
-- REFRESH MATERIALIZED VIEW CONCURRENTLY knowledge_search_view;

-- ================================================================
-- 11. Helper Functions
-- ================================================================

-- Function to get knowledge item with all its multimodal content
CREATE OR REPLACE FUNCTION get_knowledge_item_complete(item_id INTEGER)
RETURNS TABLE (
    knowledge_item JSONB,
    multimodal_content JSONB,
    relationships JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        row_to_json(ki.*)::jsonb as knowledge_item,
        COALESCE(json_agg(DISTINCT mc.*) FILTER (WHERE mc.id IS NOT NULL), '[]'::json)::jsonb as multimodal_content,
        COALESCE(json_agg(DISTINCT kr.*) FILTER (WHERE kr.id IS NOT NULL), '[]'::json)::jsonb as relationships
    FROM knowledge_items ki
    LEFT JOIN multimodal_content mc ON ki.id = mc.knowledge_item_id
    LEFT JOIN knowledge_relationships kr ON ki.id = kr.from_item_id OR ki.id = kr.to_item_id
    WHERE ki.id = item_id
    GROUP BY ki.id;
END;
$$ LANGUAGE plpgsql;

-- Function to search across all knowledge types
CREATE OR REPLACE FUNCTION search_knowledge(
    search_query TEXT,
    kb_type_filter VARCHAR(100) DEFAULT NULL,
    limit_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    id INTEGER,
    type_name VARCHAR(100),
    title VARCHAR(500),
    content_snippet TEXT,
    relevance_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ki.id,
        kt.type_name,
        ki.title,
        LEFT(ki.content, 500) as content_snippet,
        ts_rank(to_tsvector('english', ki.content || ' ' || COALESCE(ki.title, '')), 
                plainto_tsquery('english', search_query)) as relevance_score
    FROM knowledge_items ki
    JOIN kb_types kt ON ki.kb_type_id = kt.id
    WHERE 
        (kb_type_filter IS NULL OR kt.type_name = kb_type_filter)
        AND ki.status = 'active'
        AND to_tsvector('english', ki.content || ' ' || COALESCE(ki.title, '')) @@ plainto_tsquery('english', search_query)
    ORDER BY relevance_score DESC
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- Comments for Documentation
-- ================================================================

COMMENT ON TABLE kb_types IS 'Registry of knowledge base types (documents, code, tables, images, etc.)';
COMMENT ON TABLE knowledge_items IS 'Unified polymorphic storage for all knowledge items';
COMMENT ON TABLE multimodal_content IS 'Type-specific multimodal content (images, tables, formulas)';
COMMENT ON TABLE knowledge_relationships IS 'Relationships between knowledge items across types';
COMMENT ON TABLE kb_tables IS 'Enhanced storage for tabular data with multiple formats';
COMMENT ON TABLE kb_images IS 'Image storage with AI descriptions and visual embeddings';
COMMENT ON TABLE kb_formulas IS 'Mathematical formula storage with LaTeX and analysis';
COMMENT ON TABLE knowledge_usage IS 'Analytics for knowledge item usage and relevance';
COMMENT ON TABLE knowledge_collections IS 'User-defined collections of knowledge items';

-- ================================================================
-- Verification Queries
-- ================================================================
-- SELECT * FROM kb_types;
-- SELECT type_name, COUNT(*) FROM knowledge_items ki JOIN kb_types kt ON ki.kb_type_id = kt.id GROUP BY type_name;
-- SELECT * FROM search_knowledge('quantum computing', NULL, 5);
-- SELECT * FROM get_knowledge_item_complete(1);

