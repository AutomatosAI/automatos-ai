-- Migration 007: Knowledge Graph Entity System
-- Adds entity extraction and relationship tracking to multimodal knowledge base
-- Part of Hybrid RAG integration (PRD-21)

-- ============================================================================
-- ENTITIES TABLE
-- ============================================================================
-- Stores extracted entities (people, technologies, concepts, organizations)
CREATE TABLE IF NOT EXISTS kb_entities (
    id SERIAL PRIMARY KEY,
    entity_name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,  -- person, technology, concept, organization, location, event
    canonical_name VARCHAR(255),         -- normalized form for matching
    description TEXT,
    embedding vector(1536),              -- semantic embedding for similarity search
    mention_count INTEGER DEFAULT 0,     -- how many times mentioned across all documents
    importance_score FLOAT DEFAULT 0.0,  -- calculated from mentions + relationships
    metadata JSONB DEFAULT '{}',         -- flexible metadata storage
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Unique constraint on canonical name to prevent duplicates
CREATE UNIQUE INDEX idx_kb_entities_canonical ON kb_entities(LOWER(canonical_name));

-- Index for entity type filtering
CREATE INDEX idx_kb_entities_type ON kb_entities(entity_type);

-- Index for importance scoring
CREATE INDEX idx_kb_entities_importance ON kb_entities(importance_score DESC);

-- Vector similarity index for entity search
CREATE INDEX idx_kb_entities_embedding ON kb_entities USING ivfflat (embedding vector_cosine_ops);

-- Full-text search index
CREATE INDEX idx_kb_entities_name_fts ON kb_entities USING gin(to_tsvector('english', entity_name || ' ' || COALESCE(description, '')));

-- ============================================================================
-- ENTITY MENTIONS TABLE
-- ============================================================================
-- Links entities to knowledge items (many-to-many relationship)
CREATE TABLE IF NOT EXISTS knowledge_entity_mentions (
    id SERIAL PRIMARY KEY,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE,
    entity_id INTEGER REFERENCES kb_entities(id) ON DELETE CASCADE,
    mention_context TEXT,                -- surrounding text where entity was mentioned
    confidence FLOAT DEFAULT 1.0,        -- extraction confidence (0.0-1.0)
    position_in_source INTEGER,          -- character position in source document
    extraction_method VARCHAR(50),       -- 'spacy', 'llm', 'manual'
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for finding all mentions of an entity
CREATE INDEX idx_entity_mentions_entity ON knowledge_entity_mentions(entity_id);

-- Index for finding all entities in a document
CREATE INDEX idx_entity_mentions_knowledge ON knowledge_entity_mentions(knowledge_item_id);

-- Unique constraint to prevent duplicate mentions
CREATE UNIQUE INDEX idx_entity_mentions_unique ON knowledge_entity_mentions(knowledge_item_id, entity_id, position_in_source);

-- ============================================================================
-- ENTITY RELATIONSHIPS TABLE
-- ============================================================================
-- Stores relationships between entities (the "graph" part)
CREATE TABLE IF NOT EXISTS entity_relationships (
    id SERIAL PRIMARY KEY,
    from_entity_id INTEGER REFERENCES kb_entities(id) ON DELETE CASCADE,
    to_entity_id INTEGER REFERENCES kb_entities(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,  -- is_part_of, uses, created_by, related_to, etc.
    strength FLOAT DEFAULT 1.0,               -- relationship strength (0.0-1.0)
    evidence_source_id INTEGER REFERENCES knowledge_items(id) ON DELETE SET NULL,  -- which document supports this
    evidence_text TEXT,                       -- supporting text from document
    bidirectional BOOLEAN DEFAULT false,      -- whether relationship works both ways
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for graph traversal
CREATE INDEX idx_entity_relationships_from ON entity_relationships(from_entity_id);
CREATE INDEX idx_entity_relationships_to ON entity_relationships(to_entity_id);
CREATE INDEX idx_entity_relationships_type ON entity_relationships(relationship_type);
CREATE INDEX idx_entity_relationships_strength ON entity_relationships(strength DESC);

-- Prevent duplicate relationships
CREATE UNIQUE INDEX idx_entity_relationships_unique ON entity_relationships(from_entity_id, to_entity_id, relationship_type);

-- ============================================================================
-- ENTITY CLUSTERS TABLE
-- ============================================================================
-- Stores semantic clusters of related entities for topic discovery
CREATE TABLE IF NOT EXISTS entity_clusters (
    id SERIAL PRIMARY KEY,
    cluster_name VARCHAR(255),
    cluster_topic VARCHAR(500),           -- auto-generated topic description
    entity_ids INTEGER[],                 -- array of entity IDs in this cluster
    size INTEGER DEFAULT 0,               -- number of entities
    coherence_score FLOAT DEFAULT 0.0,    -- how well entities cluster together
    keywords TEXT[],                      -- representative keywords
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index for cluster size
CREATE INDEX idx_entity_clusters_size ON entity_clusters(size DESC);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to update entity mention count
CREATE OR REPLACE FUNCTION update_entity_mention_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE kb_entities 
        SET mention_count = mention_count + 1,
            updated_at = NOW()
        WHERE id = NEW.entity_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE kb_entities 
        SET mention_count = GREATEST(0, mention_count - 1),
            updated_at = NOW()
        WHERE id = OLD.entity_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update mention counts
DROP TRIGGER IF EXISTS trigger_update_entity_mention_count ON knowledge_entity_mentions;
CREATE TRIGGER trigger_update_entity_mention_count
AFTER INSERT OR DELETE ON knowledge_entity_mentions
FOR EACH ROW EXECUTE FUNCTION update_entity_mention_count();

-- Function to calculate entity importance score
CREATE OR REPLACE FUNCTION calculate_entity_importance(entity_id_param INTEGER)
RETURNS FLOAT AS $$
DECLARE
    mention_score FLOAT;
    relationship_score FLOAT;
    centrality_score FLOAT;
    final_score FLOAT;
BEGIN
    -- Score based on mentions (normalized to 0-1)
    SELECT LEAST(1.0, mention_count / 100.0) INTO mention_score
    FROM kb_entities WHERE id = entity_id_param;
    
    -- Score based on number of relationships (normalized to 0-1)
    SELECT LEAST(1.0, COUNT(*) / 50.0) INTO relationship_score
    FROM entity_relationships
    WHERE from_entity_id = entity_id_param OR to_entity_id = entity_id_param;
    
    -- Calculate centrality (how many unique entities it connects to)
    SELECT LEAST(1.0, COUNT(DISTINCT CASE 
        WHEN from_entity_id = entity_id_param THEN to_entity_id 
        ELSE from_entity_id 
    END) / 30.0) INTO centrality_score
    FROM entity_relationships
    WHERE from_entity_id = entity_id_param OR to_entity_id = entity_id_param;
    
    -- Weighted combination
    final_score := (mention_score * 0.4) + (relationship_score * 0.3) + (centrality_score * 0.3);
    
    RETURN final_score;
END;
$$ LANGUAGE plpgsql;

-- Function to update all entity importance scores
CREATE OR REPLACE FUNCTION update_all_entity_importance_scores()
RETURNS void AS $$
BEGIN
    UPDATE kb_entities
    SET importance_score = calculate_entity_importance(id),
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SEED DATA: Entity Types
-- ============================================================================
-- Predefined entity types for classification
INSERT INTO kb_types (type_name, display_name, description, processor_class, supports_embedding, supports_search, enabled)
VALUES 
    ('entity', 'Entity', 'Extracted entities from documents (people, concepts, technologies)', 'EntityProcessor', true, true, true)
ON CONFLICT (type_name) DO NOTHING;

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Most important entities
CREATE OR REPLACE VIEW v_top_entities AS
SELECT 
    e.id,
    e.entity_name,
    e.entity_type,
    e.mention_count,
    e.importance_score,
    COUNT(DISTINCT em.knowledge_item_id) as document_count,
    COUNT(DISTINCT er.id) as relationship_count
FROM kb_entities e
LEFT JOIN knowledge_entity_mentions em ON e.id = em.entity_id
LEFT JOIN entity_relationships er ON e.id = er.from_entity_id OR e.id = er.to_entity_id
GROUP BY e.id, e.entity_name, e.entity_type, e.mention_count, e.importance_score
ORDER BY e.importance_score DESC;

-- View: Entity co-occurrence (entities that appear together frequently)
CREATE OR REPLACE VIEW v_entity_cooccurrence AS
SELECT 
    e1.id as entity1_id,
    e1.entity_name as entity1_name,
    e2.id as entity2_id,
    e2.entity_name as entity2_name,
    COUNT(*) as cooccurrence_count
FROM knowledge_entity_mentions em1
JOIN knowledge_entity_mentions em2 ON em1.knowledge_item_id = em2.knowledge_item_id
JOIN kb_entities e1 ON em1.entity_id = e1.id
JOIN kb_entities e2 ON em2.entity_id = e2.id
WHERE em1.entity_id < em2.entity_id  -- Avoid duplicates
GROUP BY e1.id, e1.entity_name, e2.id, e2.entity_name
HAVING COUNT(*) > 2  -- Only show significant co-occurrences
ORDER BY cooccurrence_count DESC;

-- ============================================================================
-- SAMPLE QUERIES (for testing)
-- ============================================================================

-- Find all documents mentioning an entity
-- SELECT ki.* 
-- FROM knowledge_items ki
-- JOIN knowledge_entity_mentions em ON ki.id = em.knowledge_item_id
-- JOIN kb_entities e ON em.entity_id = e.id
-- WHERE e.entity_name ILIKE '%GPT-4%';

-- Find related entities (1-hop)
-- SELECT e2.entity_name, er.relationship_type, er.strength
-- FROM entity_relationships er
-- JOIN kb_entities e1 ON er.from_entity_id = e1.id
-- JOIN kb_entities e2 ON er.to_entity_id = e2.id
-- WHERE e1.entity_name = 'GPT-4'
-- ORDER BY er.strength DESC;

-- Find connection path between two entities (2-hops)
-- WITH RECURSIVE entity_path AS (
--     SELECT from_entity_id, to_entity_id, relationship_type, 1 as depth, 
--            ARRAY[from_entity_id] as path
--     FROM entity_relationships
--     WHERE from_entity_id = (SELECT id FROM kb_entities WHERE entity_name = 'GPT-4')
--     UNION
--     SELECT er.from_entity_id, er.to_entity_id, er.relationship_type, ep.depth + 1,
--            ep.path || er.from_entity_id
--     FROM entity_relationships er
--     JOIN entity_path ep ON er.from_entity_id = ep.to_entity_id
--     WHERE ep.depth < 3 AND NOT er.from_entity_id = ANY(ep.path)
-- )
-- SELECT * FROM entity_path 
-- WHERE to_entity_id = (SELECT id FROM kb_entities WHERE entity_name = 'Neural Networks');

-- ============================================================================
-- GRANTS (for application user)
-- ============================================================================
-- GRANT ALL PRIVILEGES ON kb_entities TO postgres;
-- GRANT ALL PRIVILEGES ON knowledge_entity_mentions TO postgres;
-- GRANT ALL PRIVILEGES ON entity_relationships TO postgres;
-- GRANT ALL PRIVILEGES ON entity_clusters TO postgres;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- This migration adds comprehensive entity and relationship tracking to the
-- multimodal knowledge base, enabling graph-based queries and visualizations.


