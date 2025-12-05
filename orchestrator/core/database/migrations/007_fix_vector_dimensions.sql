-- Migration 007: Fix Vector Dimensions
-- =====================================
-- 
-- Problem: document_chunks.embedding is vector(1536) but system uses 384-dim embeddings
-- Solution: Change to 1024 dimensions (supports bge-large-en-v1.5)
--
-- IMPORTANT: This will invalidate existing embeddings - documents need re-processing
--
-- Run with: psql -h localhost -U your_user -d your_db -f 007_fix_vector_dimensions.sql

BEGIN;

-- Record migration
INSERT INTO schema_migrations (version, name, applied_at)
VALUES ('007', 'fix_vector_dimensions', NOW())
ON CONFLICT (version) DO NOTHING;

-- ============================================================================
-- Step 1: Drop existing embedding indexes
-- ============================================================================

DROP INDEX IF EXISTS idx_document_chunks_embedding;
DROP INDEX IF EXISTS idx_knowledge_items_embedding;
DROP INDEX IF EXISTS idx_memory_items_embedding;
DROP INDEX IF EXISTS idx_kb_entities_embedding;

-- ============================================================================
-- Step 2: Alter vector dimensions
-- ============================================================================

-- document_chunks: 1536 -> 1024
ALTER TABLE document_chunks 
ALTER COLUMN embedding TYPE vector(1024) 
USING NULL;  -- Set to NULL since dimensions changed

-- knowledge_items: if exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'knowledge_items' AND column_name = 'embedding') THEN
        ALTER TABLE knowledge_items ALTER COLUMN embedding TYPE vector(1024) USING NULL;
    END IF;
END $$;

-- kb_entities: if exists  
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'kb_entities' AND column_name = 'embedding') THEN
        ALTER TABLE kb_entities ALTER COLUMN embedding TYPE vector(1024) USING NULL;
    END IF;
END $$;

-- memory_items is already 384 - leave it for now or upgrade
-- ALTER TABLE memory_items ALTER COLUMN embedding TYPE vector(1024) USING NULL;

-- ============================================================================
-- Step 3: Recreate indexes with HNSW for fast search
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
ON document_chunks USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 256);

-- Only create if table exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'knowledge_items') THEN
        CREATE INDEX IF NOT EXISTS idx_knowledge_items_embedding 
        ON knowledge_items USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 256);
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'kb_entities') THEN
        CREATE INDEX IF NOT EXISTS idx_kb_entities_embedding 
        ON kb_entities USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 256);
    END IF;
END $$;

-- ============================================================================
-- Step 4: Mark all documents for re-processing
-- ============================================================================

UPDATE documents 
SET status = 'pending_reprocess', 
    processed_date = NULL
WHERE status = 'completed';

-- Clear old chunks (they have wrong dimension embeddings)
DELETE FROM document_chunks WHERE embedding IS NULL;

-- ============================================================================
-- Step 5: Add system setting for vector dimension
-- ============================================================================

INSERT INTO system_settings (category, key, value, default_value, description, is_sensitive, is_required)
VALUES (
    'embedding',
    'vector_dimension',
    '1024',
    '1024',
    'Vector embedding dimension (must match embedding model)',
    false,
    true
)
ON CONFLICT (category, key) DO UPDATE SET value = '1024';

COMMIT;

-- ============================================================================
-- Post-migration notes:
-- ============================================================================
-- 1. All existing document embeddings are now NULL
-- 2. Documents marked as 'pending_reprocess'
-- 3. Run document re-processing job to regenerate embeddings
-- 4. Update system settings to use bge-large-en-v1.5 (1024 dims)
-- ============================================================================

