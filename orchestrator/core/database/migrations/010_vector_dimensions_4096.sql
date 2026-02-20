-- Migration 010: Upgrade vector dimensions to 4096 for Qwen3 Embedding 8B
-- ==========================================================================
--
-- Qwen3-Embedding-8B natively outputs 4096 dimensions.
-- Truncating to 1024 loses ~75% of the embedding information.
-- This migration widens all vector columns to 4096d for maximum quality.
--
-- IMPORTANT: All existing embeddings must be re-generated after this migration
-- since they were created at the old dimension. Run a full re-sync after applying.

-- Step 1: Drop existing vector indexes (they reference the old dimension)
DROP INDEX IF EXISTS idx_document_chunks_embedding;
DROP INDEX IF EXISTS idx_document_chunks_embedding_hnsw;
DROP INDEX IF EXISTS idx_knowledge_items_embedding;
DROP INDEX IF EXISTS idx_memory_items_embedding;
DROP INDEX IF EXISTS idx_knowledge_nodes_embedding;
DROP INDEX IF EXISTS idx_kb_entities_embedding;

-- Step 2: Alter vector columns to 4096 dimensions
-- document_chunks (primary RAG storage)
ALTER TABLE document_chunks
    ALTER COLUMN embedding TYPE vector(4096)
    USING NULL;

-- knowledge_items (multimodal knowledge)
ALTER TABLE knowledge_items
    ALTER COLUMN embedding TYPE vector(4096)
    USING NULL;

-- memory_items (conversation memory)
ALTER TABLE memory_items
    ALTER COLUMN embedding TYPE vector(4096)
    USING NULL;

-- knowledge_nodes (graph nodes)
DO $$ BEGIN
    ALTER TABLE knowledge_nodes
        ALTER COLUMN embedding TYPE vector(4096)
        USING NULL;
EXCEPTION WHEN undefined_table THEN NULL;
END $$;

-- kb_entities
DO $$ BEGIN
    ALTER TABLE kb_entities
        ALTER COLUMN embedding TYPE vector(4096)
        USING NULL;
EXCEPTION WHEN undefined_table THEN NULL;
END $$;

-- Step 3: Recreate indexes at new dimension
-- HNSW index for document_chunks (best for recall quality)
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- IVFFlat indexes for other tables
CREATE INDEX IF NOT EXISTS idx_knowledge_items_embedding
    ON knowledge_items USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_memory_items_embedding
    ON memory_items USING ivfflat (embedding vector_cosine_ops);

-- Step 4: Update system settings to reflect new dimension
UPDATE system_settings
SET value = '4096',
    default_value = '4096'
WHERE key = 'vector_store_dimensions';

-- Update embedding provider and model defaults
UPDATE system_settings
SET value = 'openrouter',
    default_value = 'openrouter'
WHERE key = 'embedding_provider';

UPDATE system_settings
SET value = 'qwen/qwen3-embedding-8b',
    default_value = 'qwen/qwen3-embedding-8b'
WHERE key = 'embedding_model';

-- Step 5: Clear all existing embeddings (they're the wrong dimension now)
-- Documents remain but need re-processing
UPDATE documents SET status = 'pending', chunk_count = 0;
DELETE FROM document_chunks;

SELECT 'Migration 010 complete: vector dimensions upgraded to 4096' AS status;
SELECT 'ACTION REQUIRED: Re-sync all documents to generate new embeddings' AS next_step;
