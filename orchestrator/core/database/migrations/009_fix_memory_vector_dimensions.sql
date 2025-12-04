-- Migration 009: Fix memory_items embedding dimension from 384 to 1024
-- Required for BAAI/bge-large-en-v1.5 model

-- Step 1: Drop indexes on embedding column
DROP INDEX IF EXISTS idx_memory_items_embedding;
DROP INDEX IF EXISTS memory_items_embedding_idx;

-- Step 2: Clear existing memory data (embeddings incompatible)
DELETE FROM memory_items;

-- Step 3: Alter column type
ALTER TABLE memory_items 
ALTER COLUMN embedding TYPE vector(1024);

-- Step 4: Recreate index
CREATE INDEX idx_memory_items_embedding 
ON memory_items USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Done
SELECT 'memory_items.embedding migrated to vector(1024)' as status;

