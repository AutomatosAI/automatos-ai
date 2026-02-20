-- Migration 010: Upgrade to Qwen3 Embedding 8B (4096 dimensions via OpenRouter)
-- =============================================================================
--
-- Switches embedding provider to OpenRouter with Qwen3-Embedding-8B (4096d).
-- Vectors are stored in S3 Vectors (not pgvector), so no ALTER TABLE needed.
-- This migration only updates system_settings and clears document metadata
-- so all documents get re-embedded at the new dimension.
--
-- S3 INDEX NOTE: The existing S3 index (dim=1024) must be deleted separately
-- before re-syncing. The app will auto-create a new index at 4096d on startup.
-- Use: python -m scripts.recreate_s3_index  (or delete via AWS console)

-- Step 1: Update system settings for new embedding config
-- Schema: system_settings(category, key, value, ...)
UPDATE system_settings
SET value = '4096',
    updated_at = NOW()
WHERE category = 'general' AND key = 'vector_store_dimensions';

UPDATE system_settings
SET value = 'openrouter',
    updated_at = NOW()
WHERE category = 'general' AND key = 'embedding_provider';

UPDATE system_settings
SET value = 'qwen/qwen3-embedding-8b',
    updated_at = NOW()
WHERE category = 'general' AND key = 'embedding_model';

-- Step 2: Clear all document chunks (old 1024d embeddings are useless)
DELETE FROM document_chunks;

-- Step 3: Reset all documents to pending so they get re-processed
UPDATE documents SET status = 'pending', chunk_count = 0, updated_at = NOW();

-- Step 4: Clear entity data (will be re-extracted during re-sync)
DO $$ BEGIN
    DELETE FROM entity_mentions;
EXCEPTION WHEN undefined_table THEN NULL;
END $$;

DO $$ BEGIN
    DELETE FROM entity_relationships;
EXCEPTION WHEN undefined_table THEN NULL;
END $$;

SELECT 'Migration 010 complete: switched to OpenRouter/Qwen3-8B (4096d)' AS status;
SELECT 'ACTION REQUIRED: Delete S3 index + update Railway env vars + re-sync docs' AS next_step;
