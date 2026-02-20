-- Migration 010: Full RAG Reset — OpenRouter/Qwen3-8B @ 4096 dimensions
-- =============================================================================
--
-- NUCLEAR RESET: Wipes ALL document, chunk, entity, knowledge, and memory data.
-- Vectors stored in S3 (not pgvector) — no ALTER TABLE needed for vector columns.
-- After this migration, S3 index must also be deleted/recreated at 4096d.
--
-- Run: python -m scripts.recreate_s3_index  (handles S3 side)
--
-- Post-migration Railway env vars:
--   S3_VECTORS_DIMENSION=4096
--   EMBEDDING_PROVIDER=openrouter
--   EMBEDDING_MODEL=qwen/qwen3-embedding-8b
--   OPENROUTER_API_KEY=<key>

BEGIN;

-- =============================================================================
-- Step 1: Wipe leaf/junction tables first (FK dependency order)
-- =============================================================================

-- RAG v3 entity graph (migration 20260218)
DO $$ BEGIN TRUNCATE document_relationships CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE document_entities CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE rag_feedback CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;

-- Knowledge graph entities (migration 007)
DO $$ BEGIN TRUNCATE knowledge_entity_mentions CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE entity_relationships CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE entity_clusters CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE kb_entities CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;

-- Knowledge base content (PRD-19)
DO $$ BEGIN TRUNCATE knowledge_collection_items CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE knowledge_collections CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE knowledge_usage CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE knowledge_relationships CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE kb_tables CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE kb_images CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE kb_formulas CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE multimodal_content CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE knowledge_items CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;

-- Knowledge graph nodes/edges
DO $$ BEGIN TRUNCATE knowledge_edges CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE knowledge_nodes CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE external_knowledge CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;

-- Memory
DO $$ BEGIN TRUNCATE memory_items CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;

-- Document analytics
DO $$ BEGIN TRUNCATE document_usage CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;

-- =============================================================================
-- Step 2: Wipe core document tables
-- =============================================================================

TRUNCATE document_chunks CASCADE;
DELETE FROM documents;

-- Cloud sync metadata
DO $$ BEGIN TRUNCATE cloud_sync_jobs CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE cloud_documents CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;
DO $$ BEGIN TRUNCATE cloud_sync_config CASCADE; EXCEPTION WHEN undefined_table THEN NULL; END $$;

-- =============================================================================
-- Step 3: Update system settings for new embedding config
-- =============================================================================

UPDATE system_settings
SET value = '4096', updated_at = NOW()
WHERE category = 'general' AND key = 'vector_store_dimensions';

UPDATE system_settings
SET value = 'openrouter', updated_at = NOW()
WHERE category = 'general' AND key = 'embedding_provider';

UPDATE system_settings
SET value = 'qwen/qwen3-embedding-8b', updated_at = NOW()
WHERE category = 'general' AND key = 'embedding_model';

COMMIT;

SELECT 'Migration 010: Full RAG reset complete' AS status;
SELECT 'NEXT: Run recreate_s3_index.py + update Railway env vars + re-sync' AS action;
