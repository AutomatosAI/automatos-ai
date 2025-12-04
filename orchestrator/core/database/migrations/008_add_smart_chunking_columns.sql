-- Migration 008: Add Smart Chunking Columns
-- Adds parent_content and headers for hierarchical chunking

-- Add parent_content column (stores the full parent section for context expansion)
ALTER TABLE document_chunks 
ADD COLUMN IF NOT EXISTS parent_content TEXT;

-- Add headers column (stores markdown header hierarchy as JSON)
ALTER TABLE document_chunks 
ADD COLUMN IF NOT EXISTS headers JSONB DEFAULT '{}';

-- Add chunk_type column (child, parent, standalone)
ALTER TABLE document_chunks 
ADD COLUMN IF NOT EXISTS chunk_type VARCHAR(20) DEFAULT 'child';

-- Create index for header search
CREATE INDEX IF NOT EXISTS idx_document_chunks_headers 
ON document_chunks USING gin(headers);

-- Create full-text search index for BM25-style keyword search
CREATE INDEX IF NOT EXISTS idx_document_chunks_content_fts 
ON document_chunks USING gin(to_tsvector('english', content));

-- Log migration
DO $$
BEGIN
    RAISE NOTICE 'Migration 008: Smart chunking columns added successfully';
END $$;

