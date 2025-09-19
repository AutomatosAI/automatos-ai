-- PostgreSQL pgvector Extension Initialization
-- This script enables the pgvector extension for vector similarity search

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Log pgvector initialization
DO $$
BEGIN
    RAISE NOTICE 'pgvector extension enabled successfully';
END $$;
