-- Migration: Add tags column to agents table
-- Description: Introduces lightweight capability tags for agents to support semantic matching.

ALTER TABLE agents
    ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]'::jsonb;

COMMENT ON COLUMN agents.tags IS 'Lightweight capability tags used for semantic matching';

