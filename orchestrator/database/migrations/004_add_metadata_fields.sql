-- Migration to add missing metadata fields
-- Run this to fix the MetaData assignment error

-- Add execution_metadata column to workflow_executions table (metadata is reserved in SQLAlchemy)
ALTER TABLE workflow_executions 
ADD COLUMN IF NOT EXISTS execution_metadata JSON DEFAULT '{}';

-- Add models_used column for tracking model usage
ALTER TABLE workflow_executions 
ADD COLUMN IF NOT EXISTS models_used JSON DEFAULT '[]';

-- Add last_execution column to workflows table for quick access (CRITICAL FIX)
ALTER TABLE workflows 
ADD COLUMN IF NOT EXISTS last_execution JSON DEFAULT NULL;

-- Indexes removed - GIN only works with JSONB, not JSON

-- Update any existing executions with empty execution_metadata
UPDATE workflow_executions 
SET execution_metadata = '{}' 
WHERE execution_metadata IS NULL;
