-- PRD-18: Fix - Add credential_id column to agent_tool_assignments
-- This is a standalone fix for the column that didn't get created

BEGIN;

-- Add credential_id column to agent_tool_assignments if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agent_tool_assignments' 
        AND column_name = 'credential_id'
    ) THEN
        ALTER TABLE agent_tool_assignments 
        ADD COLUMN credential_id INTEGER REFERENCES credentials(id) ON DELETE SET NULL;
        
        CREATE INDEX IF NOT EXISTS idx_agent_tool_assignments_credential_id 
        ON agent_tool_assignments(credential_id);
        
        RAISE NOTICE 'Added credential_id column to agent_tool_assignments';
    ELSE
        RAISE NOTICE 'credential_id column already exists';
    END IF;
END $$;

COMMIT;

