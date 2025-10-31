-- Fix credential_audit_logs table schema
-- Add missing created_at column

-- Check if created_at column exists, if not add it
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'credential_audit_logs' 
        AND column_name = 'created_at'
    ) THEN
        ALTER TABLE credential_audit_logs ADD COLUMN created_at TIMESTAMP DEFAULT NOW();
    END IF;
END $$;

-- Update existing records to have created_at = timestamp
UPDATE credential_audit_logs 
SET created_at = timestamp 
WHERE created_at IS NULL;

-- Make created_at NOT NULL
ALTER TABLE credential_audit_logs ALTER COLUMN created_at SET NOT NULL;
