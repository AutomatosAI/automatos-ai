-- Initialize the orchestrator database
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create basic tables if they don't exist
-- This will be handled by Alembic migrations, but we need the extension
