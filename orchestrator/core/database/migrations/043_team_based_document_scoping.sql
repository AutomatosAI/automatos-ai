-- PRD-124: Team-Based Document Scoping
-- Adds team_access array to documents for agent-team visibility control.
-- Adds team field to sdk_api_keys for API key team locking.
-- Adds index on agents.team for fast lookups.
--
-- Filtering rule everywhere:
--   WHERE (team_access = '{}' OR :team IS NULL OR :team = ANY(team_access))

-- 1. Documents: team_access array (empty = visible to all)
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS team_access TEXT[] DEFAULT '{}';

CREATE INDEX IF NOT EXISTS idx_documents_team_access
    ON documents USING GIN (team_access);

-- 2. SDK API keys: optional team lock
ALTER TABLE sdk_api_keys
    ADD COLUMN IF NOT EXISTS team VARCHAR(100);

-- 3. Agents: index on existing team column for fast lookups
CREATE INDEX IF NOT EXISTS idx_agents_team
    ON agents (team) WHERE team IS NOT NULL;
