-- ================================================================
-- PRD-22: Anthropic Skills Integration - Database Migration
-- ================================================================
-- Git-backed skill loading with progressive disclosure
-- Date: 2025-10-29
-- ================================================================

-- =========================================================================
-- 1. Enhance skills table with new columns for PRD-22
-- =========================================================================

ALTER TABLE skills ADD COLUMN IF NOT EXISTS prompt_template TEXT;
ALTER TABLE skills ADD COLUMN IF NOT EXISTS skill_version VARCHAR(20);
ALTER TABLE skills ADD COLUMN IF NOT EXISTS skill_source VARCHAR(50);
ALTER TABLE skills ADD COLUMN IF NOT EXISTS git_repo_url TEXT;
ALTER TABLE skills ADD COLUMN IF NOT EXISTS git_commit_sha VARCHAR(40);
ALTER TABLE skills ADD COLUMN IF NOT EXISTS git_branch VARCHAR(100);
ALTER TABLE skills ADD COLUMN IF NOT EXISTS filesystem_path TEXT;
ALTER TABLE skills ADD COLUMN IF NOT EXISTS tags JSONB;
ALTER TABLE skills ADD COLUMN IF NOT EXISTS skill_metadata JSONB;
ALTER TABLE skills ADD COLUMN IF NOT EXISTS last_sync_at TIMESTAMP WITH TIME ZONE;

-- Set defaults for existing rows
UPDATE skills 
SET skill_source = 'builtin-seeds',
    skill_version = '1.0.0',
    tags = '[]'::jsonb,
    skill_metadata = '{}'::jsonb
WHERE skill_source IS NULL;

-- =========================================================================
-- 2. Create skill_files table for progressive disclosure
-- =========================================================================

CREATE TABLE IF NOT EXISTS skill_files (
    id SERIAL PRIMARY KEY,
    skill_id INTEGER NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    file_path VARCHAR(500) NOT NULL,  -- Relative path within skill directory
    file_type VARCHAR(50) NOT NULL,   -- core, resource, script, example
    content_summary TEXT,             -- Brief summary for indexing
    file_size_bytes INTEGER,
    estimated_tokens INTEGER,         -- Estimated token count for this file
    load_level INTEGER NOT NULL DEFAULT 3,  -- 1=metadata, 2=core, 3=resource
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_skill_files_skill_id ON skill_files(skill_id);
CREATE INDEX IF NOT EXISTS idx_skill_files_type ON skill_files(file_type);
CREATE INDEX IF NOT EXISTS idx_skill_files_level ON skill_files(load_level);

-- =========================================================================
-- 3. Create skill_sources table for Git repository tracking
-- =========================================================================

CREATE TABLE IF NOT EXISTS skill_sources (
    id SERIAL PRIMARY KEY,
    source_name VARCHAR(100) NOT NULL UNIQUE,  -- Unique identifier for this source
    source_type VARCHAR(50) NOT NULL,          -- git, upload, seed, marketplace
    git_url TEXT,                              -- Git repository URL
    git_branch VARCHAR(100) DEFAULT 'main',
    git_commit_sha VARCHAR(40),                -- Current commit SHA
    local_cache_path TEXT NOT NULL,            -- Path in ~/.automat قب/skills/
    auto_update BOOLEAN DEFAULT FALSE,         -- Auto-update from Git on schedule
    update_frequency_hours INTEGER DEFAULT 24,
    skills_discovered INTEGER DEFAULT 0,       -- Number of skills found in this source
    status VARCHAR(50) DEFAULT 'active',       -- active, error, syncing, inactive
    last_sync_at TIMESTAMP WITH TIME ZONE,
    last_sync_status VARCHAR(50),              -- success, error, partial
    last_sync_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    created_by VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_skill_sources_type ON skill_sources(source_type);
CREATE INDEX IF NOT EXISTS idx_skill_sources_status ON skill_sources(status);
CREATE INDEX IF NOT EXISTS idx_skill_sources_name ON skill_sources(source_name);

-- Insert default skill sources
INSERT INTO skill_sources (source_name, source_type, local_cache_path, skills_discovered, status)
VALUES 
    ('builtin-seeds', 'seed', '~/.automatos/skills/builtin-seeds/', 
     (SELECT COUNT(*) FROM skills WHERE skill_source = 'builtin-seeds' OR skill_source IS NULL), 
     'active'),
    ('anthropic-official', 'git', '~/.automatos/skills/anthropic-official/', 0, 'inactive')
ON CONFLICT (source_name) DO NOTHING;

-- =========================================================================
-- 4. Create skill_versions table for version history
-- =========================================================================

CREATE TABLE IF NOT EXISTS skill_versions (
    id SERIAL PRIMARY KEY,
    skill_id INTEGER NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    version VARCHAR(20) NOT NULL,
    git_commit_sha VARCHAR(40),
    changelog TEXT,
    is_current BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    created_by VARCHAR(255),
    UNIQUE(skill_id, version)
);

CREATE INDEX IF NOT EXISTS idx_skill_versions_skill_id ON skill_versions(skill_id);
CREATE INDEX IF NOT EXISTS idx_skill_versions_current ON skill_versions(is_current);

-- =========================================================================
-- 5. Create skill_audit_log table for security and tracking
-- =========================================================================

CREATE TABLE IF NOT EXISTS skill_audit_log (
    id SERIAL PRIMARY KEY,
    skill_id INTEGER REFERENCES skills(id) ON DELETE SET NULL,
    source_id INTEGER REFERENCES skill_sources(id) ON DELETE SET NULL,
    action VARCHAR(50) NOT NULL,  -- create, update, delete, load, execute, sync
    action_details JSONB,
    user_id VARCHAR(255),
    ip_address VARCHAR(50),
    user_agent TEXT,
    status VARCHAR(50) NOT NULL,  -- success, error, warning
    error_message TEXT,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_skill_audit_action ON skill_audit_log(action);
CREATE INDEX IF NOT EXISTS idx_skill_audit_created ON skill_audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_skill_audit_skill ON skill_audit_log(skill_id);
CREATE INDEX IF NOT EXISTS idx_skill_audit_source ON skill_audit_log(source_id);

-- =========================================================================
-- 6. Add performance indexes
-- =========================================================================

CREATE INDEX IF NOT EXISTS idx_skills_source_active ON skills(skill_source, is_active);

-- GIN index for tags array searching (PostgreSQL specific)
CREATE INDEX IF NOT EXISTS idx_skills_tags_gin ON skills USING GIN (tags jsonb_path_ops);

-- =========================================================================
-- Trigger for updated_at timestamps
-- =========================================================================

CREATE OR REPLACE FUNCTION update_skill_files_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_skill_files_updated_at 
    BEFORE UPDATE ON skill_files 
    FOR EACH ROW EXECUTE FUNCTION update_skill_files_updated_at();

CREATE OR REPLACE FUNCTION update_skill_sources_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_skill_sources_updated_at 
    BEFORE UPDATE ON skill_sources 
    FOR EACH ROW EXECUTE FUNCTION update_skill_sources_updated_at();



