-- ================================================================
-- AUTOMATOS AI PLATFORM - DATABASE MIGRATION SCRIPT
-- ================================================================
-- Migration: Add 9-Stage Workflow Enhancement Columns
-- Created: 2025-11-24
-- Purpose: Add missing columns from 9-stage workflow system to existing database
-- Scope: Agents table quality/performance metrics
-- Safe to run multiple times (uses IF NOT EXISTS / ADD COLUMN IF NOT EXISTS)
-- ================================================================

-- ================================================================
-- AGENTS TABLE - 9-Stage Workflow Quality Metrics
-- ================================================================

DO $$
BEGIN
    -- Quality and emergence scores
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'quality_score'
    ) THEN
        ALTER TABLE agents ADD COLUMN quality_score FLOAT;
        RAISE NOTICE 'Added column: agents.quality_score';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'emergence_score'
    ) THEN
        ALTER TABLE agents ADD COLUMN emergence_score FLOAT;
        RAISE NOTICE 'Added column: agents.emergence_score';
    END IF;

    -- Performance metrics
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'performance'
    ) THEN
        ALTER TABLE agents ADD COLUMN performance FLOAT;
        RAISE NOTICE 'Added column: agents.performance';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'reliability'
    ) THEN
        ALTER TABLE agents ADD COLUMN reliability FLOAT;
        RAISE NOTICE 'Added column: agents.reliability';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'readiness'
    ) THEN
        ALTER TABLE agents ADD COLUMN readiness FLOAT;
        RAISE NOTICE 'Added column: agents.readiness';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'coherence'
    ) THEN
        ALTER TABLE agents ADD COLUMN coherence FLOAT;
        RAISE NOTICE 'Added column: agents.coherence';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'efficiency'
    ) THEN
        ALTER TABLE agents ADD COLUMN efficiency FLOAT;
        RAISE NOTICE 'Added column: agents.efficiency';
    END IF;

    -- ECI (Emergent Capability Index)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'eci'
    ) THEN
        ALTER TABLE agents ADD COLUMN eci FLOAT;
        RAISE NOTICE 'Added column: agents.eci';
    END IF;

    -- Validity and discriminatory power
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'validity'
    ) THEN
        ALTER TABLE agents ADD COLUMN validity FLOAT;
        RAISE NOTICE 'Added column: agents.validity';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'discriminatory_power'
    ) THEN
        ALTER TABLE agents ADD COLUMN discriminatory_power FLOAT;
        RAISE NOTICE 'Added column: agents.discriminatory_power';
    END IF;

    -- Model configuration (PRD-15)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'model_config'
    ) THEN
        ALTER TABLE agents ADD COLUMN model_config JSONB DEFAULT '{"provider": "openrouter", "model_id": "google/gemini-2.5-flash", "temperature": 0.7}'::jsonb;
        RAISE NOTICE 'Added column: agents.model_config';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'model_usage_stats'
    ) THEN
        ALTER TABLE agents ADD COLUMN model_usage_stats JSONB DEFAULT '{"total_tokens": 0, "total_cost": 0.0}'::jsonb;
        RAISE NOTICE 'Added column: agents.model_usage_stats';
    END IF;

    -- Priority and concurrency settings
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'priority_level'
    ) THEN
        ALTER TABLE agents ADD COLUMN priority_level VARCHAR(50) DEFAULT 'medium';
        RAISE NOTICE 'Added column: agents.priority_level';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'max_concurrent_tasks'
    ) THEN
        ALTER TABLE agents ADD COLUMN max_concurrent_tasks INTEGER DEFAULT 5;
        RAISE NOTICE 'Added column: agents.max_concurrent_tasks';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'auto_start'
    ) THEN
        ALTER TABLE agents ADD COLUMN auto_start BOOLEAN DEFAULT FALSE;
        RAISE NOTICE 'Added column: agents.auto_start';
    END IF;

    -- Tags column (if not JSONB)
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agents' AND column_name = 'tags'
        AND data_type != 'jsonb'
    ) THEN
        -- Convert existing tags to JSONB if needed
        ALTER TABLE agents ALTER COLUMN tags TYPE JSONB USING tags::jsonb;
        RAISE NOTICE 'Converted agents.tags to JSONB';
    END IF;

END $$;

-- ================================================================
-- WORKFLOWS TABLE - 9-Stage Enhancement Fields
-- ================================================================

DO $$
BEGIN
    -- Goal field (separate from description)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'workflows' AND column_name = 'goal'
    ) THEN
        ALTER TABLE workflows ADD COLUMN goal TEXT;
        RAISE NOTICE 'Added column: workflows.goal';
    END IF;

    -- Context field
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'workflows' AND column_name = 'context'
    ) THEN
        ALTER TABLE workflows ADD COLUMN context TEXT;
        RAISE NOTICE 'Added column: workflows.context';
    END IF;

    -- Last execution summary
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'workflows' AND column_name = 'last_execution'
    ) THEN
        ALTER TABLE workflows ADD COLUMN last_execution JSONB;
        RAISE NOTICE 'Added column: workflows.last_execution';
    END IF;

    -- Workflow metrics
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'workflows' AND column_name = 'priority'
    ) THEN
        ALTER TABLE workflows ADD COLUMN priority VARCHAR(50);
        RAISE NOTICE 'Added column: workflows.priority';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'workflows' AND column_name = 'expected_duration'
    ) THEN
        ALTER TABLE workflows ADD COLUMN expected_duration INTEGER;
        RAISE NOTICE 'Added column: workflows.expected_duration';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'workflows' AND column_name = 'complexity_score'
    ) THEN
        ALTER TABLE workflows ADD COLUMN complexity_score FLOAT;
        RAISE NOTICE 'Added column: workflows.complexity_score';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'workflows' AND column_name = 'success_rate'
    ) THEN
        ALTER TABLE workflows ADD COLUMN success_rate FLOAT;
        RAISE NOTICE 'Added column: workflows.success_rate';
    END IF;

    -- Default policy ID
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'workflows' AND column_name = 'default_policy_id'
    ) THEN
        ALTER TABLE workflows ADD COLUMN default_policy_id VARCHAR(128);
        RAISE NOTICE 'Added column: workflows.default_policy_id';
    END IF;

    -- Ensure tags is JSONB
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'workflows' AND column_name = 'tags'
    ) THEN
        ALTER TABLE workflows ADD COLUMN tags JSONB DEFAULT '[]'::jsonb;
        RAISE NOTICE 'Added column: workflows.tags';
    ELSIF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'workflows' AND column_name = 'tags'
        AND data_type != 'jsonb'
    ) THEN
        ALTER TABLE workflows ALTER COLUMN tags TYPE JSONB USING tags::jsonb;
        RAISE NOTICE 'Converted workflows.tags to JSONB';
    END IF;

END $$;

-- ================================================================
-- WORKFLOW_EXECUTIONS TABLE - Model Tracking
-- ================================================================

DO $$
BEGIN
    -- Track which models were used in execution
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'workflow_executions' AND column_name = 'models_used'
    ) THEN
        ALTER TABLE workflow_executions ADD COLUMN models_used JSONB DEFAULT '[]'::jsonb;
        RAISE NOTICE 'Added column: workflow_executions.models_used';
    END IF;

    -- Execution metadata
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'workflow_executions' AND column_name = 'execution_metadata'
    ) THEN
        ALTER TABLE workflow_executions ADD COLUMN execution_metadata JSON DEFAULT '{}'::json;
        RAISE NOTICE 'Added column: workflow_executions.execution_metadata';
    END IF;

END $$;

-- ================================================================
-- CREATE INDEXES for Performance (if not exist)
-- ================================================================

CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(agent_type);
CREATE INDEX IF NOT EXISTS idx_agents_model_config ON agents USING GIN (model_config);
CREATE INDEX IF NOT EXISTS idx_agents_quality_score ON agents(quality_score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_agents_eci ON agents(eci DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS idx_workflows_owner ON workflows(owner);
CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflows_tags ON workflows USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_workflows_complexity ON workflows(complexity_score DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(status);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_models ON workflow_executions USING GIN (models_used);

-- ================================================================
-- VERIFICATION QUERIES
-- ================================================================

-- Show what was added
DO $$
DECLARE
    rec RECORD;
BEGIN
    RAISE NOTICE '================================================';
    RAISE NOTICE 'MIGRATION COMPLETE - Verification Results:';
    RAISE NOTICE '================================================';
    
    -- Count agents columns
    SELECT COUNT(*) as col_count INTO rec
    FROM information_schema.columns 
    WHERE table_name = 'agents';
    RAISE NOTICE 'Agents table has % columns', rec.col_count;
    
    -- Count workflows columns
    SELECT COUNT(*) as col_count INTO rec
    FROM information_schema.columns 
    WHERE table_name = 'workflows';
    RAISE NOTICE 'Workflows table has % columns', rec.col_count;
    
    -- Count workflow_executions columns
    SELECT COUNT(*) as col_count INTO rec
    FROM information_schema.columns 
    WHERE table_name = 'workflow_executions';
    RAISE NOTICE 'Workflow_executions table has % columns', rec.col_count;
    
    RAISE NOTICE '================================================';
    RAISE NOTICE 'Run the following query to verify new columns:';
    RAISE NOTICE 'SELECT column_name FROM information_schema.columns WHERE table_name = ''agents'' ORDER BY ordinal_position;';
    RAISE NOTICE '================================================';
END $$;

-- ================================================================
-- END OF MIGRATION
-- ================================================================
