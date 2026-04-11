-- Add missing agent evaluation columns if they don't exist
-- These columns are in the schema but may be missing from existing databases

DO $$ 
BEGIN
    -- Add quality_score if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='agents' AND column_name='quality_score') THEN
        ALTER TABLE agents ADD COLUMN quality_score FLOAT;
    END IF;
    
    -- Add emergence_score if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='agents' AND column_name='emergence_score') THEN
        ALTER TABLE agents ADD COLUMN emergence_score FLOAT;
    END IF;
    
    -- Add performance if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='agents' AND column_name='performance') THEN
        ALTER TABLE agents ADD COLUMN performance FLOAT;
    END IF;
    
    -- Add reliability if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='agents' AND column_name='reliability') THEN
        ALTER TABLE agents ADD COLUMN reliability FLOAT;
    END IF;
    
    -- Add readiness if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='agents' AND column_name='readiness') THEN
        ALTER TABLE agents ADD COLUMN readiness FLOAT;
    END IF;
    
    -- Add coherence if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='agents' AND column_name='coherence') THEN
        ALTER TABLE agents ADD COLUMN coherence FLOAT;
    END IF;
    
    -- Add efficiency if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='agents' AND column_name='efficiency') THEN
        ALTER TABLE agents ADD COLUMN efficiency FLOAT;
    END IF;
    
    -- Add eci if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='agents' AND column_name='eci') THEN
        ALTER TABLE agents ADD COLUMN eci FLOAT;
    END IF;
    
    -- Add validity if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='agents' AND column_name='validity') THEN
        ALTER TABLE agents ADD COLUMN validity FLOAT;
    END IF;
    
    -- Add discriminatory_power if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='agents' AND column_name='discriminatory_power') THEN
        ALTER TABLE agents ADD COLUMN discriminatory_power FLOAT;
    END IF;
    
    -- Add model_config if missing
    -- NOTE: No DEFAULT — agents must get their model_config from either
    -- their creator (e.g. AgentFactory, wizard seeds) or resolve at runtime
    -- via _get_default_llm_config_from_settings. A sticky DEFAULT here
    -- silently shackled Mission Zero agents to gpt-4 (8K ctx). See
    -- drop_agents_model_config_default alembic migration (2026-04-11).
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='agents' AND column_name='model_config') THEN
        ALTER TABLE agents ADD COLUMN model_config JSONB;
    END IF;
    
    -- Add model_usage_stats if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='agents' AND column_name='model_usage_stats') THEN
        ALTER TABLE agents ADD COLUMN model_usage_stats JSONB DEFAULT '{"total_tokens": 0, "total_cost": 0.0}'::jsonb;
    END IF;
END $$;

