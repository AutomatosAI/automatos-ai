-- Migration: Fix agent context window limits for PRD-22 skill support
-- Date: 2025-11-07
-- Purpose: Update agents using gpt-4 (8K limit) to gpt-4o (128K limit) to support skill injection

-- Update Technical Writer Pro (Agent ID 96)
UPDATE agents
SET configuration = jsonb_set(
    COALESCE(configuration, '{}'::jsonb),
    '{llm_config}',
    '{
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 4000,
        "context_window": 128000
    }'::jsonb
)
WHERE id = 96
AND name = 'Technical Writer Pro';

-- Update Document Generation (Agent ID 102)
UPDATE agents
SET configuration = jsonb_set(
    COALESCE(configuration, '{}'::jsonb),
    '{llm_config}',
    '{
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 4000,
        "context_window": 128000
    }'::jsonb
)
WHERE id = 102
AND name = 'Document Generation';

-- Update any other agents that might be using the old gpt-4 model
UPDATE agents
SET configuration = jsonb_set(
    COALESCE(configuration, '{}'::jsonb),
    '{llm_config}',
    jsonb_set(
        COALESCE(configuration->'llm_config', '{}'::jsonb),
        '{model}',
        '"gpt-4o"'
    ) || jsonb_set(
        COALESCE(configuration->'llm_config', '{}'::jsonb),
        '{context_window}',
        '128000'
    ) || jsonb_set(
        COALESCE(configuration->'llm_config', '{}'::jsonb),
        '{max_tokens}',
        '4000'
    )
)
WHERE configuration->'llm_config'->>'model' = 'gpt-4'
AND configuration->'llm_config'->>'context_window' = '8192';

-- Verify the changes
SELECT 
    id,
    name,
    agent_type,
    configuration->'llm_config'->>'model' as model,
    configuration->'llm_config'->>'context_window' as context_window,
    configuration->'llm_config'->>'max_tokens' as max_tokens
FROM agents
WHERE id IN (96, 102)
OR configuration->'llm_config'->>'model' = 'gpt-4o';

