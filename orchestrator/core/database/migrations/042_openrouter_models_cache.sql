-- Migration 042: OpenRouter models cache
-- Creates local cache tables for the OpenRouter model catalog.
-- Pattern mirrors composio_apps_cache for consistency.

-- ==========================================================
-- openrouter_models_cache
-- ==========================================================

CREATE TABLE IF NOT EXISTS openrouter_models_cache (
    id              SERIAL PRIMARY KEY,
    model_id        VARCHAR(255) NOT NULL UNIQUE,
    slug            VARCHAR(255),
    display_name    VARCHAR(255) NOT NULL,
    description     TEXT,
    provider        VARCHAR(100) NOT NULL,

    -- Pricing (per token, stored as float)
    prompt_cost         DOUBLE PRECISION DEFAULT 0,
    completion_cost     DOUBLE PRECISION DEFAULT 0,
    image_cost          DOUBLE PRECISION DEFAULT 0,
    request_cost        DOUBLE PRECISION DEFAULT 0,
    cache_read_cost     DOUBLE PRECISION DEFAULT 0,
    cache_write_cost    DOUBLE PRECISION DEFAULT 0,

    -- Sizing
    context_length          INTEGER DEFAULT 0,
    max_completion_tokens   INTEGER DEFAULT 0,

    -- Architecture
    modality            VARCHAR(100),
    input_modalities    JSONB DEFAULT '[]'::jsonb,
    output_modalities   JSONB DEFAULT '[]'::jsonb,
    tokenizer           VARCHAR(100),

    -- Capabilities
    supports_tools      BOOLEAN DEFAULT false,
    supports_vision     BOOLEAN DEFAULT false,
    supports_streaming  BOOLEAN DEFAULT true,
    supports_json_mode  BOOLEAN DEFAULT false,
    supports_reasoning  BOOLEAN DEFAULT false,
    supported_parameters JSONB DEFAULT '[]'::jsonb,

    -- Provider info
    top_provider_context    INTEGER DEFAULT 0,
    top_provider_max_tokens INTEGER DEFAULT 0,
    is_moderated            BOOLEAN DEFAULT false,

    -- Categorization
    category    VARCHAR(50),
    tier        VARCHAR(50),
    tags        JSONB DEFAULT '[]'::jsonb,

    -- Status
    status              VARCHAR(50) DEFAULT 'active',
    created_timestamp   INTEGER,

    -- Raw data (full OpenRouter response)
    metadata    JSONB DEFAULT '{}'::jsonb,

    -- Sync timestamps
    last_synced_at  TIMESTAMP DEFAULT NOW(),
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_or_cache_provider ON openrouter_models_cache(provider);
CREATE INDEX IF NOT EXISTS idx_or_cache_category ON openrouter_models_cache(category);
CREATE INDEX IF NOT EXISTS idx_or_cache_tier ON openrouter_models_cache(tier);
CREATE INDEX IF NOT EXISTS idx_or_cache_status ON openrouter_models_cache(status);
CREATE INDEX IF NOT EXISTS idx_or_cache_context ON openrouter_models_cache(context_length);
CREATE INDEX IF NOT EXISTS idx_or_cache_cost ON openrouter_models_cache(prompt_cost);

-- ==========================================================
-- openrouter_sync_jobs
-- ==========================================================

CREATE TABLE IF NOT EXISTS openrouter_sync_jobs (
    id              SERIAL PRIMARY KEY,
    job_type        VARCHAR(50) NOT NULL,
    status          VARCHAR(50) NOT NULL DEFAULT 'pending',
    models_synced   INTEGER DEFAULT 0,
    models_added    INTEGER DEFAULT 0,
    models_updated  INTEGER DEFAULT 0,
    errors_count    INTEGER DEFAULT 0,
    error_details   JSONB,
    started_at      TIMESTAMP DEFAULT NOW(),
    completed_at    TIMESTAMP,
    duration_ms     INTEGER,
    metadata        JSONB DEFAULT '{}'::jsonb
);
