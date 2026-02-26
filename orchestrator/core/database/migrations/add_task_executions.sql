-- ================================================================
-- Migration: Add task_executions table
-- PRD-56: Infrastructure Scaling & Physical Workspaces (Phase 2)
-- Date: 2026-02-25
-- ================================================================

CREATE TABLE IF NOT EXISTS task_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id),

    -- Task definition
    task_type VARCHAR(50) NOT NULL,
    agent_id INTEGER REFERENCES agents(id),
    prompt TEXT,
    configuration JSONB DEFAULT '{}',

    -- Execution metadata
    priority VARCHAR(20) DEFAULT 'normal',
    runner_backend VARCHAR(20) NOT NULL DEFAULT 'local',

    -- Resource tracking
    resources_requested JSONB DEFAULT '{}',
    resources_used JSONB DEFAULT '{}',

    -- Lifecycle
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Results
    result JSONB,
    error_message TEXT,
    tokens_used INTEGER DEFAULT 0,
    execution_time_ms INTEGER DEFAULT 0,

    -- Tracing
    parent_execution_id UUID REFERENCES task_executions(id),
    correlation_id VARCHAR(255),

    -- K8s metadata (Phase 3)
    k8s_namespace VARCHAR(255),
    k8s_job_name VARCHAR(255),
    k8s_pod_name VARCHAR(255),

    -- Worker metadata (Phase 2)
    worker_id VARCHAR(255),
    workspace_path TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_task_executions_workspace ON task_executions(workspace_id);
CREATE INDEX IF NOT EXISTS idx_task_executions_status ON task_executions(status);
CREATE INDEX IF NOT EXISTS idx_task_executions_correlation ON task_executions(correlation_id);
CREATE INDEX IF NOT EXISTS idx_task_executions_submitted ON task_executions(submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_task_executions_type_status ON task_executions(task_type, status);

-- Record migration
INSERT INTO schema_versions (version, description)
VALUES ('1.7.0', 'PRD-56: Add task_executions table for workspace worker task tracking')
ON CONFLICT DO NOTHING;
