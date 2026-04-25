-- Schema-only snapshot
-- Captured at: 2026-04-25T12:00:38.605635+00:00
-- Tables: 15
-- Source: pg_catalog (Railway live)

BEGIN;

-- ============================================
-- Table: dashboard_configs
-- ============================================
CREATE TABLE IF NOT EXISTS public."dashboard_configs" (
    "id" INTEGER NOT NULL DEFAULT nextval('dashboard_configs_id_seq'::regclass),
    "user_id" INTEGER,
    "config_name" VARCHAR(255),
    "layout" JSONB,
    "widgets" JSONB,
    "refresh_rate" INTEGER,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."dashboard_configs" ADD CONSTRAINT "dashboard_configs_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX dashboard_configs_pkey ON public.dashboard_configs USING btree (id);

-- ============================================
-- Table: custom_metrics
-- ============================================
CREATE TABLE IF NOT EXISTS public."custom_metrics" (
    "id" INTEGER NOT NULL DEFAULT nextval('custom_metrics_id_seq'::regclass),
    "metric_name" VARCHAR(255),
    "calculation_query" TEXT,
    "visualization_type" VARCHAR(50),
    "refresh_interval" INTEGER,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."custom_metrics" ADD CONSTRAINT "custom_metrics_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX custom_metrics_pkey ON public.custom_metrics USING btree (id);

-- ============================================
-- Table: alert_configs
-- ============================================
CREATE TABLE IF NOT EXISTS public."alert_configs" (
    "id" INTEGER NOT NULL DEFAULT nextval('alert_configs_id_seq'::regclass),
    "metric_type" VARCHAR(100),
    "threshold_value" DOUBLE PRECISION,
    "comparison_operator" VARCHAR(10),
    "alert_channel" VARCHAR(50),
    "is_active" BOOLEAN DEFAULT true,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."alert_configs" ADD CONSTRAINT "alert_configs_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX alert_configs_pkey ON public.alert_configs USING btree (id);

-- ============================================
-- Table: compliance_events
-- ============================================
CREATE TABLE IF NOT EXISTS public."compliance_events" (
    "id" INTEGER NOT NULL DEFAULT nextval('compliance_events_id_seq'::regclass),
    "workflow_id" INTEGER,
    "execution_id" INTEGER,
    "event_type" VARCHAR(50) NOT NULL,
    "status" VARCHAR(50) NOT NULL,
    "actor" VARCHAR(255),
    "notes" TEXT,
    "event_metadata" JSONB DEFAULT '{}'::jsonb,
    "created_at" TIMESTAMP DEFAULT now(),
    "updated_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."compliance_events" ADD CONSTRAINT "compliance_events_execution_id_fkey" FOREIGN KEY (execution_id) REFERENCES workflow_executions(id) ON DELETE CASCADE;
ALTER TABLE public."compliance_events" ADD CONSTRAINT "compliance_events_workflow_id_fkey" FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE;
ALTER TABLE public."compliance_events" ADD CONSTRAINT "compliance_events_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX compliance_events_pkey ON public.compliance_events USING btree (id);

-- ============================================
-- Table: marketplace_submissions
-- ============================================
CREATE TABLE IF NOT EXISTS public."marketplace_submissions" (
    "id" INTEGER NOT NULL DEFAULT nextval('marketplace_submissions_id_seq'::regclass),
    "item_id" INTEGER NOT NULL,
    "submitted_by" INTEGER NOT NULL,
    "status" VARCHAR(50) NOT NULL DEFAULT 'pending'::character varying,
    "reviewed_by" INTEGER,
    "reviewed_at" TIMESTAMP,
    "rejection_reason" TEXT,
    "submitted_at" TIMESTAMP NOT NULL DEFAULT now()
);

ALTER TABLE public."marketplace_submissions" ADD CONSTRAINT "valid_submission_status" CHECK (((status)::text = ANY ((ARRAY['pending'::character varying, 'approved'::character varying, 'rejected'::character varying])::text[])));
ALTER TABLE public."marketplace_submissions" ADD CONSTRAINT "marketplace_submissions_item_id_fkey" FOREIGN KEY (item_id) REFERENCES marketplace_items(id);
ALTER TABLE public."marketplace_submissions" ADD CONSTRAINT "marketplace_submissions_reviewed_by_fkey" FOREIGN KEY (reviewed_by) REFERENCES users(id);
ALTER TABLE public."marketplace_submissions" ADD CONSTRAINT "marketplace_submissions_submitted_by_fkey" FOREIGN KEY (submitted_by) REFERENCES users(id);
ALTER TABLE public."marketplace_submissions" ADD CONSTRAINT "marketplace_submissions_pkey" PRIMARY KEY (id);
CREATE INDEX idx_marketplace_submissions_status ON public.marketplace_submissions USING btree (status, submitted_at);
CREATE INDEX ix_marketplace_submissions_item_id ON public.marketplace_submissions USING btree (item_id);
CREATE INDEX ix_marketplace_submissions_status ON public.marketplace_submissions USING btree (status);
CREATE UNIQUE INDEX marketplace_submissions_pkey ON public.marketplace_submissions USING btree (id);

-- ============================================
-- Table: knowledge_collections
-- ============================================
CREATE TABLE IF NOT EXISTS public."knowledge_collections" (
    "id" INTEGER NOT NULL DEFAULT nextval('knowledge_collections_id_seq'::regclass),
    "name" VARCHAR(255) NOT NULL,
    "description" TEXT,
    "kb_type_id" INTEGER,
    "icon" VARCHAR(50),
    "color" VARCHAR(50),
    "visibility" VARCHAR(50) DEFAULT 'private'::character varying,
    "owner_id" VARCHAR(255),
    "metadata" JSONB DEFAULT '{}'::jsonb,
    "created_at" TIMESTAMP DEFAULT now(),
    "updated_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."knowledge_collections" ADD CONSTRAINT "knowledge_collections_kb_type_id_fkey" FOREIGN KEY (kb_type_id) REFERENCES kb_types(id);
ALTER TABLE public."knowledge_collections" ADD CONSTRAINT "knowledge_collections_pkey" PRIMARY KEY (id);
CREATE INDEX idx_knowledge_collections_owner ON public.knowledge_collections USING btree (owner_id);
CREATE UNIQUE INDEX knowledge_collections_pkey ON public.knowledge_collections USING btree (id);

-- ============================================
-- Table: knowledge_collection_items
-- ============================================
CREATE TABLE IF NOT EXISTS public."knowledge_collection_items" (
    "collection_id" INTEGER NOT NULL,
    "knowledge_item_id" INTEGER NOT NULL,
    "position" INTEGER,
    "added_at" TIMESTAMP DEFAULT now(),
    "added_by" VARCHAR(255)
);

ALTER TABLE public."knowledge_collection_items" ADD CONSTRAINT "knowledge_collection_items_collection_id_fkey" FOREIGN KEY (collection_id) REFERENCES knowledge_collections(id) ON DELETE CASCADE;
ALTER TABLE public."knowledge_collection_items" ADD CONSTRAINT "knowledge_collection_items_knowledge_item_id_fkey" FOREIGN KEY (knowledge_item_id) REFERENCES knowledge_items(id) ON DELETE CASCADE;
ALTER TABLE public."knowledge_collection_items" ADD CONSTRAINT "knowledge_collection_items_pkey" PRIMARY KEY (collection_id, knowledge_item_id);
CREATE INDEX idx_collection_items_collection ON public.knowledge_collection_items USING btree (collection_id);
CREATE INDEX idx_collection_items_item ON public.knowledge_collection_items USING btree (knowledge_item_id);
CREATE UNIQUE INDEX knowledge_collection_items_pkey ON public.knowledge_collection_items USING btree (collection_id, knowledge_item_id);

-- ============================================
-- Table: knowledge_usage
-- ============================================
CREATE TABLE IF NOT EXISTS public."knowledge_usage" (
    "id" INTEGER NOT NULL DEFAULT nextval('knowledge_usage_id_seq'::regclass),
    "knowledge_item_id" INTEGER,
    "event_type" VARCHAR(50) NOT NULL,
    "context_type" VARCHAR(100),
    "query_text" TEXT,
    "relevance_score" DOUBLE PRECISION,
    "user_rating" INTEGER,
    "user_id" VARCHAR(255),
    "session_id" VARCHAR(255),
    "metadata" JSONB DEFAULT '{}'::jsonb,
    "timestamp" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."knowledge_usage" ADD CONSTRAINT "knowledge_usage_knowledge_item_id_fkey" FOREIGN KEY (knowledge_item_id) REFERENCES knowledge_items(id) ON DELETE CASCADE;
ALTER TABLE public."knowledge_usage" ADD CONSTRAINT "knowledge_usage_pkey" PRIMARY KEY (id);
CREATE INDEX idx_knowledge_usage_event ON public.knowledge_usage USING btree (event_type);
CREATE INDEX idx_knowledge_usage_item ON public.knowledge_usage USING btree (knowledge_item_id);
CREATE INDEX idx_knowledge_usage_timestamp ON public.knowledge_usage USING btree ("timestamp" DESC);
CREATE UNIQUE INDEX knowledge_usage_pkey ON public.knowledge_usage USING btree (id);

-- ============================================
-- Table: usage_logs
-- ============================================
CREATE TABLE IF NOT EXISTS public."usage_logs" (
    "id" INTEGER NOT NULL DEFAULT nextval('usage_logs_id_seq'::regclass),
    "workspace_id" UUID NOT NULL,
    "user_id" INTEGER,
    "metric_type" VARCHAR(50) NOT NULL,
    "quantity" INTEGER NOT NULL DEFAULT 1,
    "model" VARCHAR(100),
    "metadata" JSONB DEFAULT '{}'::jsonb,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."usage_logs" ADD CONSTRAINT "usage_logs_metric_type_check" CHECK (((metric_type)::text = ANY ((ARRAY['llm_tokens_input'::character varying, 'llm_tokens_output'::character varying, 'tool_call'::character varying, 'agent_run'::character varying, 'workflow_run'::character varying, 'document_upload'::character varying, 'composio_action'::character varying])::text[])));
ALTER TABLE public."usage_logs" ADD CONSTRAINT "usage_logs_user_id_fkey" FOREIGN KEY (user_id) REFERENCES users(id);
ALTER TABLE public."usage_logs" ADD CONSTRAINT "usage_logs_workspace_id_fkey" FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE;
ALTER TABLE public."usage_logs" ADD CONSTRAINT "usage_logs_pkey" PRIMARY KEY (id);
CREATE INDEX idx_usage_created ON public.usage_logs USING btree (created_at DESC);
CREATE INDEX idx_usage_type ON public.usage_logs USING btree (metric_type);
CREATE INDEX idx_usage_workspace ON public.usage_logs USING btree (workspace_id);
CREATE INDEX idx_usage_workspace_date ON public.usage_logs USING btree (workspace_id, date(created_at));
CREATE UNIQUE INDEX usage_logs_pkey ON public.usage_logs USING btree (id);

-- ============================================
-- Table: usage_summary
-- ============================================
CREATE TABLE IF NOT EXISTS public."usage_summary" (
    "id" INTEGER NOT NULL DEFAULT nextval('usage_summary_id_seq'::regclass),
    "workspace_id" UUID NOT NULL,
    "date" DATE NOT NULL,
    "metric_type" VARCHAR(50) NOT NULL,
    "total_quantity" INTEGER NOT NULL DEFAULT 0
);

ALTER TABLE public."usage_summary" ADD CONSTRAINT "usage_summary_workspace_id_fkey" FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE;
ALTER TABLE public."usage_summary" ADD CONSTRAINT "usage_summary_pkey" PRIMARY KEY (id);
ALTER TABLE public."usage_summary" ADD CONSTRAINT "usage_summary_workspace_id_date_metric_type_key" UNIQUE (workspace_id, date, metric_type);
CREATE INDEX idx_usage_summary_workspace ON public.usage_summary USING btree (workspace_id, date);
CREATE UNIQUE INDEX usage_summary_pkey ON public.usage_summary USING btree (id);
CREATE UNIQUE INDEX usage_summary_workspace_id_date_metric_type_key ON public.usage_summary USING btree (workspace_id, date, metric_type);

-- ============================================
-- Table: search_analytics
-- ============================================
CREATE TABLE IF NOT EXISTS public."search_analytics" (
    "id" INTEGER NOT NULL DEFAULT nextval('search_analytics_id_seq'::regclass),
    "query_text" TEXT,
    "query_embedding" vector,
    "results_count" INTEGER,
    "search_mode" TEXT,
    "ranking_strategy" TEXT,
    "execution_time_ms" INTEGER,
    "timestamp" TIMESTAMP WITH TIME ZONE DEFAULT now(),
    "user_feedback" DOUBLE PRECISION
);

ALTER TABLE public."search_analytics" ADD CONSTRAINT "search_analytics_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX search_analytics_pkey ON public.search_analytics USING btree (id);
CREATE INDEX search_analytics_timestamp_idx ON public.search_analytics USING btree ("timestamp" DESC);

-- ============================================
-- Table: execution_contexts
-- ============================================
CREATE TABLE IF NOT EXISTS public."execution_contexts" (
    "id" INTEGER NOT NULL DEFAULT nextval('execution_contexts_id_seq'::regclass),
    "workflow_id" INTEGER,
    "context_data" JSONB,
    "field_state" JSONB,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."execution_contexts" ADD CONSTRAINT "execution_contexts_workflow_id_fkey" FOREIGN KEY (workflow_id) REFERENCES workflows(id);
ALTER TABLE public."execution_contexts" ADD CONSTRAINT "execution_contexts_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX execution_contexts_pkey ON public.execution_contexts USING btree (id);

-- ============================================
-- Table: integration_analysis
-- ============================================
CREATE TABLE IF NOT EXISTS public."integration_analysis" (
    "id" INTEGER NOT NULL DEFAULT nextval('integration_analysis_id_seq'::regclass),
    "integration_name" VARCHAR(255) NOT NULL,
    "status" VARCHAR(50) NOT NULL,
    "analysis_data" JSON DEFAULT '{}'::json,
    "timestamp" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."integration_analysis" ADD CONSTRAINT "integration_analysis_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX integration_analysis_pkey ON public.integration_analysis USING btree (id);

-- ============================================
-- Table: workspace_shares
-- ============================================
CREATE TABLE IF NOT EXISTS public."workspace_shares" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "workspace_id" UUID NOT NULL,
    "user_id" INTEGER NOT NULL,
    "permission" VARCHAR(20) NOT NULL DEFAULT 'view'::character varying,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public."workspace_shares" ADD CONSTRAINT "ck_workspace_shares_permission" CHECK (((permission)::text = ANY ((ARRAY['view'::character varying, 'edit'::character varying, 'admin'::character varying])::text[])));
ALTER TABLE public."workspace_shares" ADD CONSTRAINT "workspace_shares_user_id_fkey" FOREIGN KEY (user_id) REFERENCES users(id);
ALTER TABLE public."workspace_shares" ADD CONSTRAINT "workspace_shares_workspace_id_fkey" FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE;
ALTER TABLE public."workspace_shares" ADD CONSTRAINT "workspace_shares_pkey" PRIMARY KEY (id);
ALTER TABLE public."workspace_shares" ADD CONSTRAINT "uq_workspace_shares_workspace_user" UNIQUE (workspace_id, user_id);
CREATE INDEX idx_workspace_shares_user ON public.workspace_shares USING btree (user_id);
CREATE UNIQUE INDEX uq_workspace_shares_workspace_user ON public.workspace_shares USING btree (workspace_id, user_id);
CREATE UNIQUE INDEX workspace_shares_pkey ON public.workspace_shares USING btree (id);

-- ============================================
-- Table: api_keys
-- ============================================
CREATE TABLE IF NOT EXISTS public."api_keys" (
    "id" INTEGER NOT NULL DEFAULT nextval('api_keys_id_seq'::regclass),
    "workspace_id" UUID NOT NULL,
    "user_id" INTEGER NOT NULL,
    "name" VARCHAR(255) NOT NULL,
    "key_hash" VARCHAR(255) NOT NULL,
    "key_prefix" VARCHAR(12) NOT NULL,
    "scopes" JSONB DEFAULT '["read", "write"]'::jsonb,
    "last_used_at" TIMESTAMP,
    "expires_at" TIMESTAMP,
    "is_active" BOOLEAN DEFAULT true,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."api_keys" ADD CONSTRAINT "api_keys_user_id_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE public."api_keys" ADD CONSTRAINT "api_keys_workspace_id_fkey" FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE;
ALTER TABLE public."api_keys" ADD CONSTRAINT "api_keys_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX api_keys_pkey ON public.api_keys USING btree (id);
CREATE INDEX idx_api_keys_hash ON public.api_keys USING btree (key_hash);
CREATE INDEX idx_api_keys_prefix ON public.api_keys USING btree (key_prefix);
CREATE INDEX idx_api_keys_workspace ON public.api_keys USING btree (workspace_id);

COMMIT;
