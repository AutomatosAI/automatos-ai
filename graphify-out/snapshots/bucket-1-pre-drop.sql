-- Schema-only snapshot
-- Captured at: 2026-04-25T09:47:26.902998+00:00
-- Tables: 11
-- Source: pg_catalog (Railway live)

BEGIN;

-- ============================================
-- Table: b_backup_document_chunks_20251024_20260424
-- ============================================
CREATE TABLE IF NOT EXISTS public."b_backup_document_chunks_20251024_20260424" (
    "id" INTEGER,
    "document_id" INTEGER,
    "chunk_index" INTEGER,
    "content" TEXT,
    "embedding" vector,
    "metadata" JSONB,
    "created_at" TIMESTAMP
);


-- ============================================
-- Table: b_mcp_tools_backup_20260424
-- ============================================
CREATE TABLE IF NOT EXISTS public."b_mcp_tools_backup_20260424" (
    "id" INTEGER,
    "name" VARCHAR(255),
    "description" TEXT,
    "mcp_server_url" VARCHAR(500),
    "capabilities" JSONB,
    "credentials_schema" JSONB,
    "status" VARCHAR(50),
    "provider" VARCHAR(255),
    "version" VARCHAR(50),
    "icon" VARCHAR(100),
    "category" VARCHAR(100),
    "tags" text[],
    "metadata" JSONB,
    "created_by" VARCHAR(255),
    "created_at" TIMESTAMP,
    "updated_at" TIMESTAMP,
    "logo" VARCHAR(255),
    "adapter_tool_id" VARCHAR(255)
);


-- ============================================
-- Table: b_tools_backup_20260424
-- ============================================
CREATE TABLE IF NOT EXISTS public."b_tools_backup_20260424" (
    "id" INTEGER,
    "name" VARCHAR,
    "description" TEXT,
    "category" VARCHAR,
    "provider" VARCHAR,
    "version" VARCHAR,
    "icon" VARCHAR,
    "pricing" VARCHAR,
    "tags" JSON,
    "permissions" JSON,
    "required_credentials" JSON,
    "supported_environments" JSON,
    "mcp_config" JSON,
    "status" VARCHAR,
    "is_installed" BOOLEAN,
    "installation_date" TIMESTAMP,
    "last_used" TIMESTAMP,
    "usage_count" INTEGER,
    "rating" DOUBLE PRECISION,
    "created_at" TIMESTAMP,
    "last_updated" TIMESTAMP,
    "logo" VARCHAR(255)
);


-- ============================================
-- Table: b_agent_messages_20260424
-- ============================================
CREATE TABLE IF NOT EXISTS public."b_agent_messages_20260424" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "from_agent_id" INTEGER,
    "to_agent_id" INTEGER,
    "message_type" VARCHAR(50),
    "content" JSONB,
    "priority" INTEGER,
    "status" VARCHAR(50),
    "created_at" TIMESTAMP DEFAULT now(),
    "delivered_at" TIMESTAMP,
    "read_at" TIMESTAMP
);

ALTER TABLE public."b_agent_messages_20260424" ADD CONSTRAINT "agent_messages_from_agent_id_fkey" FOREIGN KEY (from_agent_id) REFERENCES agents(id);
ALTER TABLE public."b_agent_messages_20260424" ADD CONSTRAINT "agent_messages_to_agent_id_fkey" FOREIGN KEY (to_agent_id) REFERENCES agents(id);
ALTER TABLE public."b_agent_messages_20260424" ADD CONSTRAINT "agent_messages_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX agent_messages_pkey ON public.b_agent_messages_20260424 USING btree (id);
CREATE INDEX idx_agent_messages_from ON public.b_agent_messages_20260424 USING btree (from_agent_id);
CREATE INDEX idx_agent_messages_to ON public.b_agent_messages_20260424 USING btree (to_agent_id);

-- ============================================
-- Table: b_agent_performance_tracking_20260424
-- ============================================
CREATE TABLE IF NOT EXISTS public."b_agent_performance_tracking_20260424" (
    "id" INTEGER NOT NULL DEFAULT nextval('agent_performance_tracking_id_seq'::regclass),
    "agent_id" INTEGER,
    "task_id" INTEGER,
    "execution_time" DOUBLE PRECISION NOT NULL,
    "tokens_used" INTEGER DEFAULT 0,
    "success" BOOLEAN NOT NULL,
    "error_message" TEXT,
    "context_optimization_applied" BOOLEAN DEFAULT false,
    "memory_items_created" INTEGER DEFAULT 0,
    "collaboration_sessions" INTEGER DEFAULT 0,
    "recorded_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."b_agent_performance_tracking_20260424" ADD CONSTRAINT "agent_performance_tracking_agent_id_fkey" FOREIGN KEY (agent_id) REFERENCES agents(id);
ALTER TABLE public."b_agent_performance_tracking_20260424" ADD CONSTRAINT "agent_performance_tracking_task_id_fkey" FOREIGN KEY (task_id) REFERENCES tasks(id);
ALTER TABLE public."b_agent_performance_tracking_20260424" ADD CONSTRAINT "agent_performance_tracking_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX agent_performance_tracking_pkey ON public.b_agent_performance_tracking_20260424 USING btree (id);
CREATE INDEX idx_agent_performance_agent_time ON public.b_agent_performance_tracking_20260424 USING btree (agent_id, recorded_at);
CREATE INDEX idx_agent_performance_success ON public.b_agent_performance_tracking_20260424 USING btree (success, recorded_at);

-- ============================================
-- Table: b_field_states_20260424
-- ============================================
CREATE TABLE IF NOT EXISTS public."b_field_states_20260424" (
    "id" INTEGER NOT NULL DEFAULT nextval('field_states_id_seq'::regclass),
    "session_id" VARCHAR(255) NOT NULL,
    "field_id" VARCHAR(255) NOT NULL,
    "field_type" VARCHAR(100) NOT NULL,
    "field_value" REAL NOT NULL,
    "gradient" float4[] DEFAULT '{}'::real[],
    "context_data" JSON NOT NULL,
    "influence_weights" float4[] DEFAULT '{}'::real[],
    "stability" REAL DEFAULT 0.5,
    "created_at" TIMESTAMP DEFAULT now(),
    "updated_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."b_field_states_20260424" ADD CONSTRAINT "field_states_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX field_states_pkey ON public.b_field_states_20260424 USING btree (id);
CREATE INDEX idx_field_states_field_id ON public.b_field_states_20260424 USING btree (field_id);
CREATE INDEX idx_field_states_session_id ON public.b_field_states_20260424 USING btree (session_id);

-- ============================================
-- Table: b_field_interactions_20260424
-- ============================================
CREATE TABLE IF NOT EXISTS public."b_field_interactions_20260424" (
    "id" INTEGER NOT NULL DEFAULT nextval('field_interactions_id_seq'::regclass),
    "task_id" INTEGER NOT NULL,
    "user_id" INTEGER NOT NULL,
    "session_id_1" VARCHAR(255) NOT NULL,
    "session_id_2" VARCHAR(255) NOT NULL,
    "interaction_type" VARCHAR(100) NOT NULL,
    "similarity_threshold" REAL DEFAULT 0.7,
    "max_interactions" INTEGER DEFAULT 50,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."b_field_interactions_20260424" ADD CONSTRAINT "field_interactions_user_id_fkey" FOREIGN KEY (user_id) REFERENCES users(id);
ALTER TABLE public."b_field_interactions_20260424" ADD CONSTRAINT "field_interactions_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX field_interactions_pkey ON public.b_field_interactions_20260424 USING btree (id);
CREATE INDEX idx_field_interactions_task_id ON public.b_field_interactions_20260424 USING btree (task_id);

-- ============================================
-- Table: b_historical_tasks_20260424
-- ============================================
CREATE TABLE IF NOT EXISTS public."b_historical_tasks_20260424" (
    "id" INTEGER NOT NULL DEFAULT nextval('historical_tasks_id_seq'::regclass),
    "task_description" TEXT NOT NULL,
    "task_type" TEXT,
    "context_used" JSONB DEFAULT '{}'::jsonb,
    "outcome" TEXT,
    "success" BOOLEAN DEFAULT false,
    "execution_time" DOUBLE PRECISION,
    "agent_used" TEXT,
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "metadata" JSONB DEFAULT '{}'::jsonb
);

ALTER TABLE public."b_historical_tasks_20260424" ADD CONSTRAINT "historical_tasks_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX historical_tasks_pkey ON public.b_historical_tasks_20260424 USING btree (id);

-- ============================================
-- Table: b_task_assignments_20260424
-- ============================================
CREATE TABLE IF NOT EXISTS public."b_task_assignments_20260424" (
    "id" INTEGER NOT NULL DEFAULT nextval('task_assignments_id_seq'::regclass),
    "task_id" INTEGER,
    "agent_id" INTEGER,
    "assignment_score" DOUBLE PRECISION,
    "assignment_reason" TEXT,
    "status" VARCHAR(50),
    "assigned_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."b_task_assignments_20260424" ADD CONSTRAINT "task_assignments_agent_id_fkey" FOREIGN KEY (agent_id) REFERENCES agents(id);
ALTER TABLE public."b_task_assignments_20260424" ADD CONSTRAINT "task_assignments_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX task_assignments_pkey ON public.b_task_assignments_20260424 USING btree (id);

-- ============================================
-- Table: b_agent_runtimes_20260424
-- ============================================
CREATE TABLE IF NOT EXISTS public."b_agent_runtimes_20260424" (
    "id" INTEGER NOT NULL DEFAULT nextval('agent_runtimes_id_seq'::regclass),
    "agent_id" INTEGER,
    "llm_provider" VARCHAR(50),
    "model_name" VARCHAR(100),
    "temperature" DOUBLE PRECISION,
    "max_tokens" INTEGER,
    "context_window" INTEGER,
    "api_key_ref" VARCHAR(255),
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."b_agent_runtimes_20260424" ADD CONSTRAINT "agent_runtimes_agent_id_fkey" FOREIGN KEY (agent_id) REFERENCES agents(id);
ALTER TABLE public."b_agent_runtimes_20260424" ADD CONSTRAINT "agent_runtimes_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX agent_runtimes_pkey ON public.b_agent_runtimes_20260424 USING btree (id);

-- ============================================
-- Table: b_task_decompositions_20260424
-- ============================================
CREATE TABLE IF NOT EXISTS public."b_task_decompositions_20260424" (
    "id" INTEGER NOT NULL DEFAULT nextval('task_decompositions_id_seq'::regclass),
    "parent_task_id" INTEGER,
    "subtask_id" INTEGER,
    "dependency_type" VARCHAR(50),
    "execution_order" INTEGER,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."b_task_decompositions_20260424" ADD CONSTRAINT "task_decompositions_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX task_decompositions_pkey ON public.b_task_decompositions_20260424 USING btree (id);

COMMIT;
