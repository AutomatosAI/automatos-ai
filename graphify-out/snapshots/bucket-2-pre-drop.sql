-- Schema-only snapshot
-- Captured at: 2026-04-25T11:15:32.900035+00:00
-- Tables: 10
-- Source: pg_catalog (Railway live)

BEGIN;

-- ============================================
-- Table: context_examples
-- ============================================
CREATE TABLE IF NOT EXISTS public."context_examples" (
    "id" INTEGER NOT NULL DEFAULT nextval('context_examples_id_seq'::regclass),
    "category" VARCHAR(100),
    "input_text" TEXT,
    "output_text" TEXT,
    "embedding" vector,
    "metadata" JSONB,
    "quality_score" DOUBLE PRECISION,
    "usage_count" INTEGER DEFAULT 0,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."context_examples" ADD CONSTRAINT "context_examples_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX context_examples_pkey ON public.context_examples USING btree (id);
CREATE INDEX idx_context_examples_embedding ON public.context_examples USING ivfflat (embedding vector_cosine_ops);

-- ============================================
-- Table: context_optimizations
-- ============================================
CREATE TABLE IF NOT EXISTS public."context_optimizations" (
    "id" INTEGER NOT NULL DEFAULT nextval('context_optimizations_id_seq'::regclass),
    "task_id" INTEGER,
    "original_tokens" INTEGER,
    "optimized_tokens" INTEGER,
    "information_gain" DOUBLE PRECISION,
    "optimization_strategy" VARCHAR(100),
    "metrics" JSONB,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."context_optimizations" ADD CONSTRAINT "context_optimizations_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX context_optimizations_pkey ON public.context_optimizations USING btree (id);

-- ============================================
-- Table: context_patterns
-- ============================================
CREATE TABLE IF NOT EXISTS public."context_patterns" (
    "id" INTEGER NOT NULL DEFAULT nextval('context_patterns_id_seq'::regclass),
    "pattern_name" VARCHAR(255),
    "pattern_type" VARCHAR(50),
    "pattern_structure" JSONB,
    "applicable_tasks" JSONB,
    "effectiveness_score" DOUBLE PRECISION,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."context_patterns" ADD CONSTRAINT "context_patterns_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX context_patterns_pkey ON public.context_patterns USING btree (id);

-- ============================================
-- Table: context_queries
-- ============================================
CREATE TABLE IF NOT EXISTS public."context_queries" (
    "id" INTEGER NOT NULL DEFAULT nextval('context_queries_id_seq'::regclass),
    "query_text" TEXT NOT NULL,
    "query_type" VARCHAR(100) NOT NULL,
    "results" JSON DEFAULT '{}'::json,
    "response_time_ms" INTEGER,
    "success" BOOLEAN DEFAULT true,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."context_queries" ADD CONSTRAINT "context_queries_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX context_queries_pkey ON public.context_queries USING btree (id);

-- ============================================
-- Table: context_sources
-- ============================================
CREATE TABLE IF NOT EXISTS public."context_sources" (
    "id" INTEGER NOT NULL DEFAULT nextval('context_sources_id_seq'::regclass),
    "source_name" VARCHAR(255) NOT NULL,
    "source_type" VARCHAR(100) NOT NULL,
    "source_data" JSON DEFAULT '{}'::json,
    "is_active" BOOLEAN DEFAULT true,
    "created_at" TIMESTAMP DEFAULT now(),
    "updated_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."context_sources" ADD CONSTRAINT "context_sources_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX context_sources_pkey ON public.context_sources USING btree (id);

-- ============================================
-- Table: context_templates
-- ============================================
CREATE TABLE IF NOT EXISTS public."context_templates" (
    "id" INTEGER NOT NULL DEFAULT nextval('context_templates_id_seq'::regclass),
    "name" VARCHAR(255),
    "template_type" VARCHAR(50),
    "template_text" TEXT,
    "parameters" JSONB,
    "usage_count" INTEGER DEFAULT 0,
    "success_rate" DOUBLE PRECISION,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."context_templates" ADD CONSTRAINT "context_templates_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX context_templates_pkey ON public.context_templates USING btree (id);

-- ============================================
-- Table: context_usage
-- ============================================
CREATE TABLE IF NOT EXISTS public."context_usage" (
    "id" INTEGER NOT NULL DEFAULT nextval('context_usage_id_seq'::regclass),
    "document_id" INTEGER,
    "chunk_id" INTEGER,
    "query_text" TEXT,
    "relevance_score" DOUBLE PRECISION,
    "used_in_response" BOOLEAN DEFAULT false,
    "timestamp" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE public."context_usage" ADD CONSTRAINT "context_usage_chunk_id_fkey" FOREIGN KEY (chunk_id) REFERENCES document_chunks(id);
ALTER TABLE public."context_usage" ADD CONSTRAINT "context_usage_document_id_fkey" FOREIGN KEY (document_id) REFERENCES documents(id);
ALTER TABLE public."context_usage" ADD CONSTRAINT "context_usage_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX context_usage_pkey ON public.context_usage USING btree (id);

-- ============================================
-- Table: context_permissions
-- ============================================
CREATE TABLE IF NOT EXISTS public."context_permissions" (
    "id" INTEGER NOT NULL DEFAULT nextval('context_permissions_id_seq'::regclass),
    "context_id" UUID,
    "agent_id" INTEGER,
    "permission_level" VARCHAR(50),
    "granted_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."context_permissions" ADD CONSTRAINT "context_permissions_agent_id_fkey" FOREIGN KEY (agent_id) REFERENCES agents(id);
ALTER TABLE public."context_permissions" ADD CONSTRAINT "context_permissions_context_id_fkey" FOREIGN KEY (context_id) REFERENCES shared_contexts(id);
ALTER TABLE public."context_permissions" ADD CONSTRAINT "context_permissions_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX context_permissions_pkey ON public.context_permissions USING btree (id);

-- ============================================
-- Table: entity_clusters
-- ============================================
CREATE TABLE IF NOT EXISTS public."entity_clusters" (
    "id" INTEGER NOT NULL DEFAULT nextval('entity_clusters_id_seq'::regclass),
    "cluster_name" VARCHAR(255),
    "cluster_topic" VARCHAR(500),
    "entity_ids" int4[],
    "size" INTEGER DEFAULT 0,
    "coherence_score" DOUBLE PRECISION DEFAULT 0.0,
    "keywords" text[],
    "created_at" TIMESTAMP DEFAULT now(),
    "updated_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."entity_clusters" ADD CONSTRAINT "entity_clusters_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX entity_clusters_pkey ON public.entity_clusters USING btree (id);
CREATE INDEX idx_entity_clusters_size ON public.entity_clusters USING btree (size DESC);

-- ============================================
-- Table: shared_contexts
-- ============================================
CREATE TABLE IF NOT EXISTS public."shared_contexts" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "name" VARCHAR(255),
    "team_id" UUID,
    "context_data" JSONB,
    "version" INTEGER DEFAULT 1,
    "created_at" TIMESTAMP DEFAULT now(),
    "updated_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."shared_contexts" ADD CONSTRAINT "shared_contexts_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX shared_contexts_pkey ON public.shared_contexts USING btree (id);

COMMIT;
