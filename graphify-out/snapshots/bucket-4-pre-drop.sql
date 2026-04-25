-- Schema-only snapshot
-- Captured at: 2026-04-25T11:56:42.342621+00:00
-- Tables: 4
-- Source: pg_catalog (Railway live)

BEGIN;

-- ============================================
-- Table: vector_documents
-- ============================================
CREATE TABLE IF NOT EXISTS public."vector_documents" (
    "id" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "embedding" vector NOT NULL,
    "metadata" JSONB DEFAULT '{}'::jsonb,
    "timestamp" TIMESTAMP WITH TIME ZONE DEFAULT now(),
    "source" TEXT,
    "document_type" TEXT,
    "importance_score" DOUBLE PRECISION DEFAULT 0.0,
    "content_hash" TEXT,
    "created_at" TIMESTAMP WITH TIME ZONE DEFAULT now(),
    "updated_at" TIMESTAMP WITH TIME ZONE DEFAULT now()
);

ALTER TABLE public."vector_documents" ADD CONSTRAINT "vector_documents_pkey" PRIMARY KEY (id);
CREATE INDEX vector_documents_document_type_idx ON public.vector_documents USING btree (document_type);
CREATE INDEX vector_documents_embedding_idx ON public.vector_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists='100');
CREATE INDEX vector_documents_importance_idx ON public.vector_documents USING btree (importance_score DESC);
CREATE INDEX vector_documents_metadata_idx ON public.vector_documents USING gin (metadata);
CREATE UNIQUE INDEX vector_documents_pkey ON public.vector_documents USING btree (id);
CREATE INDEX vector_documents_source_idx ON public.vector_documents USING btree (source);
CREATE INDEX vector_documents_timestamp_idx ON public.vector_documents USING btree ("timestamp" DESC);

-- ============================================
-- Table: document_embeddings
-- ============================================
CREATE TABLE IF NOT EXISTS public."document_embeddings" (
    "id" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "embedding" vector NOT NULL,
    "metadata" JSONB DEFAULT '{}'::jsonb,
    "source_file" TEXT NOT NULL,
    "chunk_index" INTEGER NOT NULL,
    "content_type" TEXT DEFAULT 'text'::text,
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE public."document_embeddings" ADD CONSTRAINT "document_embeddings_pkey" PRIMARY KEY (id);
CREATE INDEX document_embeddings_content_type_idx ON public.document_embeddings USING btree (content_type);
CREATE INDEX document_embeddings_embedding_idx ON public.document_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='200');
CREATE INDEX document_embeddings_metadata_idx ON public.document_embeddings USING gin (metadata);
CREATE UNIQUE INDEX document_embeddings_pkey ON public.document_embeddings USING btree (id);
CREATE INDEX document_embeddings_source_file_idx ON public.document_embeddings USING btree (source_file);

-- ============================================
-- Table: agent_memories
-- ============================================
CREATE TABLE IF NOT EXISTS public."agent_memories" (
    "id" INTEGER NOT NULL DEFAULT nextval('agent_memories_id_seq'::regclass),
    "agent_id" INTEGER,
    "memory_type" VARCHAR(50) DEFAULT 'experience'::character varying,
    "content" TEXT NOT NULL,
    "metadata" JSONB DEFAULT '{}'::jsonb,
    "embedding" vector,
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE public."agent_memories" ADD CONSTRAINT "agent_memories_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX agent_memories_pkey ON public.agent_memories USING btree (id);

-- ============================================
-- Table: analytics_snapshots
-- ============================================
CREATE TABLE IF NOT EXISTS public."analytics_snapshots" (
    "id" INTEGER NOT NULL DEFAULT nextval('analytics_snapshots_id_seq'::regclass),
    "snapshot_type" VARCHAR(50),
    "metrics" JSONB,
    "timestamp" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."analytics_snapshots" ADD CONSTRAINT "analytics_snapshots_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX analytics_snapshots_pkey ON public.analytics_snapshots USING btree (id);
CREATE INDEX idx_analytics_snapshots_type_time ON public.analytics_snapshots USING btree (snapshot_type, "timestamp");

COMMIT;
