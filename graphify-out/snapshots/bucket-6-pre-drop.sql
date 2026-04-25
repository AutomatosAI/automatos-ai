-- Schema-only snapshot
-- Captured at: 2026-04-25T12:02:11.148473+00:00
-- Tables: 4
-- Source: pg_catalog (Railway live)

BEGIN;

-- ============================================
-- Table: playbooks
-- ============================================
CREATE TABLE IF NOT EXISTS public."playbooks" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "name" VARCHAR(128) NOT NULL,
    "tenant_id" VARCHAR(128),
    "pattern" TEXT NOT NULL,
    "support" INTEGER NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public."playbooks" ADD CONSTRAINT "playbooks_pkey" PRIMARY KEY (id);
CREATE INDEX ix_playbooks_name ON public.playbooks USING btree (name);
CREATE INDEX ix_playbooks_tenant_id ON public.playbooks USING btree (tenant_id);
CREATE UNIQUE INDEX playbooks_pkey ON public.playbooks USING btree (id);

-- ============================================
-- Table: schema_versions
-- ============================================
CREATE TABLE IF NOT EXISTS public."schema_versions" (
    "version" VARCHAR(20) NOT NULL,
    "applied_at" TIMESTAMP DEFAULT now(),
    "description" TEXT
);

ALTER TABLE public."schema_versions" ADD CONSTRAINT "schema_versions_pkey" PRIMARY KEY (version);
CREATE UNIQUE INDEX schema_versions_pkey ON public.schema_versions USING btree (version);

-- ============================================
-- Table: code_symbols
-- ============================================
CREATE TABLE IF NOT EXISTS public."code_symbols" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "project" VARCHAR(128) NOT NULL,
    "file_path" TEXT NOT NULL,
    "symbol_name" VARCHAR(256) NOT NULL,
    "symbol_type" VARCHAR(32) NOT NULL,
    "signature" TEXT,
    "docstring" TEXT,
    "start_line" INTEGER,
    "end_line" INTEGER,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public."code_symbols" ADD CONSTRAINT "code_symbols_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX code_symbols_pkey ON public.code_symbols USING btree (id);
CREATE INDEX ix_code_symbols_project ON public.code_symbols USING btree (project);
CREATE INDEX ix_code_symbols_project_name ON public.code_symbols USING btree (project, symbol_name);
CREATE INDEX ix_code_symbols_symbol_name ON public.code_symbols USING btree (symbol_name);

-- ============================================
-- Table: code_edges
-- ============================================
CREATE TABLE IF NOT EXISTS public."code_edges" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "project" VARCHAR(128) NOT NULL,
    "src_symbol_id" UUID NOT NULL,
    "dst_symbol_id" UUID NOT NULL,
    "edge_type" VARCHAR(32) NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public."code_edges" ADD CONSTRAINT "code_edges_dst_symbol_id_fkey" FOREIGN KEY (dst_symbol_id) REFERENCES code_symbols(id) ON DELETE CASCADE;
ALTER TABLE public."code_edges" ADD CONSTRAINT "code_edges_src_symbol_id_fkey" FOREIGN KEY (src_symbol_id) REFERENCES code_symbols(id) ON DELETE CASCADE;
ALTER TABLE public."code_edges" ADD CONSTRAINT "code_edges_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX code_edges_pkey ON public.code_edges USING btree (id);
CREATE INDEX ix_code_edges_dst_symbol_id ON public.code_edges USING btree (dst_symbol_id);
CREATE INDEX ix_code_edges_project ON public.code_edges USING btree (project);
CREATE INDEX ix_code_edges_project_type ON public.code_edges USING btree (project, edge_type);
CREATE INDEX ix_code_edges_src_symbol_id ON public.code_edges USING btree (src_symbol_id);

COMMIT;
