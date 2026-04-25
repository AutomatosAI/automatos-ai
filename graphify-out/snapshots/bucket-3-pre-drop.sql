-- Schema-only snapshot
-- Captured at: 2026-04-25T11:55:09.236876+00:00
-- Tables: 7
-- Source: pg_catalog (Railway live)

BEGIN;

-- ============================================
-- Table: agent_coordination
-- ============================================
CREATE TABLE IF NOT EXISTS public."agent_coordination" (
    "id" INTEGER NOT NULL DEFAULT nextval('agent_coordination_id_seq'::regclass),
    "task_id" INTEGER NOT NULL,
    "user_id" INTEGER NOT NULL,
    "agents" text[] NOT NULL,
    "strategy" VARCHAR(100) NOT NULL,
    "load_balance" BOOLEAN DEFAULT false,
    "context" JSON DEFAULT '{}'::json,
    "status" VARCHAR(50) DEFAULT 'pending'::character varying,
    "created_at" TIMESTAMP DEFAULT now(),
    "updated_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."agent_coordination" ADD CONSTRAINT "agent_coordination_user_id_fkey" FOREIGN KEY (user_id) REFERENCES users(id);
ALTER TABLE public."agent_coordination" ADD CONSTRAINT "agent_coordination_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX agent_coordination_pkey ON public.agent_coordination USING btree (id);

-- ============================================
-- Table: multi_agent_reasoning
-- ============================================
CREATE TABLE IF NOT EXISTS public."multi_agent_reasoning" (
    "id" INTEGER NOT NULL DEFAULT nextval('multi_agent_reasoning_id_seq'::regclass),
    "task_id" INTEGER NOT NULL,
    "user_id" INTEGER NOT NULL,
    "agents" text[] NOT NULL,
    "task" JSON NOT NULL,
    "strategy" VARCHAR(100) NOT NULL,
    "timeout_seconds" INTEGER DEFAULT 300,
    "status" VARCHAR(50) DEFAULT 'pending'::character varying,
    "result" JSON DEFAULT '{}'::json,
    "created_at" TIMESTAMP DEFAULT now(),
    "updated_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."multi_agent_reasoning" ADD CONSTRAINT "multi_agent_reasoning_user_id_fkey" FOREIGN KEY (user_id) REFERENCES users(id);
ALTER TABLE public."multi_agent_reasoning" ADD CONSTRAINT "multi_agent_reasoning_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX multi_agent_reasoning_pkey ON public.multi_agent_reasoning USING btree (id);

-- ============================================
-- Table: agent_behavior_monitoring
-- ============================================
CREATE TABLE IF NOT EXISTS public."agent_behavior_monitoring" (
    "id" INTEGER NOT NULL DEFAULT nextval('agent_behavior_monitoring_id_seq'::regclass),
    "session_id" VARCHAR(255) NOT NULL,
    "agents" text[] NOT NULL,
    "interactions" JSON DEFAULT '[]'::json,
    "task_data" JSON DEFAULT '{}'::json,
    "monitoring_start" TIMESTAMP DEFAULT now(),
    "monitoring_end" TIMESTAMP,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."agent_behavior_monitoring" ADD CONSTRAINT "agent_behavior_monitoring_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX agent_behavior_monitoring_pkey ON public.agent_behavior_monitoring USING btree (id);

-- ============================================
-- Table: agent_performance
-- ============================================
CREATE TABLE IF NOT EXISTS public."agent_performance" (
    "id" INTEGER NOT NULL DEFAULT nextval('agent_performance_id_seq'::regclass),
    "agent_id" INTEGER,
    "task_id" INTEGER,
    "execution_time" DOUBLE PRECISION,
    "token_usage" JSONB,
    "quality_score" DOUBLE PRECISION,
    "error_count" INTEGER,
    "success" BOOLEAN,
    "recorded_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."agent_performance" ADD CONSTRAINT "agent_performance_agent_id_fkey" FOREIGN KEY (agent_id) REFERENCES agents(id);
ALTER TABLE public."agent_performance" ADD CONSTRAINT "agent_performance_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX agent_performance_pkey ON public.agent_performance USING btree (id);

-- ============================================
-- Table: collaboration_proposals
-- ============================================
CREATE TABLE IF NOT EXISTS public."collaboration_proposals" (
    "id" INTEGER NOT NULL DEFAULT nextval('collaboration_proposals_id_seq'::regclass),
    "session_id" UUID,
    "agent_id" INTEGER,
    "proposal_type" VARCHAR(50),
    "proposal_content" JSON NOT NULL,
    "confidence" DOUBLE PRECISION,
    "tokens_used" INTEGER,
    "execution_time" DOUBLE PRECISION,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."collaboration_proposals" ADD CONSTRAINT "collaboration_proposals_agent_id_fkey" FOREIGN KEY (agent_id) REFERENCES agents(id);
ALTER TABLE public."collaboration_proposals" ADD CONSTRAINT "collaboration_proposals_session_id_fkey" FOREIGN KEY (session_id) REFERENCES collaboration_sessions(id) ON DELETE CASCADE;
ALTER TABLE public."collaboration_proposals" ADD CONSTRAINT "collaboration_proposals_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX collaboration_proposals_pkey ON public.collaboration_proposals USING btree (id);

-- ============================================
-- Table: consensus_votes
-- ============================================
CREATE TABLE IF NOT EXISTS public."consensus_votes" (
    "id" INTEGER NOT NULL DEFAULT nextval('consensus_votes_id_seq'::regclass),
    "session_id" UUID,
    "proposal_id" INTEGER,
    "agent_id" INTEGER,
    "vote_weight" DOUBLE PRECISION,
    "vote_value" VARCHAR(50),
    "reasoning" TEXT,
    "created_at" TIMESTAMP DEFAULT now()
);

ALTER TABLE public."consensus_votes" ADD CONSTRAINT "consensus_votes_agent_id_fkey" FOREIGN KEY (agent_id) REFERENCES agents(id);
ALTER TABLE public."consensus_votes" ADD CONSTRAINT "consensus_votes_proposal_id_fkey" FOREIGN KEY (proposal_id) REFERENCES collaboration_proposals(id) ON DELETE CASCADE;
ALTER TABLE public."consensus_votes" ADD CONSTRAINT "consensus_votes_session_id_fkey" FOREIGN KEY (session_id) REFERENCES collaboration_sessions(id) ON DELETE CASCADE;
ALTER TABLE public."consensus_votes" ADD CONSTRAINT "consensus_votes_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX consensus_votes_pkey ON public.consensus_votes USING btree (id);

-- ============================================
-- Table: message_broadcasts
-- ============================================
CREATE TABLE IF NOT EXISTS public."message_broadcasts" (
    "id" VARCHAR(255) NOT NULL,
    "from_agent_id" INTEGER,
    "team_agents" JSON,
    "message_type" VARCHAR(50),
    "content" JSON,
    "priority" INTEGER,
    "delivered_to" JSON,
    "failed_deliveries" JSON,
    "created_at" TIMESTAMP
);

ALTER TABLE public."message_broadcasts" ADD CONSTRAINT "message_broadcasts_from_agent_id_fkey" FOREIGN KEY (from_agent_id) REFERENCES agents(id);
ALTER TABLE public."message_broadcasts" ADD CONSTRAINT "message_broadcasts_pkey" PRIMARY KEY (id);
CREATE UNIQUE INDEX message_broadcasts_pkey ON public.message_broadcasts USING btree (id);

COMMIT;
