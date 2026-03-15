"""PRD-82A: Sequential Mission Coordinator — Orchestration Tables

Create 4 new tables for the orchestration subsystem:
  - orchestration_runs: top-level mission execution
  - orchestration_tasks: individual tasks within a mission
  - orchestration_task_dependencies: DAG edges between tasks
  - orchestration_events: append-only audit log

Alter 2 existing tables:
  - board_tasks: add orchestration_run_id and orchestration_task_id FKs
  - agent_reports: add orchestration_task_id FK

Source: PRD-82A Sections 4-9, PRD-101 Section 12-13

Revision ID: prd82a_orchestration_tables
Revises: None (standalone — safe to run anytime)
Create Date: 2026-03-15
"""
from alembic import op
import sqlalchemy as sa

revision = "prd82a_orchestration_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -----------------------------------------------------------------------
    # 1. orchestration_runs — top-level mission execution
    # -----------------------------------------------------------------------
    op.execute("""
        CREATE TABLE IF NOT EXISTS orchestration_runs (
            id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id            UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            goal                    TEXT NOT NULL,
            plan                    JSONB,
            config                  JSONB DEFAULT '{}',
            state                   VARCHAR(30) NOT NULL DEFAULT 'pending'
                CHECK (state IN (
                    'pending', 'planning', 'awaiting_approval', 'running',
                    'paused', 'verifying', 'awaiting_human',
                    'completed', 'failed', 'cancelled'
                )),
            state_type              VARCHAR(10) NOT NULL DEFAULT 'initial',
            created_by              VARCHAR(255) NOT NULL,
            assigned_coordinator_id INTEGER REFERENCES agents(id) ON DELETE SET NULL,
            output_summary          JSONB,
            token_budget_estimate   INTEGER,
            tokens_used             INTEGER NOT NULL DEFAULT 0,
            max_retries             INTEGER NOT NULL DEFAULT 3,
            max_concurrent          INTEGER NOT NULL DEFAULT 1,
            started_at              TIMESTAMPTZ,
            completed_at            TIMESTAMPTZ,
            created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            version_id              INTEGER NOT NULL DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS ix_orchestration_runs_workspace_id
            ON orchestration_runs(workspace_id);
    """)

    # -----------------------------------------------------------------------
    # 2. orchestration_tasks — individual tasks within a mission
    # -----------------------------------------------------------------------
    op.execute("""
        CREATE TABLE IF NOT EXISTS orchestration_tasks (
            id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id                  UUID NOT NULL REFERENCES orchestration_runs(id) ON DELETE CASCADE,
            title                   VARCHAR(500) NOT NULL,
            description             TEXT,
            task_type               VARCHAR(30) NOT NULL DEFAULT 'llm_generation',
            sequence_number         INTEGER NOT NULL,
            agent_role              VARCHAR(100),
            state                   VARCHAR(30) NOT NULL DEFAULT 'pending'
                CHECK (state IN (
                    'pending', 'queued', 'assigned', 'running',
                    'completed', 'verifying', 'verified',
                    'failed', 'skipped', 'stalled', 'retrying'
                )),
            state_type              VARCHAR(10) NOT NULL DEFAULT 'initial',
            assigned_agent_id       INTEGER REFERENCES agents(id) ON DELETE SET NULL,
            verification_criteria   JSONB,
            input_context           JSONB,
            output                  TEXT,
            output_metadata         JSONB,
            failure_reason_code     VARCHAR(50),
            failure_detail          TEXT,
            attempt_number          INTEGER NOT NULL DEFAULT 0,
            max_retries             INTEGER NOT NULL DEFAULT 3,
            tokens_used             INTEGER NOT NULL DEFAULT 0,
            started_at              TIMESTAMPTZ,
            completed_at            TIMESTAMPTZ,
            created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            version_id              INTEGER NOT NULL DEFAULT 1
        );

        -- FK index
        CREATE INDEX IF NOT EXISTS ix_orchestration_tasks_run_id
            ON orchestration_tasks(run_id);

        -- Composite index for ordering tasks within a run
        CREATE INDEX IF NOT EXISTS ix_orchestration_tasks_run_sequence
            ON orchestration_tasks(run_id, sequence_number);

        -- Partial index on active (non-terminal) states for coordinator tick queries
        CREATE INDEX IF NOT EXISTS ix_orchestration_tasks_active
            ON orchestration_tasks(run_id, state)
            WHERE state NOT IN ('verified', 'failed', 'skipped');
    """)

    # -----------------------------------------------------------------------
    # 3. orchestration_task_dependencies — DAG edges
    # -----------------------------------------------------------------------
    op.execute("""
        CREATE TABLE IF NOT EXISTS orchestration_task_dependencies (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            task_id             UUID NOT NULL REFERENCES orchestration_tasks(id) ON DELETE CASCADE,
            depends_on_task_id  UUID NOT NULL REFERENCES orchestration_tasks(id) ON DELETE CASCADE,
            trigger_rule        VARCHAR(30) NOT NULL DEFAULT 'all_success',
            CONSTRAINT uq_orchestration_task_dep_pair UNIQUE (task_id, depends_on_task_id)
        );

        CREATE INDEX IF NOT EXISTS ix_orchestration_task_dependencies_task_id
            ON orchestration_task_dependencies(task_id);

        CREATE INDEX IF NOT EXISTS ix_orchestration_task_dependencies_depends_on
            ON orchestration_task_dependencies(depends_on_task_id);
    """)

    # -----------------------------------------------------------------------
    # 4. orchestration_events — append-only audit log
    # -----------------------------------------------------------------------
    op.execute("""
        CREATE TABLE IF NOT EXISTS orchestration_events (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id          UUID NOT NULL REFERENCES orchestration_runs(id) ON DELETE CASCADE,
            task_id         UUID REFERENCES orchestration_tasks(id) ON DELETE CASCADE,
            event_type      VARCHAR(50) NOT NULL,
            actor_type      VARCHAR(20) NOT NULL,
            actor_id        VARCHAR(255),
            old_state       VARCHAR(30),
            new_state       VARCHAR(30),
            payload         JSONB,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS ix_orchestration_events_run_id
            ON orchestration_events(run_id);

        CREATE INDEX IF NOT EXISTS ix_orchestration_events_task_id
            ON orchestration_events(task_id);

        -- Composite index for event timeline queries per run
        CREATE INDEX IF NOT EXISTS ix_orchestration_events_run_created
            ON orchestration_events(run_id, created_at);
    """)

    # -----------------------------------------------------------------------
    # 5. ALTER existing tables — add nullable FKs
    # Use CONCURRENTLY for indexes on existing tables (requires separate statements)
    # -----------------------------------------------------------------------

    # board_tasks: add orchestration_run_id and orchestration_task_id
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'board_tasks' AND column_name = 'orchestration_run_id'
            ) THEN
                ALTER TABLE board_tasks
                    ADD COLUMN orchestration_run_id UUID REFERENCES orchestration_runs(id) ON DELETE SET NULL;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'board_tasks' AND column_name = 'orchestration_task_id'
            ) THEN
                ALTER TABLE board_tasks
                    ADD COLUMN orchestration_task_id UUID REFERENCES orchestration_tasks(id) ON DELETE SET NULL;
            END IF;
        END $$;
    """)

    # agent_reports: add orchestration_task_id
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'agent_reports' AND column_name = 'orchestration_task_id'
            ) THEN
                ALTER TABLE agent_reports
                    ADD COLUMN orchestration_task_id UUID REFERENCES orchestration_tasks(id) ON DELETE SET NULL;
            END IF;
        END $$;
    """)

    # Indexes on existing tables (not CONCURRENTLY since we're inside a transaction)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_board_tasks_orchestration_run_id
            ON board_tasks(orchestration_run_id)
            WHERE orchestration_run_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS ix_board_tasks_orchestration_task_id
            ON board_tasks(orchestration_task_id)
            WHERE orchestration_task_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS ix_agent_reports_orchestration_task_id
            ON agent_reports(orchestration_task_id)
            WHERE orchestration_task_id IS NOT NULL;
    """)


def downgrade() -> None:
    # Remove FK columns from existing tables
    op.execute("""
        ALTER TABLE agent_reports DROP COLUMN IF EXISTS orchestration_task_id;
        ALTER TABLE board_tasks DROP COLUMN IF EXISTS orchestration_task_id;
        ALTER TABLE board_tasks DROP COLUMN IF EXISTS orchestration_run_id;
    """)

    # Drop tables in reverse dependency order
    op.execute("""
        DROP TABLE IF EXISTS orchestration_events;
        DROP TABLE IF EXISTS orchestration_task_dependencies;
        DROP TABLE IF EXISTS orchestration_tasks;
        DROP TABLE IF EXISTS orchestration_runs;
    """)
