"""PRD-79: L2 Short-Term Memory Table

Create memory_short_term table for the 5-layer memory stack.
Stores raw exchanges, recipe summaries, heartbeat logs, and tool results
with Ebbinghaus decay scoring for graduated importance.

Standalone migration (down_revision = None).
"""

from alembic import op
import sqlalchemy as sa

revision = "prd79_memory_short_term"
down_revision = None  # Standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS memory_short_term (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id    UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            agent_id        INTEGER REFERENCES agents(id) ON DELETE SET NULL,

            -- Content
            content         TEXT NOT NULL,
            content_type    VARCHAR(30) NOT NULL DEFAULT 'exchange',
            -- content_type: 'exchange', 'recipe_summary', 'heartbeat_log',
            --               'tool_result', 'session_decision'

            -- Scoring
            importance      FLOAT NOT NULL DEFAULT 0.5,
            decay_score     FLOAT NOT NULL DEFAULT 1.0,
            access_count    INTEGER NOT NULL DEFAULT 0,

            -- Metadata
            metadata        JSONB NOT NULL DEFAULT '{}',

            -- Promotion tracking
            promoted_to_l3  BOOLEAN NOT NULL DEFAULT false,
            promoted_at     TIMESTAMPTZ,
            archived_at     TIMESTAMPTZ,

            -- Timestamps
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        -- Time-range queries (Context Router temporal fetch)
        CREATE INDEX IF NOT EXISTS ix_mem_st_ws_created
            ON memory_short_term(workspace_id, created_at DESC);

        -- Decay job: find items needing score update
        CREATE INDEX IF NOT EXISTS ix_mem_st_ws_decay
            ON memory_short_term(workspace_id, decay_score)
            WHERE archived_at IS NULL;

        -- Promotion job: find candidates
        CREATE INDEX IF NOT EXISTS ix_mem_st_ws_promote
            ON memory_short_term(workspace_id, promoted_to_l3)
            WHERE promoted_to_l3 = false AND archived_at IS NULL;

        -- Per-agent filtering
        CREATE INDEX IF NOT EXISTS ix_mem_st_ws_agent
            ON memory_short_term(workspace_id, agent_id, created_at DESC);

        -- Content type filtering
        CREATE INDEX IF NOT EXISTS ix_mem_st_ws_type
            ON memory_short_term(workspace_id, content_type, created_at DESC);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS memory_short_term;")
