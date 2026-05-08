"""PRD-140 — permission_bypass_log

Queryable audit trail for every time the hierarchy permission system grants
a bypass (e.g. Auto, HARNESS service actor, platform admin). Stored in its
own table — not buried in app logs — so the workspace owner can answer
"did anyone bypass anything in the last 7 days?" with one SQL query.

Idempotent.
"""

from alembic import op


revision = "prd140_permission_bypass_log"
down_revision = "wave3_escalation_level"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS permission_bypass_log (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id    UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            actor_agent_id  INTEGER REFERENCES agents(id) ON DELETE SET NULL,
            actor_name      VARCHAR(255) NOT NULL,
            actor_kind      VARCHAR(40)  NOT NULL,
            target_type     VARCHAR(40)  NOT NULL,
            target_id       VARCHAR(255),
            change_type     VARCHAR(60)  NOT NULL,
            reason          VARCHAR(80)  NOT NULL,
            source          VARCHAR(120),
            metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_perm_bypass_workspace_created "
        "ON permission_bypass_log (workspace_id, created_at DESC);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_perm_bypass_actor "
        "ON permission_bypass_log (workspace_id, actor_agent_id, created_at DESC) "
        "WHERE actor_agent_id IS NOT NULL;"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_perm_bypass_target "
        "ON permission_bypass_log (workspace_id, target_type, target_id);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_perm_bypass_target;")
    op.execute("DROP INDEX IF EXISTS ix_perm_bypass_actor;")
    op.execute("DROP INDEX IF EXISTS ix_perm_bypass_workspace_created;")
    op.execute("DROP TABLE IF EXISTS permission_bypass_log;")
