"""PRD-181 S2 (F060) — approval_grants table (durable, scoped, expiring, revocable)

The tool-agnostic approval-grant record that gates board tasks, playbook runs,
and future scheduled/webhook agents hitting an ``ask`` tier. Justified new table:
no existing table records "a human granted permission for action Y on subject Z
until time T, revocable" — approval state today is transient run/task state.

Idempotent. Chained onto the S1 audit migration.
"""

from alembic import op


revision = "prd181_s2_approval_grants"
down_revision = "prd181_s1_audit_actor"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS approval_grants (
            id                  SERIAL PRIMARY KEY,
            workspace_id        UUID NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
            subject_type        VARCHAR(30)  NOT NULL,
            subject_id          VARCHAR(255) NOT NULL,
            tool_name           VARCHAR(255),
            risk_tier           VARCHAR(40),
            agent_id            INTEGER,
            status              VARCHAR(20)  NOT NULL DEFAULT 'pending',
            reason              TEXT,
            estimated_cost_usd  VARCHAR(32),
            requested_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
            expires_at          TIMESTAMPTZ,
            granted_at          TIMESTAMPTZ,
            granted_by          VARCHAR(255),
            revoked_at          TIMESTAMPTZ,
            revoked_by          VARCHAR(255),
            details             JSONB DEFAULT '{}'::jsonb
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_approval_grants_workspace_id "
        "ON approval_grants (workspace_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_approval_grants_subject "
        "ON approval_grants (workspace_id, subject_type, subject_id, status);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_approval_grants_subject;")
    op.execute("DROP INDEX IF EXISTS ix_approval_grants_workspace_id;")
    op.execute("DROP TABLE IF EXISTS approval_grants;")
