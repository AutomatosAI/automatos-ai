"""PRD-234 S1a — cli_hosts table + board_tasks.runtime_ref (session-mode backend contract)

Two additive objects, one migration:

* ``cli_hosts`` — the paired local processes that run ``runtime: cli`` tickets
  (pairing state + host-token digest + announced capabilities).
* ``board_tasks.runtime_ref`` (JSONB, nullable) — the session reference a claimed
  ticket carries: host id, pre-assigned session id, attempt, provider/model,
  live tool, transcript path, exit reason. Existing rows untouched.

Idempotent (IF NOT EXISTS everywhere) so a re-run or a create_all-first boot is
safe. Local edition only in practice (the feature is boot-gated), but the schema
is the same in every edition — an unused column and an empty table.
"""

from alembic import op


revision = "prd234_s1a_cli_hosts_runtime_ref"
down_revision = "prd_workspace_models_backfill"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS cli_hosts (
            id                 UUID PRIMARY KEY,
            workspace_id       UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            name               VARCHAR(120) NOT NULL DEFAULT 'cli-host',
            status             VARCHAR(16) NOT NULL DEFAULT 'pending',
            pairing_code_hash  VARCHAR(64),
            pairing_expires_at TIMESTAMPTZ,
            token_hash         VARCHAR(64) UNIQUE,
            capabilities       JSONB,
            last_seen_at       TIMESTAMPTZ,
            paired_at          TIMESTAMPTZ,
            revoked_at         TIMESTAMPTZ,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_cli_hosts_workspace_id ON cli_hosts (workspace_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_cli_hosts_workspace_status "
        "ON cli_hosts (workspace_id, status);"
    )
    op.execute("ALTER TABLE board_tasks ADD COLUMN IF NOT EXISTS runtime_ref JSONB;")


def downgrade() -> None:
    op.execute("ALTER TABLE board_tasks DROP COLUMN IF EXISTS runtime_ref;")
    op.execute("DROP TABLE IF EXISTS cli_hosts;")
