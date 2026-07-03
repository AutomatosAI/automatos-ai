"""PRD-181 S1 — audit_logs: nullable user_id + actor_type (Art.12 record-keeping)

Every policy verdict (allow / ask / deny) is now recorded per tenant, including
tool calls made by non-human actors (agent / heartbeat / scheduled). Those have
no ``users`` row, so:

  - ``user_id`` becomes NULLABLE (a NULL user is a non-human actor).
  - the FK to ``users`` is recreated ``ON DELETE SET NULL`` so a GDPR user
    erasure never orphans an audit row on a dangling FK.
  - ``actor_type`` ('user' | 'agent' | 'system') records which; the fine-grained
    identity lives in ``details`` (agent id, trace, error code).

Idempotent. Chained onto the Wave 1/3 hardening lineage.
"""

from alembic import op


revision = "prd181_s1_audit_actor"
down_revision = "wave3_escalation_level"
branch_labels = None
depends_on = None


_FK_NAME = "audit_logs_user_id_fkey"


def upgrade() -> None:
    # 1) user_id nullable — a non-human actor has no users row.
    op.execute("ALTER TABLE audit_logs ALTER COLUMN user_id DROP NOT NULL;")

    # 2) actor_type — default 'user' so existing rows read as human actions.
    op.execute(
        "ALTER TABLE audit_logs "
        "ADD COLUMN IF NOT EXISTS actor_type VARCHAR(20) NOT NULL DEFAULT 'user';"
    )

    # 3) recreate the users FK as ON DELETE SET NULL (was the default NO ACTION),
    #    so erasing a user leaves the audit trail intact with actor context in
    #    details. Drop-if-exists then add — idempotent across replays.
    op.execute(f"ALTER TABLE audit_logs DROP CONSTRAINT IF EXISTS {_FK_NAME};")
    op.execute(
        f"ALTER TABLE audit_logs ADD CONSTRAINT {_FK_NAME} "
        "FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL;"
    )

    # 4) an index for the per-tenant Art.12 read (verdict stream by workspace).
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_audit_logs_ws_action "
        "ON audit_logs (workspace_id, action, created_at);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_audit_logs_ws_action;")
    # Restore the plain FK (NO ACTION) and drop actor_type. user_id is left
    # nullable — reintroducing NOT NULL would fail if non-human rows exist.
    op.execute(f"ALTER TABLE audit_logs DROP CONSTRAINT IF EXISTS {_FK_NAME};")
    op.execute(
        f"ALTER TABLE audit_logs ADD CONSTRAINT {_FK_NAME} "
        "FOREIGN KEY (user_id) REFERENCES users (id);"
    )
    op.execute("ALTER TABLE audit_logs DROP COLUMN IF EXISTS actor_type;")
