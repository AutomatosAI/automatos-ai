"""PRD-128: Unified notification system — notification_preferences + notifications tables

Creates the two tables that back the unified notification pipeline:

* ``notification_preferences`` — per-workspace (and optional per-user) routing
  rules. Multiple rows may exist for the same ``(workspace_id, user_id,
  event_type)`` so a single event can fan out to several destinations
  (e.g. both ``in_app`` and ``telegram``). There is deliberately **no**
  unique constraint on that tuple.

* ``notifications`` — in-app notification inbox rows surfaced by the bell
  dropdown in the frontend.

Chained off ``prd127_attachment_ids`` (the most recent standalone head)
rather than ``None`` so the migration advances the alembic head cleanly.
"""

from alembic import op
import sqlalchemy as sa


revision = "prd128_notifications"
down_revision = "prd127_attachment_ids"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---------------------------------------------------------------- prefs
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS notification_preferences (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id    UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            user_id         INTEGER REFERENCES users(id) ON DELETE CASCADE,
            event_type      VARCHAR(50) NOT NULL,
            destination     VARCHAR(30) NOT NULL DEFAULT 'in_app',
            channel_connection_id UUID REFERENCES channel_connections(id) ON DELETE SET NULL,
            enabled         BOOLEAN NOT NULL DEFAULT TRUE,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_notif_prefs_workspace "
        "ON notification_preferences (workspace_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_notif_prefs_user "
        "ON notification_preferences (user_id) WHERE user_id IS NOT NULL;"
    )

    # ------------------------------------------------------- notifications
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS notifications (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id    UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            user_id         INTEGER REFERENCES users(id) ON DELETE CASCADE,
            event_type      VARCHAR(50) NOT NULL,
            title           VARCHAR(255) NOT NULL,
            message         TEXT,
            link_type       VARCHAR(30),
            link_id         TEXT,
            agent_id        INTEGER REFERENCES agents(id) ON DELETE SET NULL,
            agent_name      VARCHAR(100),
            status          VARCHAR(20) NOT NULL DEFAULT 'ok',
            read_at         TIMESTAMPTZ,
            dismissed_at    TIMESTAMPTZ,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_notifications_workspace "
        "ON notifications (workspace_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_notifications_user "
        "ON notifications (user_id) WHERE user_id IS NOT NULL;"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_notifications_unread_ws "
        "ON notifications (workspace_id, created_at DESC) "
        "WHERE read_at IS NULL AND dismissed_at IS NULL;"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_notifications_unread_ws;")
    op.execute("DROP INDEX IF EXISTS ix_notifications_user;")
    op.execute("DROP INDEX IF EXISTS ix_notifications_workspace;")
    op.execute("DROP TABLE IF EXISTS notifications;")

    op.execute("DROP INDEX IF EXISTS ix_notif_prefs_user;")
    op.execute("DROP INDEX IF EXISTS ix_notif_prefs_workspace;")
    op.execute("DROP TABLE IF EXISTS notification_preferences;")
