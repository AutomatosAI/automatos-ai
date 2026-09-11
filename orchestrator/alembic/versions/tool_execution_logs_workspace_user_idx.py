"""tool_execution_logs — index (workspace_id, user_id) for the team page's activity query.

The team page now groups a workspace's tool runs by member
(``api/team._activity_by_user``: ``WHERE workspace_id = :ws AND user_id IN (…)
GROUP BY user_id``). ``tool_execution_logs`` indexes ``agent_id``, ``status`` and
``executed_at`` — nothing that query filters on — and it is an audit table that
only grows, so every team-page load would sequential-scan it. ``chats`` already
carries ``ix_chats_workspace_user`` on the same pair; this brings the tool log
in line.

Two homes on purpose: the model's ``__table_args__`` gives a create_all-first
install the index (fresh databases are stamped past migrations), and this
migration gives it to every existing database. Same index name in both, so an
upgraded database and a fresh one end up identical and neither path builds it
twice. Idempotent (IF NOT EXISTS); downgrade drops only the index.

Revision ID: tool_execution_logs_workspace_user_idx
Revises: users_last_sign_in_column
Create Date: 2026-09-11
"""

from alembic import op


revision = "tool_execution_logs_workspace_user_idx"
down_revision = "users_last_sign_in_column"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_tool_execution_logs_workspace_user "
        "ON tool_execution_logs (workspace_id, user_id);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_tool_execution_logs_workspace_user;")
