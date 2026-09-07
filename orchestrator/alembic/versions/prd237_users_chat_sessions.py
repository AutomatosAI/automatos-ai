"""PRD-237 S6 — users.chat_sessions: the hosted edition's open-conversation tabs

One nullable JSONB column on ``users``, keyed by workspace id::

    {"<workspace_uuid>": {"activeChatId": "...", "draftOpen": false,
                          "openChatIds": ["..."], "lastReadAt": {"...": 1725...},
                          "updatedAt": "2026-09-07T20:00:00"}}

Per user, per workspace, no membership dependency (personal-workspace owners
have no ``workspace_members`` row). The local edition keeps its session in the
browser only (owner decision D1, 2026-09-07) — the column simply stays NULL
there. Idempotent (IF NOT EXISTS) so a re-run or a create_all-first boot is safe.
"""

from alembic import op


revision = "prd237_users_chat_sessions"
down_revision = "prd234_s1a_cli_hosts_runtime_ref"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS chat_sessions JSONB;")


def downgrade() -> None:
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS chat_sessions;")
