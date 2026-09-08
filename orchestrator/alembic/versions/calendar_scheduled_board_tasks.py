"""Calendar — a scheduled task can file a board ticket when it fires.

PRD-77 scheduled tasks always opened a chat with their target agent. The
Command Centre calendar now lets an operator (from the board's Create Task
dialog) or Auto (``platform_schedule_task`` with ``deliver_as='board_task'``)
schedule a BOARD TASK for later: the row waits on the calendar and is filed on
the board at fire time, where the dispatcher runs it like any other ticket.

Columns on ``agent_scheduled_tasks``:

- ``deliver_as``          ``'chat'`` (the PRD-77 path, default) | ``'board_task'``
- ``payload``             JSONB — the ticket to file: title, priority, review_mode, tags
- ``created_by_user_id``  the operator who scheduled it (NULL for agent-scheduled
                          rows); the consent actor when the ticket is filed
- ``created_by_agent_id`` and ``target_agent_id`` become NULLABLE: an operator is
  not an agent, and a board ticket may be filed unassigned (inbox).

Idempotent (IF NOT EXISTS / catalog-checked constraint) so a re-run or a
create_all-first boot is safe.
"""

from alembic import op


revision = "calendar_scheduled_board_tasks"
down_revision = "prd237_users_chat_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE agent_scheduled_tasks ALTER COLUMN created_by_agent_id DROP NOT NULL;")
    op.execute("ALTER TABLE agent_scheduled_tasks ALTER COLUMN target_agent_id DROP NOT NULL;")
    op.execute("ALTER TABLE agent_scheduled_tasks ADD COLUMN IF NOT EXISTS created_by_user_id VARCHAR(255);")
    op.execute(
        "ALTER TABLE agent_scheduled_tasks "
        "ADD COLUMN IF NOT EXISTS deliver_as VARCHAR(20) NOT NULL DEFAULT 'chat';"
    )
    op.execute("ALTER TABLE agent_scheduled_tasks ADD COLUMN IF NOT EXISTS payload JSONB;")
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'ck_scheduled_task_deliver_as'
            ) THEN
                ALTER TABLE agent_scheduled_tasks
                    ADD CONSTRAINT ck_scheduled_task_deliver_as
                    CHECK (deliver_as IN ('chat', 'board_task'));
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE agent_scheduled_tasks DROP CONSTRAINT IF EXISTS ck_scheduled_task_deliver_as;")
    op.execute("ALTER TABLE agent_scheduled_tasks DROP COLUMN IF EXISTS payload;")
    op.execute("ALTER TABLE agent_scheduled_tasks DROP COLUMN IF EXISTS deliver_as;")
    op.execute("ALTER TABLE agent_scheduled_tasks DROP COLUMN IF EXISTS created_by_user_id;")
    # created_by_agent_id / target_agent_id stay nullable: NOT NULL cannot be
    # restored while operator-scheduled or unassigned rows exist.
