"""PRD-252 R4: a ticket's number in its workspace (#0042); one head again.

Night 6: two tickets with the same title could not be told apart, and Auto named
tickets by their place in a list ("tasks 2 and 3"). Every ticket but a mission
step now takes the next number from its workspace's counter when it is inserted
(``core/models/ticket_numbers.py``), and a number is never reused, even after a
delete. A mission step shows its mission card's number and its step, #0051.3
(D5), so it takes none.

- ``board_tasks.workspace_seq``: the number, unique per workspace.
- ``workspace_ticket_counters``: the last number each workspace gave out.
- The backfill numbers existing tickets in the order they were created, after
  any number a workspace already has, and sets each counter to its highest.

Every step is idempotent, so it survives a database ``create_all`` built and the
tolerant replay. It also merges the two heads #852 left (``prd251_wave2`` and
``document_chunks_ingestion_columns``), which share their parent and touch
different tables.

Revision ID: prd252_ticket_numbers
Revises: prd251_wave2, document_chunks_ingestion_columns
Create Date: 2026-10-02
"""
from alembic import op

revision = "prd252_ticket_numbers"
down_revision = ("prd251_wave2", "document_chunks_ingestion_columns")
branch_labels = None
depends_on = None

STEP_SOURCE = "orchestration_task"

BACKFILL = f"""
    WITH numbered AS (
        SELECT t.id,
               COALESCE(given.top, 0)
               + ROW_NUMBER() OVER (PARTITION BY t.workspace_id ORDER BY t.created_at, t.id) AS seq
          FROM board_tasks t
          LEFT JOIN (SELECT workspace_id, MAX(workspace_seq) AS top FROM board_tasks
                      WHERE workspace_seq IS NOT NULL GROUP BY workspace_id) given
            ON given.workspace_id = t.workspace_id
         WHERE t.workspace_seq IS NULL AND t.source_type <> '{STEP_SOURCE}'
    )
    UPDATE board_tasks SET workspace_seq = numbered.seq
      FROM numbered WHERE board_tasks.id = numbered.id
"""

COUNTERS = """
    INSERT INTO workspace_ticket_counters (workspace_id, last_seq)
    SELECT workspace_id, MAX(workspace_seq) FROM board_tasks
     WHERE workspace_seq IS NOT NULL GROUP BY workspace_id
    ON CONFLICT (workspace_id)
    DO UPDATE SET last_seq = GREATEST(workspace_ticket_counters.last_seq, EXCLUDED.last_seq)
"""


def upgrade() -> None:
    op.execute("ALTER TABLE board_tasks ADD COLUMN IF NOT EXISTS workspace_seq INTEGER")
    op.execute("""
        CREATE TABLE IF NOT EXISTS workspace_ticket_counters (
            workspace_id UUID PRIMARY KEY REFERENCES workspaces(id) ON DELETE CASCADE,
            last_seq INTEGER NOT NULL DEFAULT 0
        )
    """)
    op.execute(BACKFILL)
    op.execute(COUNTERS)
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_board_tasks_workspace_seq "
        "ON board_tasks (workspace_id, workspace_seq) WHERE workspace_seq IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_board_tasks_workspace_seq")
    op.execute("DROP TABLE IF EXISTS workspace_ticket_counters")
    op.execute("ALTER TABLE board_tasks DROP COLUMN IF EXISTS workspace_seq")
