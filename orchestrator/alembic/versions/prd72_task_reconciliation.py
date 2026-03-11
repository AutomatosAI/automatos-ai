"""PRD-72: Task Reconciliation — Stall Detection + Auto-Retry

Add attempt_count and retry_of columns to recipe_executions.
Add partial index on (status, started_at) for stall detection queries.
"""

from alembic import op
import sqlalchemy as sa

revision = "prd72_task_reconciliation"
down_revision = None  # Standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE recipe_executions
            ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 1;

        ALTER TABLE recipe_executions
            ADD COLUMN IF NOT EXISTS retry_of VARCHAR(255);

        CREATE INDEX IF NOT EXISTS ix_recipe_exec_stall
            ON recipe_executions(status, started_at)
            WHERE status IN ('running', 'pending');
    """)


def downgrade() -> None:
    op.execute("""
        DROP INDEX IF EXISTS ix_recipe_exec_stall;
        ALTER TABLE recipe_executions DROP COLUMN IF EXISTS retry_of;
        ALTER TABLE recipe_executions DROP COLUMN IF EXISTS attempt_count;
    """)
