"""llm_usage (workspace_id, execution_id) — the index the mission budget reads.

F153: the dispatcher's budget gate sums a run's llm_usage rows by
(workspace_id, execution_id = mission:<run>) on every dispatch check, and
llm_usage had no index on execution_id. It is built CONCURRENTLY, outside the
migration's transaction, so a large llm_usage is not locked during the deploy.
Idempotent (IF NOT EXISTS / IF EXISTS). A concurrent build that failed part-way
leaves an INVALID index of this name, which IF NOT EXISTS would keep for good,
so an invalid one is dropped first.

Revision ID: llm_usage_execution_index
Revises: f156_harness_task_ledger
Create Date: 2026-09-25
"""
import sqlalchemy as sa
from alembic import op

revision = "llm_usage_execution_index"
down_revision = "f156_harness_task_ledger"
branch_labels = None
depends_on = None


INDEX = "idx_llm_usage_workspace_execution"


def upgrade() -> None:
    with op.get_context().autocommit_block():
        invalid = op.get_bind().execute(sa.text(
            "SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE c.relname = :name AND NOT i.indisvalid"), {"name": INDEX}).first()
        if invalid:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX}")
        op.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX} ON llm_usage (workspace_id, execution_id)")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX}")
