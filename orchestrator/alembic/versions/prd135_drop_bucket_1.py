"""PRD-135 §12 — Bucket 1: drop 11 backup tables

These are b_*_<date> snapshot copies left over from earlier rename/cleanup
passes (PRD-131, PRD-129, etc). All have:

- Zero rows (verified 2026-04-25 from runtime overlay + pg_class)
- Zero inbound code references (graphify dead-tables report)
- Zero runtime hits across 12-hour pg_stat_statements window
- All FK constraints are outbound (b_table → live), so dropping them
  cannot break any live table.

Tables dropped:
  b_backup_document_chunks_20251024_20260424   (9.3 MB, the largest)
  b_mcp_tools_backup_20260424                  (520 kB)
  b_tools_backup_20260424                      (520 kB)
  b_agent_messages_20260424                    (32 kB)
  b_agent_performance_tracking_20260424        (32 kB)
  b_field_states_20260424                      (32 kB)
  b_field_interactions_20260424                (24 kB)
  b_historical_tasks_20260424                  (16 kB)
  b_task_assignments_20260424                  (16 kB)
  b_agent_runtimes_20260424                    (8 kB)
  b_task_decompositions_20260424               (8 kB)

Total disk reclaimed: ~11 MB.

Rollback: schema-only snapshot at
graphify-out/snapshots/bucket-1-pre-drop.sql captures column types,
constraints, and indexes. Re-creating these tables would NOT restore
data (all zero rows when dropped) but is a one-shot recovery if any
unexpected dependency surfaces. The snapshot is the canonical recovery
artifact, NOT this migration's downgrade() — the downgrade() here is
intentionally a no-op because reconstructing 11 tables blindly carries
its own risk and is never the right move when no data is involved.

Revision ID: prd135_drop_bucket_1
Revises: None  (independent head, matches alembic convention in this repo)
Create Date: 2026-04-25
"""

from alembic import op


# revision identifiers, used by Alembic.
# (Short ID — alembic_version.version_num is varchar(32))
revision = "prd135_drop_bucket_1"
down_revision = None
branch_labels = None
depends_on = None


_BACKUP_TABLES = (
    "b_backup_document_chunks_20251024_20260424",
    "b_mcp_tools_backup_20260424",
    "b_tools_backup_20260424",
    "b_agent_messages_20260424",
    "b_agent_performance_tracking_20260424",
    "b_field_states_20260424",
    "b_field_interactions_20260424",
    "b_historical_tasks_20260424",
    "b_task_assignments_20260424",
    "b_agent_runtimes_20260424",
    "b_task_decompositions_20260424",
)


def upgrade() -> None:
    # CASCADE drops the outbound FK constraints with the table itself.
    # IF EXISTS makes the migration idempotent if a table was already
    # dropped manually.
    for table in _BACKUP_TABLES:
        op.execute(f'DROP TABLE IF EXISTS public."{table}" CASCADE;')


def downgrade() -> None:
    # See module docstring: no-op by design. Restore from
    # graphify-out/snapshots/bucket-1-pre-drop.sql if recovery needed.
    pass
