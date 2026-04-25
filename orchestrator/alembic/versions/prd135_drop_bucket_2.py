"""PRD-135 §12 — Bucket 2: drop 10 context-engineering experiment tables

Old "context engine" wave that didn't ship. Current context system uses
`documents` / `document_chunks` / `memory_short_term`. None of these 10
tables have any inbound code edges or runtime calls in the 12-hour
pg_stat_statements observation window.

Tables dropped:
  context_examples           (1.6 MB — largest)
  context_optimizations
  context_patterns
  context_queries
  context_sources
  context_templates
  context_usage              (FK to live tables — outbound only)
  context_permissions        (FK outbound)
  entity_clusters
  shared_contexts            (FK outbound)

All 10: zero rows, zero runtime hits, all FKs outbound (dead → live),
so dropping cannot break any live table.

Smoke test after drop: RAG / document retrieval flow. Upload a doc,
ask a question, verify chunks return.

Rollback: schema-only snapshot at
graphify-out/snapshots/bucket-2-pre-drop.sql captures column types,
constraints, and indexes. Re-creating these tables would NOT restore
data (all zero rows when dropped). The snapshot is the canonical
recovery artifact, NOT this migration's downgrade().

Revision ID: prd135_drop_bucket_2
Revises: None  (independent head, matches alembic convention in this repo)
Create Date: 2026-04-25
"""

from alembic import op


# revision identifiers, used by Alembic.
# (Short ID — alembic_version.version_num is varchar(32))
revision = "prd135_drop_bucket_2"
down_revision = None
branch_labels = None
depends_on = None


_CONTEXT_TABLES = (
    "context_examples",
    "context_optimizations",
    "context_patterns",
    "context_queries",
    "context_sources",
    "context_templates",
    "context_usage",
    "context_permissions",
    "entity_clusters",
    "shared_contexts",
)


def upgrade() -> None:
    # CASCADE drops outbound FK constraints with the table itself.
    # IF EXISTS makes the migration idempotent.
    for table in _CONTEXT_TABLES:
        op.execute(f'DROP TABLE IF EXISTS public."{table}" CASCADE;')


def downgrade() -> None:
    # See module docstring: no-op by design. Restore from
    # graphify-out/snapshots/bucket-2-pre-drop.sql if recovery needed.
    pass
