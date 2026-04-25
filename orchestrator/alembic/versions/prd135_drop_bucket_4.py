"""PRD-135 §12 — Bucket 4: drop 4 superseded RAG/memory tables

These were the legacy RAG/memory paths, replaced by current systems:

  vector_documents      (1.0 MB, the largest) — legacy RAG path,
                        replaced by S3 Vectors + document_chunks
                        (alive: 644+ runtime calls).
  document_embeddings   — same legacy RAG path, same replacement.
  agent_memories        — replaced by memory_short_term
                        (alive: 418+ runtime calls).
  analytics_snapshots   — dashboard prototype path, replaced by
                        widgets that read live tables directly.

All 4: zero rows, zero inbound code edges, zero runtime hits in the
12-hour pg_stat_statements observation window.

Smoke test after drop:
  - RAG question on uploaded doc → confirms document_chunks path live
  - Agent recalls prior turn → confirms memory_short_term path live
  - Dashboard widget renders → confirms widget→live-tables path

Rollback: schema-only snapshot at
graphify-out/snapshots/bucket-4-pre-drop.sql captures column types,
constraints, and indexes. Re-creating these tables would NOT restore
data (all zero rows when dropped). The snapshot is the canonical
recovery artifact, NOT this migration's downgrade().

Revision ID: prd135_drop_bucket_4
Revises: None  (independent head, matches alembic convention in this repo)
Create Date: 2026-04-25
"""

from alembic import op


# revision identifiers, used by Alembic.
# (Short ID — alembic_version.version_num is varchar(32))
revision = "prd135_drop_bucket_4"
down_revision = None
branch_labels = None
depends_on = None


_SUPERSEDED_TABLES = (
    "vector_documents",
    "document_embeddings",
    "agent_memories",
    "analytics_snapshots",
)


def upgrade() -> None:
    # CASCADE drops outbound FK constraints with the table itself.
    # IF EXISTS makes the migration idempotent.
    for table in _SUPERSEDED_TABLES:
        op.execute(f'DROP TABLE IF EXISTS public."{table}" CASCADE;')


def downgrade() -> None:
    # See module docstring: no-op by design. Restore from
    # graphify-out/snapshots/bucket-4-pre-drop.sql if recovery needed.
    pass
