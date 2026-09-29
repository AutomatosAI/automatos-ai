"""document_chunks — the columns ingestion writes, on every schema path (#825).

``document_chunks`` has no model. On the fresh path (``init_fresh_db``) it is
built by ``scripts/init_test_db.init_db()``'s vector-free raw DDL, and the
columns ingestion writes came only from ``core/database/migrations/*.sql`` and
the retired ``init_complete_schema.sql``, which nothing on the boot path runs.
So every database ``init_fresh_db`` built lacks them, and every document upload
fails (``column "embedding" of relation "document_chunks" does not exist``):

- ``parent_content``, ``headers`` and ``workspace_id`` — both ingestion modes
  (``modules/rag/ingestion/manager.py``) insert them;
- ``embedding`` — the local (pgvector) mode inserts it and
  ``pgvector_local_backend`` reads it back. A ``vector`` WITHOUT a fixed
  dimension: the dimension is a runtime setting (``dimensions``, default 2048)
  and a ``vector(N)`` column rejects every other size. Only where the
  ``vector`` type exists (stock Postgres, e.g. CI, has no pgvector);
- ``chunk_type`` — the smart-chunking column (legacy 008) the chunker labels.

Additive and idempotent (``ADD COLUMN IF NOT EXISTS``): a no-op on databases
that already have them (prod, anything built from the old snapshot), and on the
fresh path the tolerant replay applies it after ``init_db()`` has created the
table (``build_schema`` stage 2, re-run in stage 5).

Revision ID: document_chunks_ingestion_columns
Revises: prd251w1_merge_heads
Create Date: 2026-09-29
"""
from alembic import op

revision = "document_chunks_ingestion_columns"
down_revision = "prd251w1_merge_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        DO $$
        BEGIN
            IF to_regclass('document_chunks') IS NULL THEN
                RETURN;
            END IF;
            ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS parent_content TEXT;
            ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS headers JSONB DEFAULT '{}'::jsonb;
            ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS chunk_type VARCHAR(20) DEFAULT 'child';
            ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS workspace_id UUID;
            IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'vector') THEN
                ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding vector;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    """No-op: most databases had these columns before this revision (from the
    old snapshot and the legacy SQL); dropping them would destroy their chunks."""
