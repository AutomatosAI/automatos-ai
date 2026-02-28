"""RAG v3: hybrid search tsvector, feedback table, knowledge_items workspace_id

Revision ID: 20260218_rag_v3
Revises: 20260215_heartbeat_channels
Create Date: 2026-02-18

Adds:
- search_vector tsvector column to document_chunks for hybrid full-text search
- GIN index and auto-update trigger on search_vector
- rag_feedback table for collecting user feedback on RAG responses
- workspace_id column on knowledge_items for workspace scoping
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20260218_rag_v3'
down_revision = '20260215_heartbeat_channels'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # -------------------------------------------------------------------------
    # 1. Add search_vector tsvector column to document_chunks
    # -------------------------------------------------------------------------
    conn.execute(sa.text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'document_chunks' AND column_name = 'search_vector'
            ) THEN
                ALTER TABLE document_chunks ADD COLUMN search_vector tsvector;
            END IF;
        END
        $$;
    """))

    # -------------------------------------------------------------------------
    # 2. Populate existing rows
    # -------------------------------------------------------------------------
    conn.execute(sa.text("""
        UPDATE document_chunks
        SET search_vector = to_tsvector('english', COALESCE(content, ''))
        WHERE search_vector IS NULL
    """))

    # -------------------------------------------------------------------------
    # 3. GIN index on search_vector
    # -------------------------------------------------------------------------
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_document_chunks_search_vector
        ON document_chunks USING GIN (search_vector)
    """))

    # -------------------------------------------------------------------------
    # 4. Trigger function to auto-update search_vector on INSERT/UPDATE
    # -------------------------------------------------------------------------
    conn.execute(sa.text("""
        CREATE OR REPLACE FUNCTION update_search_vector()
        RETURNS trigger AS $$
        BEGIN
            NEW.search_vector := to_tsvector('english', COALESCE(NEW.content, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """))

    conn.execute(sa.text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_trigger WHERE tgname = 'trg_document_chunks_search_vector'
            ) THEN
                CREATE TRIGGER trg_document_chunks_search_vector
                BEFORE INSERT OR UPDATE OF content ON document_chunks
                FOR EACH ROW
                EXECUTE FUNCTION update_search_vector();
            END IF;
        END
        $$;
    """))

    # -------------------------------------------------------------------------
    # 5. Create rag_feedback table
    # -------------------------------------------------------------------------
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS rag_feedback (
            id SERIAL PRIMARY KEY,
            query TEXT NOT NULL,
            response_text TEXT,
            chunk_ids INTEGER[],
            document_ids INTEGER[],
            rating INTEGER CHECK (rating >= 1 AND rating <= 5),
            feedback_type VARCHAR(50),
            correction_text TEXT,
            rag_config_id INTEGER,
            execution_time_ms FLOAT,
            workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """))

    # -------------------------------------------------------------------------
    # 6. Indexes on rag_feedback
    # -------------------------------------------------------------------------
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_rag_feedback_workspace_id
        ON rag_feedback (workspace_id)
    """))
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_rag_feedback_created_at
        ON rag_feedback (created_at)
    """))

    # -------------------------------------------------------------------------
    # 7. Add workspace_id to knowledge_items (if table exists)
    # -------------------------------------------------------------------------
    conn.execute(sa.text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'knowledge_items'
            ) THEN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'knowledge_items' AND column_name = 'workspace_id'
                ) THEN
                    ALTER TABLE knowledge_items
                    ADD COLUMN workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE;
                END IF;

                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE indexname = 'idx_knowledge_items_workspace_id'
                ) THEN
                    CREATE INDEX idx_knowledge_items_workspace_id
                    ON knowledge_items (workspace_id);
                END IF;
            END IF;
        END
        $$;
    """))


def downgrade() -> None:
    conn = op.get_bind()

    # Remove workspace_id from knowledge_items (if table and column exist)
    conn.execute(sa.text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_items' AND column_name = 'workspace_id'
            ) THEN
                DROP INDEX IF EXISTS idx_knowledge_items_workspace_id;
                ALTER TABLE knowledge_items DROP COLUMN workspace_id;
            END IF;
        END
        $$;
    """))

    # Drop rag_feedback indexes and table
    conn.execute(sa.text("DROP INDEX IF EXISTS idx_rag_feedback_created_at"))
    conn.execute(sa.text("DROP INDEX IF EXISTS idx_rag_feedback_workspace_id"))
    conn.execute(sa.text("DROP TABLE IF EXISTS rag_feedback"))

    # Drop trigger and function
    conn.execute(sa.text("""
        DROP TRIGGER IF EXISTS trg_document_chunks_search_vector ON document_chunks
    """))
    conn.execute(sa.text("DROP FUNCTION IF EXISTS update_search_vector()"))

    # Drop GIN index and search_vector column
    conn.execute(sa.text("DROP INDEX IF EXISTS idx_document_chunks_search_vector"))
    conn.execute(sa.text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'document_chunks' AND column_name = 'search_vector'
            ) THEN
                ALTER TABLE document_chunks DROP COLUMN search_vector;
            END IF;
        END
        $$;
    """))
