"""RAG v3: entity extraction and graph retrieval tables

Revision ID: 20260218_rag_v3_entities
Revises: 20260218_rag_v3
Create Date: 2026-02-18

Adds:
- document_entities table for extracted named entities per document
- document_relationships table for entity-to-entity graph edges
- Indexes for efficient lookups by document, workspace, entity name, and graph traversal
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20260218_rag_v3_entities'
down_revision = '20260218_rag_v3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # -------------------------------------------------------------------------
    # 1. Create document_entities table
    # -------------------------------------------------------------------------
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS document_entities (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            entity_type VARCHAR(50),
            confidence FLOAT,
            mention_count INTEGER DEFAULT 1,
            workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(document_id, name, entity_type)
        )
    """))

    # -------------------------------------------------------------------------
    # 2. Indexes on document_entities
    # -------------------------------------------------------------------------
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_document_entities_document_id
        ON document_entities (document_id)
    """))
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_document_entities_workspace_id
        ON document_entities (workspace_id)
    """))
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_document_entities_name
        ON document_entities (name)
    """))

    # -------------------------------------------------------------------------
    # 3. Create document_relationships table
    # -------------------------------------------------------------------------
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS document_relationships (
            id SERIAL PRIMARY KEY,
            source_entity_id INTEGER REFERENCES document_entities(id) ON DELETE CASCADE,
            target_entity_id INTEGER REFERENCES document_entities(id) ON DELETE CASCADE,
            relationship_type VARCHAR(100),
            weight FLOAT DEFAULT 1.0,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            metadata JSONB DEFAULT '{}',
            workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """))

    # -------------------------------------------------------------------------
    # 4. Indexes on document_relationships
    # -------------------------------------------------------------------------
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_document_relationships_source
        ON document_relationships (source_entity_id)
    """))
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_document_relationships_target
        ON document_relationships (target_entity_id)
    """))
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_document_relationships_document_id
        ON document_relationships (document_id)
    """))


def downgrade() -> None:
    conn = op.get_bind()

    # Drop document_relationships indexes and table first (FK dependency)
    conn.execute(sa.text("DROP INDEX IF EXISTS idx_document_relationships_document_id"))
    conn.execute(sa.text("DROP INDEX IF EXISTS idx_document_relationships_target"))
    conn.execute(sa.text("DROP INDEX IF EXISTS idx_document_relationships_source"))
    conn.execute(sa.text("DROP TABLE IF EXISTS document_relationships"))

    # Drop document_entities indexes and table
    conn.execute(sa.text("DROP INDEX IF EXISTS idx_document_entities_name"))
    conn.execute(sa.text("DROP INDEX IF EXISTS idx_document_entities_workspace_id"))
    conn.execute(sa.text("DROP INDEX IF EXISTS idx_document_entities_document_id"))
    conn.execute(sa.text("DROP TABLE IF EXISTS document_entities"))
