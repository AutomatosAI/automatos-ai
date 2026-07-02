"""
Initialize test database for PRD-42 Cloud Document Sync testing.

Creates all necessary tables from SQLAlchemy models without running migrations.
"""

import sys
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database.database import Base, engine
from core.models import *  # Import all models
from core.models.cloud_sync import CloudDocument, CloudSyncConfig, CloudSyncJob
from core.models.composio import ComposioConnection, ComposioEntity
# PRD-79 memory tables (L2 memory_short_term etc.) are defined under
# modules.memory, not core.models, so `from core.models import *` never
# registers them on Base. Import the module so create_all builds memory_short_term
# — the L2 transcript store the memory restart/isolation tests depend on.
import modules.memory.models  # noqa: F401,E402  (registers MemoryShortTerm on Base)

def init_db():
    """Create all tables from models."""
    print("🔧 Creating all database tables...")

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Raw-DDL tables create_all() can't build (no SQLAlchemy model). The RAG chunk
    # store is a pgvector table created in prod via init_complete_schema.sql; the
    # integration tests (read_document / grep / pinned docs) only read
    # document_id / chunk_index / content, and the stock postgres:15 CI service has
    # no pgvector extension — so build the vector-free shape those tests need.
    from sqlalchemy import text as _raw_sql

    with engine.begin() as conn:
        conn.execute(_raw_sql("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))

    # PRD-175: seed the single local workspace that the AUTH_EDITION=local
    # anonymous session resolves to. The CI test job runs the suite under
    # AUTH_EDITION=local + DEFAULT_WORKSPACE_ID (config.py:422); the endpoint
    # tests that fall through to the anonymous dev-fallback
    # (core/auth/hybrid.py) resolve to config.DEFAULT_WORKSPACE_ID and then
    # assert the workspace exists. Read the id from the canonical config (no
    # os.getenv here — CLAUDE.md §4) and seed it idempotently so the row is
    # present whatever DEFAULT_WORKSPACE_ID the environment sets.
    from config import config as _app_config

    _default_ws_id = (_app_config.DEFAULT_WORKSPACE_ID or "").strip()
    if _default_ws_id:
        with engine.begin() as conn:
            conn.execute(
                _raw_sql(
                    "INSERT INTO workspaces (id, name, slug, is_personal, is_active) "
                    "VALUES (:id, 'Local Workspace', 'local', TRUE, TRUE) "
                    "ON CONFLICT (id) DO NOTHING"
                ),
                {"id": _default_ws_id},
            )
        print(f"   ✅ seeded local workspace {_default_ws_id}")

    print("✅ Database initialized successfully!")
    print(f"   Tables created: {len(Base.metadata.tables)}")

    # List some key tables
    key_tables = [
        'workspaces', 'composio_entities', 'composio_connections',
        'cloud_sync_config', 'cloud_documents', 'cloud_sync_jobs'
    ]

    existing = [t for t in key_tables if t in Base.metadata.tables]
    print(f"\n📋 Key tables for PRD-42:")
    for table in existing:
        print(f"   ✅ {table}")

    missing = [t for t in key_tables if t not in Base.metadata.tables]
    if missing:
        print(f"\n⚠️  Missing tables:")
        for table in missing:
            print(f"   ❌ {table}")

if __name__ == "__main__":
    init_db()
