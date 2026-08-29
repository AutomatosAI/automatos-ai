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

    # CodeGraph tables (PRD-62). Like document_chunks, these have NO SQLAlchemy
    # model — codegraph_service.py talks to them via raw SQL, and they exist in
    # prod only through alembic/versions/20260218_fix_codegraph_schema_v2.py.
    # create_all() therefore never builds them, so the F056
    # orchestrator-module-tests job (which runs on stock postgres:15) failed the
    # codegraph tests with 'relation "codegraph_projects" does not exist'. Build
    # the vector-free shape here, mirroring the migration's columns 1:1.
    #
    # The pgvector `embedding` column on codegraph_symbols is deliberately
    # omitted: codegraph_service._ensure_embedding_dimension() adds it at runtime
    # via `ALTER TABLE ... TYPE vector(N)`, which needs the pgvector extension the
    # stock CI service does not have. Tests that exercise that path (full
    # indexing / semantic search) skip when pgvector is unavailable; the
    # table-only tests (list/delete) just need these relations to exist.
    with engine.begin() as conn:
        conn.execute(_raw_sql("""
            CREATE TABLE IF NOT EXISTS codegraph_projects (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                source_type VARCHAR(50) DEFAULT 'github',
                source_url TEXT,
                branch VARCHAR(255) DEFAULT 'main',
                status VARCHAR(50) DEFAULT 'pending',
                total_files INTEGER DEFAULT 0,
                total_symbols INTEGER DEFAULT 0,
                total_relationships INTEGER DEFAULT 0,
                language VARCHAR(50),
                last_indexed TIMESTAMPTZ,
                index_duration_seconds FLOAT,
                exclude_patterns TEXT[] DEFAULT '{}',
                auto_reindex BOOLEAN DEFAULT FALSE,
                workspace_id UUID NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(_raw_sql(
            "CREATE INDEX IF NOT EXISTS idx_codegraph_projects_workspace "
            "ON codegraph_projects (workspace_id)"
        ))
        conn.execute(_raw_sql("""
            CREATE TABLE IF NOT EXISTS codegraph_files (
                id SERIAL PRIMARY KEY,
                project_id INTEGER NOT NULL REFERENCES codegraph_projects(id) ON DELETE CASCADE,
                file_path TEXT NOT NULL,
                file_hash VARCHAR(64),
                file_size INTEGER,
                lines_of_code INTEGER,
                language VARCHAR(50),
                workspace_id UUID NOT NULL,
                indexed_at TIMESTAMPTZ DEFAULT NOW(),
                CONSTRAINT uq_codegraph_files_project_path UNIQUE (project_id, file_path)
            )
        """))
        conn.execute(_raw_sql(
            "CREATE INDEX IF NOT EXISTS idx_codegraph_files_project "
            "ON codegraph_files (project_id)"
        ))
        conn.execute(_raw_sql(
            "CREATE INDEX IF NOT EXISTS idx_codegraph_files_hash "
            "ON codegraph_files (file_hash)"
        ))
        conn.execute(_raw_sql("""
            CREATE TABLE IF NOT EXISTS codegraph_symbols (
                id SERIAL PRIMARY KEY,
                project_id INTEGER NOT NULL REFERENCES codegraph_projects(id) ON DELETE CASCADE,
                symbol_type VARCHAR(50) NOT NULL,
                name VARCHAR(255) NOT NULL,
                qualified_name TEXT,
                file_path TEXT NOT NULL,
                line_number INTEGER,
                signature TEXT,
                docstring TEXT,
                code_snippet TEXT,
                metadata JSONB DEFAULT '{}',
                workspace_id UUID NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(_raw_sql(
            "CREATE INDEX IF NOT EXISTS idx_codegraph_symbols_project "
            "ON codegraph_symbols (project_id)"
        ))
        conn.execute(_raw_sql(
            "CREATE INDEX IF NOT EXISTS idx_codegraph_symbols_name "
            "ON codegraph_symbols (name)"
        ))
        conn.execute(_raw_sql(
            "CREATE INDEX IF NOT EXISTS idx_codegraph_symbols_file "
            "ON codegraph_symbols (project_id, file_path)"
        ))
        conn.execute(_raw_sql("""
            CREATE TABLE IF NOT EXISTS codegraph_relationships (
                id SERIAL PRIMARY KEY,
                project_id INTEGER NOT NULL REFERENCES codegraph_projects(id) ON DELETE CASCADE,
                from_symbol_id INTEGER REFERENCES codegraph_symbols(id) ON DELETE CASCADE,
                to_symbol_id INTEGER REFERENCES codegraph_symbols(id) ON DELETE CASCADE,
                relationship_type VARCHAR(50) NOT NULL,
                metadata JSONB DEFAULT '{}',
                workspace_id UUID NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(_raw_sql(
            "CREATE INDEX IF NOT EXISTS idx_codegraph_rels_project "
            "ON codegraph_relationships (project_id)"
        ))
        conn.execute(_raw_sql("""
            CREATE TABLE IF NOT EXISTS codegraph_query_logs (
                id SERIAL PRIMARY KEY,
                project_id INTEGER REFERENCES codegraph_projects(id) ON DELETE CASCADE,
                query_text TEXT NOT NULL,
                query_type VARCHAR(50),
                results_count INTEGER,
                duration_ms FLOAT,
                workspace_id UUID NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))

        # Model-less tables migrations ALTER but nothing CREATEs (PRD-209
        # schema-drift orphans, 2026-08-29) + kb_types (their FK target; live
        # readers in RAG ingestion / tool registry): live code reads all three
        # (team_access/analytics/knowledge_multimodal · workspace_purge ·
        # nl2sql schema provider). DDL ported verbatim from the retired
        # init_complete_schema.sql. Keep scripts/ci/schema_drift_check.py's
        # RAW_DDL_EXTRAS in sync with this block.
        conn.execute(_raw_sql("""
            CREATE TABLE IF NOT EXISTS kb_types (
                id SERIAL PRIMARY KEY,
                type_name VARCHAR(100) UNIQUE NOT NULL,
                display_name VARCHAR(255) NOT NULL,
                description TEXT,
                icon VARCHAR(50),
                processor_class VARCHAR(255),
                storage_strategy VARCHAR(100),
                supports_embedding BOOLEAN DEFAULT true,
                supports_search BOOLEAN DEFAULT true,
                supports_relationships BOOLEAN DEFAULT false,
                enabled BOOLEAN DEFAULT true,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(_raw_sql("""
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id SERIAL PRIMARY KEY,
                kb_type_id INTEGER REFERENCES kb_types(id) ON DELETE CASCADE,
                parent_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE,
                source_type VARCHAR(100),
                source_id VARCHAR(255),
                title VARCHAR(500),
                content TEXT NOT NULL,
                summary TEXT,
                embedding vector(4096),
                metadata JSONB DEFAULT '{}',
                quality_score FLOAT DEFAULT 0.0,
                importance_score FLOAT DEFAULT 0.0,
                complexity_score FLOAT DEFAULT 0.0,
                confidence_score FLOAT DEFAULT 1.0,
                visibility VARCHAR(50) DEFAULT 'system',
                owner_id VARCHAR(255),
                permissions JSONB DEFAULT '{}',
                status VARCHAR(50) DEFAULT 'active',
                version INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                accessed_at TIMESTAMP,
                indexed_at TIMESTAMP
            )
        """))
        conn.execute(_raw_sql("""
            CREATE TABLE IF NOT EXISTS tool_usage_logs (
                id SERIAL PRIMARY KEY,
                execution_id INTEGER,  -- legacy workflow_executions ref; no FK on the fresh path
                agent_id INTEGER REFERENCES agents(id) NOT NULL,
                tool_id INTEGER NOT NULL,  -- legacy mcp_tools ref (dead table); no FK on the fresh path
                method_called VARCHAR(255),
                input_data JSONB,
                output_data JSONB,
                success BOOLEAN,
                execution_time_ms INTEGER,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        # agent_tool_assignments — the agent↔tool assignment table the marketplace
        # API reads AND writes (api/marketplace.py, cascade_installer, provisioning)
        # yet no model and no migration creates it: it lived only in the retired
        # hand-written init SQL (PRD-209 live-test finding, 2026-08-29) — and even
        # that snapshot was stale: live code uses a TEXT tool_id (app slug) and a
        # created_at column. Shape derived from the code's actual reads/writes.
        conn.execute(_raw_sql("""
            CREATE TABLE IF NOT EXISTS agent_tool_assignments (
                id SERIAL PRIMARY KEY,
                agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE NOT NULL,
                tool_id VARCHAR(255) NOT NULL,  -- app slug ('gmail'); api/marketplace.py LOWER()s it
                credential_id INTEGER REFERENCES credentials(id) ON DELETE SET NULL,
                enabled BOOLEAN DEFAULT TRUE,
                permissions JSON DEFAULT '{}',
                configuration JSON DEFAULT '{}',
                assigned_at TIMESTAMP DEFAULT NOW(),
                created_at TIMESTAMP DEFAULT NOW(),   -- api/marketplace.py INSERTs created_at
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(agent_id, tool_id)
            )
        """))
        conn.execute(_raw_sql("""
            CREATE TABLE IF NOT EXISTS learning_outcomes (
                id SERIAL PRIMARY KEY,
                agent_id INTEGER REFERENCES agents(id),
                outcome_type VARCHAR(100),
                outcome_data JSONB,
                success_rate FLOAT,
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
