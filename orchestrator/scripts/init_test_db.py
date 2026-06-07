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
