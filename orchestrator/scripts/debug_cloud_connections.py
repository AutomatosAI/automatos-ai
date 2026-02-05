"""
Debug script to check cloud connections and their mapping to workspaces.

Usage:
    cd orchestrator
    source venv/bin/activate
    python scripts/debug_cloud_connections.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from core.models.composio import ComposioConnection, ComposioEntity
from core.models.cloud_sync import CloudSyncConfig, CloudDocument
from core.config import settings

def main():
    # Create database connection
    engine = create_engine(settings.DATABASE_URL)

    with Session(engine) as db:
        print("\n" + "="*80)
        print("COMPOSIO ENTITIES (Workspaces)")
        print("="*80)
        entities = db.query(ComposioEntity).all()
        if not entities:
            print("❌ No entities found! This is the problem - no workspace entity exists.")
            print("   Solutions:")
            print("   1. Create an entity in the database")
            print("   2. Link connections to an entity when they're created in /tools")
            return

        for entity in entities:
            print(f"\n📁 Entity ID: {entity.id}")
            print(f"   Entity Name: {entity.entity_id}")
            print(f"   Workspace ID: {entity.workspace_id}")

        print("\n" + "="*80)
        print("COMPOSIO CONNECTIONS")
        print("="*80)
        connections = db.query(ComposioConnection).all()
        if not connections:
            print("❌ No connections found!")
            print("   Check /tools page - are Google Drive/Dropbox showing as connected?")
            return

        for conn in connections:
            print(f"\n🔗 Connection ID: {conn.id}")
            print(f"   App: {conn.app_name}")
            print(f"   Status: {conn.status}")
            print(f"   Entity ID: {conn.entity_id}")
            print(f"   Connection ID: {conn.connection_id}")
            print(f"   Connected At: {conn.connected_at}")
            print(f"   Sync Enabled: {conn.sync_enabled}")
            print(f"   Total Docs Synced: {conn.total_documents_synced}")

            # Check if entity exists
            entity = db.query(ComposioEntity).filter(
                ComposioEntity.id == conn.entity_id
            ).first()
            if entity:
                print(f"   ✅ Linked to Entity: {entity.entity_id} (Workspace: {entity.workspace_id})")
            else:
                print(f"   ❌ Entity not found! Connection is orphaned.")

            # Check sync config
            sync_cfg = db.query(CloudSyncConfig).filter(
                CloudSyncConfig.connection_id == conn.id
            ).first()
            if sync_cfg:
                print(f"   ✅ Sync Config: Root folder = {sync_cfg.root_folder_path}")
            else:
                print(f"   ⚠️  No sync config (user hasn't selected a root folder yet)")

        print("\n" + "="*80)
        print("CLOUD DOCUMENTS")
        print("="*80)
        docs = db.query(CloudDocument).all()
        if not docs:
            print("ℹ️  No cloud documents synced yet (expected if user hasn't clicked 'Sync Now')")
        else:
            print(f"Found {len(docs)} synced documents")
            for doc in docs[:5]:  # Show first 5
                print(f"  - {doc.filename} ({doc.file_type}) - {doc.chunk_count} chunks")

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"✅ Entities: {len(entities)}")
        print(f"✅ Connections: {len(connections)}")
        print(f"✅ Cloud Documents: {len(docs)}")

        # Check if connections are properly linked
        orphaned = [c for c in connections if not db.query(ComposioEntity).filter(
            ComposioEntity.id == c.entity_id
        ).first()]
        if orphaned:
            print(f"\n❌ PROBLEM: {len(orphaned)} orphaned connections (no entity link)")
            print("   These connections won't show in the UI!")
            print("   Fix: Update entity_id to link them to a workspace entity")
        else:
            print(f"\n✅ All connections properly linked to entities")

        print("\n" + "="*80)
        print("API ENDPOINT TEST")
        print("="*80)
        print("To test the API endpoint manually:")
        print(f"  curl http://localhost:8000/api/cloud-documents/connections \\")
        print(f"    -H 'Authorization: Bearer YOUR_TOKEN'")
        print("\nCheck browser console in /documents page for API errors")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()
