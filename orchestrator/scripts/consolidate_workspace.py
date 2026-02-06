"""
Consolidate all connections into a single workspace entity.

This script:
1. Identifies the main workspace entity (the one with most connections)
2. Moves all connections from other entities to the main entity
3. Removes duplicate/test entities

Usage:
    cd orchestrator
    source venv/bin/activate
    python scripts/consolidate_workspace.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from core.models.composio import ComposioConnection, ComposioEntity
from core.config import settings

def main():
    engine = create_engine(settings.DATABASE_URL)

    with Session(engine) as db:
        print("\n" + "="*80)
        print("WORKSPACE CONSOLIDATION")
        print("="*80)

        # Get all entities
        entities = db.query(ComposioEntity).all()
        print(f"\nFound {len(entities)} entities:")
        for entity in entities:
            conn_count = db.query(ComposioConnection).filter(
                ComposioConnection.entity_id == entity.id
            ).count()
            print(f"  Entity {entity.id}: {entity.workspace_id} ({conn_count} connections)")

        if len(entities) <= 1:
            print("\n✅ Only one entity found - no consolidation needed")
            return

        # Find the entity with most connections (main workspace)
        main_entity = max(entities, key=lambda e: db.query(ComposioConnection).filter(
            ComposioConnection.entity_id == e.id
        ).count())

        print(f"\n📌 Main workspace: Entity {main_entity.id} ({main_entity.workspace_id})")

        # Move all connections to main entity
        for entity in entities:
            if entity.id == main_entity.id:
                continue

            connections = db.query(ComposioConnection).filter(
                ComposioConnection.entity_id == entity.id
            ).all()

            if connections:
                print(f"\n🔄 Moving {len(connections)} connections from Entity {entity.id} to Entity {main_entity.id}:")
                for conn in connections:
                    print(f"   - {conn.app_name} (ID: {conn.id})")
                    conn.entity_id = main_entity.id

        # Commit changes
        db.commit()
        print("\n✅ All connections consolidated to main workspace!")

        # Verify
        print("\n" + "="*80)
        print("VERIFICATION")
        print("="*80)
        all_connections = db.query(ComposioConnection).filter(
            ComposioConnection.entity_id == main_entity.id
        ).all()
        print(f"\nMain workspace now has {len(all_connections)} connections:")
        for conn in all_connections:
            print(f"  ✓ {conn.app_name} (ID: {conn.id})")

        print("\n" + "="*80)
        print("✅ CONSOLIDATION COMPLETE")
        print("="*80)
        print(f"\nAll tools now use workspace: {main_entity.workspace_id}")
        print("\nNext steps:")
        print("1. Restart the backend server")
        print("2. Refresh /documents page")
        print("3. You should see Google Drive and Dropbox cards!")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()
