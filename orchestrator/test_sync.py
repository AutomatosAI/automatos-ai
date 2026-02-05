#!/usr/bin/env python3
"""
Quick test script to trigger cloud document sync without UI
"""
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.rag.services.cloud_sync_service import CloudSyncService
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from urllib.parse import urlparse

async def test_sync():
    """Test cloud document sync"""

    # Get database URL
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not set")
        return False

    # Create database session
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        # Initialize sync service
        service = CloudSyncService(db)

        # Connection ID 30 (from your logs)
        connection_id = 30
        workspace_id = "ae8320bc-95e1-4de1-bbe9-396bef19cbf8"

        print("=" * 80)
        print("🚀 STARTING CLOUD DOCUMENT SYNC TEST")
        print("=" * 80)
        print(f"Connection ID: {connection_id}")
        print(f"Workspace ID: {workspace_id}")
        print()

        # Trigger sync
        job = await service.sync_folder(connection_id, workspace_id)

        print()
        print("=" * 80)
        print("📊 SYNC RESULTS")
        print("=" * 80)
        print(f"Job ID: {job.id}")
        print(f"Status: {job.status}")
        print(f"Files Synced: {job.files_synced}")
        print(f"Files Skipped: {job.files_skipped}")
        print(f"Files Errored: {job.files_errored}")
        print(f"Total Chunks: {job.total_chunks_created}")

        if job.status == "completed" and job.files_synced > 0:
            print()
            print("✅ SUCCESS! Documents synced and processed!")

            # Check database
            from core.models import Document
            recent_docs = db.query(Document).order_by(Document.id.desc()).limit(3).all()

            print()
            print("Recent documents:")
            for doc in recent_docs:
                status_emoji = "✅" if doc.status == "completed" else "❌"
                print(f"  {status_emoji} ID:{doc.id}, File:{doc.filename}, Status:{doc.status}, Chunks:{doc.chunk_count}")

            return True
        else:
            print()
            print(f"❌ SYNC FAILED or NO FILES SYNCED")
            if job.error_message:
                print(f"Error: {job.error_message}")
            return False

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR OCCURRED")
        print("=" * 80)
        print(f"{e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    print("\n🧪 Cloud Document Sync Test Script\n")
    success = asyncio.run(test_sync())
    print("\n" + "=" * 80)
    if success:
        print("✅ TEST PASSED - Sync working!")
    else:
        print("❌ TEST FAILED - Check errors above")
    print("=" * 80 + "\n")
    sys.exit(0 if success else 1)
