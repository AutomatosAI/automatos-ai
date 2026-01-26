#!/usr/bin/env python3
"""
Emergency Sync Script - Populate Cache Now
===========================================

Uses the new MetadataSyncService to populate composio_apps_cache.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:alrckxcy2fgvy7zxzhhv0wa37gtc690w@shortline.proxy.rlwy.net:47906/railway")
os.environ.setdefault("COMPOSIO_API_KEY", "ak_kMTigjogVWkISA6OaL6g")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import after environment is set
from core.database.database import SessionLocal
from core.metadata_cache.sync_service import MetadataSyncService
from core.composio.client import get_composio_client

async def main():
    logger.info("="*60)
    logger.info("EMERGENCY SYNC - Populating Composio Apps Cache")
    logger.info("="*60)
    
    db = SessionLocal()
    client = get_composio_client()
    sync_service = MetadataSyncService(db, client)
    
    try:
        logger.info("Starting full sync...")
        sync_service.full_sync()  # This is synchronous in the existing service
        logger.info("✅ Sync completed!")
        
        # Verify
        from core.metadata_cache.models import ComposioAppsCache
        count = db.query(ComposioAppsCache).count()
        logger.info(f"✅ Cache now has {count} apps")
        
        if count > 800:
            logger.info("SUCCESS! Cache is populated.")
        else:
            logger.warning(f"Warning: Expected 863+ apps, got {count}")
        
    except Exception as e:
        logger.error(f"❌ Sync failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()
    
    logger.info("="*60)

if __name__ == "__main__":
    # Run the sync (note: the existing sync_service.full_sync() is not async)
    main_sync = main()
    # Since the function calls sync_service.full_sync() which is sync, not async
    # Let me fix this
    logger.info("Starting sync...")
    db = SessionLocal()
    client = get_composio_client()
    sync_service = MetadataSyncService(db, client)
    
    try:
        sync_service.full_sync()
        from core.metadata_cache.models import ComposioAppsCache
        count = db.query(ComposioAppsCache).count()
        logger.info(f"✅ Cache now has {count} apps")
    finally:
        db.close()
