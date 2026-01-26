#!/usr/bin/env python3
"""Run Composio metadata sync"""
import os
import sys
import logging

# Set env first
os.environ["DATABASE_URL"] = "postgresql://postgres:alrckxcy2fgvy7zxzhhv0wa37gtc690w@shortline.proxy.rlwy.net:47906/railway"
os.environ["COMPOSIO_API_KEY"] = "ak_kMTigjogVWkISA6OaL6g"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting Composio metadata sync...")

try:
    from core.database.database import SessionLocal
    from services.metadata_sync_service import MetadataSyncService
    from services.composio_api_service import ComposioAPIService
    
    db = SessionLocal()
    composio = ComposioAPIService()
    sync_service = MetadataSyncService(db, composio)
    
    logger.info("Fetching apps from Composio...")
    import asyncio
    result = asyncio.run(sync_service.run_full_sync())
    
    logger.info(f"✅ Sync complete!")
    logger.info(f"   Apps synced: {result['apps_synced']}")
    logger.info(f"   Actions synced: {result['actions_synced']}")
    logger.info(f"   Duration: {result['duration_ms']}ms")
    
    # Verify
    from core.models.composio_cache import ComposioAppCache
    count = db.query(ComposioAppCache).count()
    logger.info(f"✅ Cache now has {count} apps total")
    
    db.close()
    
except Exception as e:
    logger.error(f"❌ Sync failed: {e}", exc_info=True)
    sys.exit(1)
