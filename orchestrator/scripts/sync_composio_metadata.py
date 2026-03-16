#!/usr/bin/env python3
"""
Sync Composio Metadata Script
==============================

Manually trigger a full metadata sync from Composio API to local cache.

Usage:
    python scripts/sync_composio_metadata.py [--dry-run]

This script:
1. Fetches all apps from Composio API
2. Fetches all actions for each app
3. Updates local cache (PostgreSQL)
4. Invalidates Redis cache
5. Updates statistics
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.database.database import SessionLocal
from core.composio.client import get_composio_client
from services.metadata_sync_service import MetadataSyncService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Sync Composio metadata to local cache')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without committing changes to database'
    )
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Composio Metadata Sync - Starting")
    logger.info("="*60)
    
    if args.dry_run:
        logger.warning("DRY RUN MODE - No changes will be committed")
    
    # Initialize
    db = SessionLocal()
    client = get_composio_client()
    sync_service = MetadataSyncService(db, client)
    
    try:
        # Run full sync
        logger.info("Starting full metadata sync...")
        sync_service.full_sync()
        
        if not args.dry_run:
            logger.info("✅ Sync completed successfully!")
            logger.info("Metadata cache is now populated.")
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Restart the backend to use new v2 endpoints")
            logger.info("2. Test marketplace load time (should be < 2s)")
            logger.info("3. Set up daily cron job for automatic syncing")
        else:
            logger.info("✅ Dry run completed - no changes made")
            db.rollback()
        
    except Exception as e:
        logger.error(f"❌ Sync failed: {e}", exc_info=True)
        db.rollback()
        sys.exit(1)
    finally:
        db.close()
    
    logger.info("="*60)


if __name__ == "__main__":
    main()
