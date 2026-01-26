"""
Metadata Sync Scheduler
========================

Background scheduler for daily metadata sync.

Usage:
    Can be run as a standalone process or integrated with main app.
"""

import logging
import schedule
import time
from datetime import datetime

from core.database.database import SessionLocal
from core.composio.client import get_composio_client
from core.metadata_cache.sync_service import MetadataSyncService

logger = logging.getLogger(__name__)


def run_daily_sync():
    """Run daily metadata sync at 2 AM UTC."""
    logger.info("="*60)
    logger.info(f"Starting scheduled metadata sync at {datetime.utcnow()}")
    logger.info("="*60)
    
    db = SessionLocal()
    client = get_composio_client()
    sync_service = MetadataSyncService(db, client)
    
    try:
        sync_service.full_sync()
        logger.info("✅ Scheduled sync completed successfully")
    except Exception as e:
        logger.error(f"❌ Scheduled sync failed: {e}", exc_info=True)
    finally:
        db.close()
    
    logger.info("="*60)


def start_scheduler():
    """Start the background scheduler."""
    logger.info("Starting metadata sync scheduler...")
    logger.info("Daily sync scheduled for 2:00 AM UTC")
    
    # Schedule daily sync at 2 AM UTC
    schedule.every().day.at("02:00").do(run_daily_sync)
    
    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Metadata Sync Scheduler - Starting")
    
    # Run initial sync immediately
    logger.info("Running initial sync before starting scheduler...")
    run_daily_sync()
    
    # Start scheduler
    start_scheduler()
