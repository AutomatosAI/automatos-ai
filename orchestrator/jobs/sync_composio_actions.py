"""
Composio Action Sync Job
========================

Daily job to refresh Composio action metadata.

Can be run via:
- Cron: python -m jobs.sync_composio_actions
- Celery: celery -A jobs.sync_composio_actions call sync_all_composio_actions
- API endpoint: POST /api/admin/sync-composio-actions

Design:
- Runs daily to detect new/changed actions
- Re-classifies only what changed (hash-based detection)
- Preserves manual overrides
- Logs progress for monitoring
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Job state tracking
_last_sync_result: Optional[Dict[str, Any]] = None
_sync_in_progress: bool = False


async def sync_all_composio_actions(
    force: bool = False
) -> Dict[str, Any]:
    """
    Sync all Composio actions for enabled apps.

    This is the main entry point for the daily sync job.

    Args:
        force: Force re-classification even for unchanged actions

    Returns:
        Dict with sync results and statistics
    """
    global _last_sync_result, _sync_in_progress

    if _sync_in_progress:
        logger.warning("Sync already in progress, skipping")
        return {"status": "skipped", "reason": "sync_in_progress"}

    _sync_in_progress = True
    start_time = datetime.utcnow()

    try:
        from core.database.database import SessionLocal
        from modules.tools.sync import ComposioActionSyncService

        logger.info("Starting Composio action sync job...")

        db = SessionLocal()

        try:
            # Initialize sync service
            # Note: composio_client would be injected from config
            sync_service = ComposioActionSyncService(
                db=db,
                composio_client=_get_composio_client(),
                llm_client=_get_llm_client(),
            )

            # Sync all enabled apps
            results = await sync_service.sync_all_enabled_apps()

            # Aggregate results
            total_actions = sum(r.total_actions for r in results)
            total_classified = sum(r.classified for r in results)
            total_skipped = sum(r.skipped for r in results)
            total_errors = sum(r.errors for r in results)
            total_duration_ms = sum(r.duration_ms for r in results)

            _last_sync_result = {
                "status": "completed",
                "started_at": start_time.isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
                "duration_ms": total_duration_ms,
                "apps_synced": len(results),
                "total_actions": total_actions,
                "classified": total_classified,
                "skipped": total_skipped,
                "errors": total_errors,
                "per_app_results": [
                    {
                        "app_id": r.app_id,
                        "total": r.total_actions,
                        "classified": r.classified,
                        "skipped": r.skipped,
                        "errors": r.errors,
                        "duration_ms": r.duration_ms,
                    }
                    for r in results
                ],
            }

            logger.info(
                f"Composio sync completed: {len(results)} apps, "
                f"{total_classified} classified, {total_skipped} skipped, "
                f"{total_errors} errors in {total_duration_ms:.0f}ms"
            )

            return _last_sync_result

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Composio sync job failed: {e}")
        _last_sync_result = {
            "status": "failed",
            "started_at": start_time.isoformat(),
            "error": str(e),
        }
        return _last_sync_result

    finally:
        _sync_in_progress = False


async def sync_single_app(
    app_id: str,
    force: bool = False
) -> Dict[str, Any]:
    """
    Sync a single Composio app's actions.

    Args:
        app_id: The Composio app ID (e.g., "slack", "github")
        force: Force re-classification

    Returns:
        Dict with sync result
    """
    from core.database.database import SessionLocal
    from modules.tools.sync import ComposioActionSyncService

    logger.info(f"Syncing single app: {app_id}")

    db = SessionLocal()

    try:
        sync_service = ComposioActionSyncService(
            db=db,
            composio_client=_get_composio_client(),
            llm_client=_get_llm_client(),
        )

        result = await sync_service.sync_app(app_id, force=force)

        return {
            "status": "completed",
            "app_id": result.app_id,
            "total_actions": result.total_actions,
            "classified": result.classified,
            "skipped": result.skipped,
            "errors": result.errors,
            "duration_ms": result.duration_ms,
        }

    except Exception as e:
        logger.error(f"Failed to sync app {app_id}: {e}")
        return {
            "status": "failed",
            "app_id": app_id,
            "error": str(e),
        }

    finally:
        db.close()


def get_sync_status() -> Dict[str, Any]:
    """
    Get the status of the sync job.

    Returns:
        Dict with last sync result and current status
    """
    return {
        "in_progress": _sync_in_progress,
        "last_sync": _last_sync_result,
    }


def _get_composio_client():
    """Get Composio client from config (lazy load)."""
    try:
        from core.composio.client import get_composio_client
        return get_composio_client()
    except ImportError:
        logger.warning("Composio client not available")
        return None


def _get_llm_client():
    """Get LLM client for classification fallback (lazy load)."""
    try:
        from core.llm import create_llm_manager
        return create_llm_manager(service_name="action_classifier")
    except ImportError:
        logger.warning("LLM client not available for classification")
        return None


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point for running the sync job."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync Composio action metadata"
    )
    parser.add_argument(
        "--app",
        type=str,
        help="Sync a single app (default: all enabled apps)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-classification of all actions"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Run sync
    if args.app:
        result = asyncio.run(sync_single_app(args.app, force=args.force))
    else:
        result = asyncio.run(sync_all_composio_actions(force=args.force))

    # Print result
    import json
    print(json.dumps(result, indent=2))

    # Exit with error code if failed
    if result.get("status") == "failed":
        exit(1)


if __name__ == "__main__":
    main()
