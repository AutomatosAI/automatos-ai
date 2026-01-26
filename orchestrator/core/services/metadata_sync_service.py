"""MetadataSyncService - Daily sync of Composio metadata to local cache.

This service handles the background synchronization of Composio metadata
to our local database cache. This is the key to achieving sub-2-second
marketplace loads.

Features:
- Full sync: Fetch all apps and actions from Composio
- Incremental sync: Update only changed data
- Scheduled execution: Runs at 2 AM UTC daily
- Error handling and retry logic
- Sync job tracking and analytics

Usage:
    service = MetadataSyncService(db_session, composio_service)
    await service.run_full_sync()
"""

import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import asyncio

from sqlalchemy.orm import Session
from sqlalchemy import func

from core.services.composio_api_service import ComposioAPIService, ComposioApp, ComposioAction

# Import models
from core.models.composio_cache import (
    ComposioAppCache,
    ComposioActionCache,
    ComposioStatsCache,
    ComposioSyncJob,
)

logger = logging.getLogger(__name__)


class MetadataSyncService:
    """Service for syncing Composio metadata to local cache.
    
    The sync process:
    1. Create sync job record (for tracking)
    2. Fetch all apps from Composio API
    3. Upsert apps into composio_apps_cache
    4. For each app, fetch and cache actions
    5. Update statistics cache
    6. Mark sync job as complete
    
    Error handling:
    - Individual app failures don't stop the sync
    - Errors are logged and tracked in the sync job
    - Retries with exponential backoff for transient failures
    """
    
    def __init__(
        self,
        db_session: Session,
        composio_service: Optional[ComposioAPIService] = None,
    ):
        """Initialize the service.
        
        Args:
            db_session: SQLAlchemy database session
            composio_service: Optional ComposioAPIService instance
        """
        self.db = db_session
        self.composio = composio_service or ComposioAPIService()
        self._sync_job: Optional[ComposioSyncJob] = None
    
    # =========================================================================
    # Main Sync Methods
    # =========================================================================
    
    async def run_full_sync(self) -> Dict[str, Any]:
        """Run a full sync of all Composio metadata.
        
        This fetches ALL apps and actions from Composio and updates
        the local cache. Use this for initial setup or periodic full refresh.
        
        Returns:
            Sync job result with statistics
        """
        logger.info("Starting full metadata sync from Composio")
        start_time = datetime.utcnow()
        
        # Create sync job
        self._sync_job = ComposioSyncJob(
            job_type="full_sync",
            status="running",
            started_at=start_time,
        )
        self.db.add(self._sync_job)
        self.db.commit()
        
        errors = []
        apps_synced = 0
        actions_synced = 0
        
        try:
            # Step 1: Fetch all apps
            logger.info("Fetching apps from Composio...")
            apps = await self.composio.fetch_all_apps()
            logger.info(f"Fetched {len(apps)} apps")
            
            # Step 2: Upsert apps and their actions
            for app in apps:
                try:
                    app_result = await self._sync_app(app)
                    apps_synced += 1
                    actions_synced += app_result.get("actions_count", 0)
                except Exception as e:
                    error_msg = f"Failed to sync app {app.name}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Step 3: Update statistics
            await self._update_stats(apps_synced, actions_synced)
            
            # Mark as complete
            self._sync_job.status = "completed"
            self._sync_job.apps_synced = apps_synced
            self._sync_job.actions_synced = actions_synced
            self._sync_job.errors_count = len(errors)
            self._sync_job.error_details = {"errors": errors} if errors else None
            
        except Exception as e:
            logger.error(f"Full sync failed: {e}")
            self._sync_job.status = "failed"
            self._sync_job.error_details = {"fatal_error": str(e)}
            errors.append(str(e))
        
        finally:
            # Calculate duration
            end_time = datetime.utcnow()
            self._sync_job.completed_at = end_time
            self._sync_job.duration_ms = int((end_time - start_time).total_seconds() * 1000)
            self.db.commit()
            
            await self.composio.close()
        
        result = {
            "job_id": self._sync_job.id,
            "status": self._sync_job.status,
            "apps_synced": apps_synced,
            "actions_synced": actions_synced,
            "errors_count": len(errors),
            "duration_ms": self._sync_job.duration_ms,
        }
        logger.info(f"Full sync completed: {result}")
        return result
    
    async def run_incremental_sync(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Run an incremental sync for recently updated apps.
        
        This is faster than a full sync but may miss some changes.
        
        Args:
            since: Only sync apps updated after this time (default: 24 hours ago)
            
        Returns:
            Sync job result
        """
        since = since or (datetime.utcnow() - timedelta(hours=24))
        logger.info(f"Starting incremental sync since {since}")
        
        start_time = datetime.utcnow()
        self._sync_job = ComposioSyncJob(
            job_type="incremental_sync",
            status="running",
            started_at=start_time,
            metadata={"since": since.isoformat()},
        )
        self.db.add(self._sync_job)
        self.db.commit()
        
        apps_synced = 0
        actions_synced = 0
        errors = []
        
        try:
            # Fetch apps that might have changed
            apps = await self.composio.fetch_all_apps()
            
            for app in apps:
                try:
                    # Check if app needs update
                    cached_app = self.db.query(ComposioAppCache).filter_by(
                        app_name=app.name
                    ).first()
                    
                    if not cached_app or cached_app.last_synced_at < since:
                        app_result = await self._sync_app(app)
                        apps_synced += 1
                        actions_synced += app_result.get("actions_count", 0)
                        
                except Exception as e:
                    errors.append(f"Failed to sync {app.name}: {e}")
            
            await self._update_stats(apps_synced, actions_synced)
            self._sync_job.status = "completed"
            
        except Exception as e:
            self._sync_job.status = "failed"
            self._sync_job.error_details = {"fatal_error": str(e)}
        
        finally:
            end_time = datetime.utcnow()
            self._sync_job.completed_at = end_time
            self._sync_job.duration_ms = int((end_time - start_time).total_seconds() * 1000)
            self._sync_job.apps_synced = apps_synced
            self._sync_job.actions_synced = actions_synced
            self._sync_job.errors_count = len(errors)
            self.db.commit()
            
            await self.composio.close()
        
        return {
            "job_id": self._sync_job.id,
            "status": self._sync_job.status,
            "apps_synced": apps_synced,
            "actions_synced": actions_synced,
        }
    
    async def sync_single_app(self, app_name: str) -> Dict[str, Any]:
        """Sync a single app's metadata.
        
        Useful when a user connects a new app - we can refresh
        just that app's metadata without a full sync.
        
        Args:
            app_name: Name of the app to sync
            
        Returns:
            Sync result for the app
        """
        logger.info(f"Syncing single app: {app_name}")
        
        start_time = datetime.utcnow()
        self._sync_job = ComposioSyncJob(
            job_type="app_sync",
            status="running",
            started_at=start_time,
            metadata={"app_name": app_name},
        )
        self.db.add(self._sync_job)
        self.db.commit()
        
        try:
            # Fetch app actions
            actions = await self.composio.fetch_app_actions(app_name)
            
            # Create/update app cache
            app_cache = self.db.query(ComposioAppCache).filter_by(
                app_name=app_name.upper()
            ).first()
            
            if not app_cache:
                app_cache = ComposioAppCache(
                    app_name=app_name.upper(),
                    app_slug=app_name.lower(),
                    display_name=app_name.title(),
                    action_count=len(actions),
                )
                self.db.add(app_cache)
            else:
                app_cache.action_count = len(actions)
                app_cache.last_synced_at = datetime.utcnow()
            
            # Sync actions
            for action in actions:
                await self._upsert_action(action)
            
            self.db.commit()
            
            self._sync_job.status = "completed"
            self._sync_job.apps_synced = 1
            self._sync_job.actions_synced = len(actions)
            
            result = {
                "app_name": app_name,
                "actions_count": len(actions),
                "status": "success",
            }
            
        except Exception as e:
            logger.error(f"Failed to sync app {app_name}: {e}")
            self._sync_job.status = "failed"
            self._sync_job.error_details = {"error": str(e)}
            result = {"app_name": app_name, "status": "error", "error": str(e)}
        
        finally:
            end_time = datetime.utcnow()
            self._sync_job.completed_at = end_time
            self._sync_job.duration_ms = int((end_time - start_time).total_seconds() * 1000)
            self.db.commit()
            
            await self.composio.close()
        
        return result
    
    # =========================================================================
    # Internal Helper Methods
    # =========================================================================
    
    async def _sync_app(self, app: ComposioApp) -> Dict[str, Any]:
        """Sync a single app and its actions.
        
        Args:
            app: ComposioApp instance
            
        Returns:
            Dict with sync results
        """
        # Upsert app
        app_cache = self.db.query(ComposioAppCache).filter_by(
            app_name=app.name
        ).first()
        
        if app_cache:
            # Update existing
            app_cache.display_name = app.display_name
            app_cache.description = app.description
            app_cache.logo_url = app.logo
            app_cache.categories = app.categories or []
            app_cache.auth_schemes = app.auth_schemes or []
            app_cache.action_count = app.actions_count
            app_cache.trigger_count = app.triggers_count
            app_cache.last_synced_at = datetime.utcnow()
        else:
            # Create new
            app_cache = ComposioAppCache(
                app_name=app.name,
                app_slug=app.key,
                display_name=app.display_name,
                description=app.description,
                logo_url=app.logo,
                categories=app.categories or [],
                auth_schemes=app.auth_schemes or [],
                action_count=app.actions_count,
                trigger_count=app.triggers_count,
                app_type="EXTERNAL",
            )
            self.db.add(app_cache)
        
        self.db.commit()
        
        # Fetch and sync actions
        actions_count = 0
        try:
            actions = await self.composio.fetch_app_actions(app.name)
            for action in actions:
                await self._upsert_action(action)
                actions_count += 1
            
            # Update app's action count
            app_cache.action_count = actions_count
            self.db.commit()
            
        except Exception as e:
            logger.warning(f"Failed to sync actions for {app.name}: {e}")
        
        return {"app_name": app.name, "actions_count": actions_count}
    
    async def _upsert_action(self, action: ComposioAction) -> None:
        """Upsert an action into the cache.
        
        Args:
            action: ComposioAction instance
        """
        action_cache = self.db.query(ComposioActionCache).filter_by(
            app_name=action.app_name.upper(),
            action_name=action.name,
        ).first()
        
        if action_cache:
            # Update existing
            action_cache.display_name = action.display_name
            action_cache.description = action.description
            action_cache.parameters = action.parameters or {}
            action_cache.tags = action.tags or []
            action_cache.last_synced_at = datetime.utcnow()
        else:
            # Create new
            action_cache = ComposioActionCache(
                app_name=action.app_name.upper(),
                action_name=action.name,
                action_slug=action.name.lower().replace("_", "-"),
                display_name=action.display_name,
                description=action.description,
                parameters=action.parameters or {},
                tags=action.tags or [],
                category=self._infer_category(action),
            )
            self.db.add(action_cache)
        
        self.db.commit()
    
    def _infer_category(self, action: ComposioAction) -> str:
        """Infer action category from name or tags.
        
        Args:
            action: ComposioAction instance
            
        Returns:
            Category string
        """
        name_lower = action.name.lower()
        
        # Category inference rules
        if any(kw in name_lower for kw in ["email", "mail", "gmail", "outlook"]):
            return "email"
        elif any(kw in name_lower for kw in ["calendar", "event", "meeting", "schedule"]):
            return "calendar"
        elif any(kw in name_lower for kw in ["slack", "message", "chat", "teams"]):
            return "messaging"
        elif any(kw in name_lower for kw in ["github", "git", "code", "repo", "commit"]):
            return "code"
        elif any(kw in name_lower for kw in ["file", "drive", "document", "storage"]):
            return "files"
        elif any(kw in name_lower for kw in ["task", "project", "jira", "trello"]):
            return "project_management"
        elif any(kw in name_lower for kw in ["crm", "sales", "contact", "lead"]):
            return "crm"
        else:
            return "general"
    
    async def _update_stats(self, apps_count: int, actions_count: int) -> None:
        """Update statistics cache after sync.
        
        Args:
            apps_count: Number of apps synced
            actions_count: Number of actions synced
        """
        # Get actual counts from database
        total_apps = self.db.query(func.count(ComposioAppCache.id)).scalar()
        total_actions = self.db.query(func.count(ComposioActionCache.id)).scalar()
        
        # Category breakdown
        categories = {}
        category_counts = self.db.query(
            ComposioAppCache.categories,
            func.count(ComposioAppCache.id)
        ).group_by(ComposioAppCache.categories).all()
        
        # Get category counts from actions
        action_categories = self.db.query(
            ComposioActionCache.category,
            func.count(ComposioActionCache.id)
        ).group_by(ComposioActionCache.category).all()
        
        for cat, count in action_categories:
            if cat:
                categories[cat] = count
        
        # Upsert stats
        stats_data = {
            "total_apps": {"count": total_apps},
            "total_actions": {"count": total_actions},
            "categories": categories,
            "last_full_sync": {
                "timestamp": datetime.utcnow().isoformat(),
                "apps_synced": apps_count,
                "actions_synced": actions_count,
            },
        }
        
        for key, value in stats_data.items():
            stat = self.db.query(ComposioStatsCache).filter_by(stat_key=key).first()
            if stat:
                stat.stat_value = value
                stat.last_updated_at = datetime.utcnow()
            else:
                stat = ComposioStatsCache(stat_key=key, stat_value=value)
                self.db.add(stat)
        
        self.db.commit()
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_last_sync_time(self) -> Optional[datetime]:
        """Get timestamp of last successful sync.
        
        Returns:
            Datetime of last sync or None
        """
        last_job = self.db.query(ComposioSyncJob).filter_by(
            status="completed"
        ).order_by(ComposioSyncJob.completed_at.desc()).first()
        
        return last_job.completed_at if last_job else None
    
    def get_sync_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sync job history.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of sync job details
        """
        jobs = self.db.query(ComposioSyncJob).order_by(
            ComposioSyncJob.started_at.desc()
        ).limit(limit).all()
        
        return [job.to_dict() for job in jobs]
    
    def needs_sync(self, max_age_hours: int = 24) -> bool:
        """Check if a new sync is needed.
        
        Args:
            max_age_hours: Maximum age of last sync before new one needed
            
        Returns:
            True if sync is needed
        """
        last_sync = self.get_last_sync_time()
        if not last_sync:
            return True
        
        age = datetime.utcnow() - last_sync
        return age > timedelta(hours=max_age_hours)


# =========================================================================
# Scheduled Job Runner
# =========================================================================

async def run_scheduled_sync():
    """Run the scheduled metadata sync.
    
    This function is called by the scheduler (e.g., Celery, APScheduler)
    to perform the daily sync at 2 AM UTC.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import os
    
    # Get database connection
    db_url = os.environ.get("DATABASE_URL", "postgresql://localhost/automatos")
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    
    with Session() as session:
        service = MetadataSyncService(session)
        
        if service.needs_sync():
            logger.info("Starting scheduled metadata sync...")
            result = await service.run_full_sync()
            logger.info(f"Scheduled sync completed: {result}")
        else:
            logger.info("Skipping scheduled sync - recent sync exists")
