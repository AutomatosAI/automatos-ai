"""Metadata sync service for Composio -> local cache.

Populates:
- composio_apps_cache
- composio_actions_cache
- composio_stats_cache
- composio_sync_jobs

Important:
- Uses the existing Composio SDK wrapper (`core.composio.client`) for fetching
  apps/actions (no httpx-based rewrite wrapper).
- Keep this module safe to import at startup.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

from sqlalchemy import func, or_, cast, String
from sqlalchemy.orm import Session

from core.composio.client import get_composio_client
from core.models.composio_cache import (
    ComposioActionCache,
    ComposioAppCache,
    ComposioStatsCache,
    ComposioSyncJob,
)

logger = logging.getLogger(__name__)

INTERNAL_APP_NAMES = {"RAG", "MEMORY", "NL2SQL", "CODEGRAPH"}


class MetadataSyncService:
    def __init__(self, db: Session):
        self.db = db
        self.client = get_composio_client()

    def run_full_sync(self) -> Dict[str, Any]:
        """Fetch ALL apps and actions and upsert into cache tables.

        IMPORTANT: This must not make one Composio API call per app.
        We fetch actions in bulk (paged) and group them by app for upsert.
        """
        logger.info("Starting full metadata sync...")
        started_at = datetime.utcnow()
        job = ComposioSyncJob(job_type="full_sync", status="running", started_at=started_at)
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        apps_synced = 0
        actions_synced = 0
        errors: List[str] = []

        try:
            logger.info("Step 1/4: Fetching available apps from Composio...")
            apps = self.client.get_available_apps() or []
            logger.info(f"Found {len(apps)} apps from Composio")
            apps_by_name: Dict[str, Dict[str, Any]] = {}
            for app in apps:
                if not isinstance(app, dict):
                    continue
                name = (app.get("name") or "").strip()
                if not name:
                    continue
                apps_by_name[name.upper()] = app

            # Bulk fetch actions (paged) and group by app_name
            logger.info("Step 2/4: Fetching all actions in bulk (this may take a while for 880+ tools)...")
            # Use higher limit per page and more max pages to ensure we get ALL actions
            bulk_actions = self.client.get_all_actions_bulk(limit=1000, max_pages=1000) or []
            logger.info(f"Fetched {len(bulk_actions)} total actions from Composio")
            actions_by_app: Dict[str, List[Dict[str, Any]]] = {}
            for action in bulk_actions:
                if not isinstance(action, dict):
                    continue
                an = (action.get("name") or "").strip()
                app_name = (action.get("app_name") or "").strip().upper()
                if not an or not app_name:
                    continue
                actions_by_app.setdefault(app_name, []).append(action)
            logger.info(f"Grouped actions into {len(actions_by_app)} apps")

            # Upsert apps first (no per-app action fetch)
            logger.info("Step 3/4: Upserting apps into cache...")
            for idx, (app_name, app) in enumerate(apps_by_name.items(), 1):
                try:
                    trigger_count = app.get("trigger_count", 0)
                    triggers = app.get("triggers", [])
                    if trigger_count > 0 or len(triggers) > 0:
                        if app_name in ("SLACK", "GMAIL", "GITHUB"):
                            logger.info(f"  Storing {trigger_count} triggers for {app_name}")
                    elif app_name in ("SLACK", "GMAIL", "GITHUB"):
                        logger.warning(f"⚠️  App {app_name} has 0 triggers - check Composio API / trigger_map")
                    self._upsert_app_only(app_name, app)
                    apps_synced += 1
                    if idx % 50 == 0 or idx == len(apps_by_name):
                        logger.info(f"  Upserted {idx}/{len(apps_by_name)} apps...")
                except Exception as e:
                    logger.error(f"Failed to upsert app {app_name}: {e}")
                    errors.append(f"{app_name}: {e}")

            # Upsert actions (and remove orphans so DB matches Composio bulk exactly)
            logger.info("Step 4/4: Upserting actions into cache (removing orphans)...")
            total_actions = sum(len(actions) for actions in actions_by_app.values())
            action_idx = 0
            batch_size = 100  # Commit in batches to avoid huge transactions
            seen: Set[Tuple[str, str]] = set()  # (app_name, action_name) to avoid bulk duplicates
            for app_name, actions in actions_by_app.items():
                # Only store actions for known apps (avoid stray/unmapped toolkits)
                if app_name not in apps_by_name:
                    continue
                keep_names = {(a.get("name") or "").strip() for a in actions if (a.get("name") or "").strip()}
                if keep_names:
                    deleted = self._delete_orphaned_actions(app_name, keep_names)
                    if deleted and app_name in ("GITHUB", "SLACK", "GMAIL"):
                        logger.info(f"  Removed {deleted} orphaned actions for {app_name}")
                for action in actions:
                    try:
                        name = (action.get("name") or "").strip()
                        if not name:
                            continue
                        key = (app_name, name)
                        if key in seen:
                            continue
                        seen.add(key)
                        self._upsert_action(app_name, name, action)
                        actions_synced += 1
                        action_idx += 1
                        
                        # Commit in batches to avoid huge transactions and memory issues
                        if action_idx % batch_size == 0:
                            self.db.commit()
                            logger.info(f"  Upserted {action_idx}/{total_actions} actions (committed batch)...")
                        elif action_idx % 500 == 0 or action_idx == total_actions:
                            logger.info(f"  Upserted {action_idx}/{total_actions} actions...")
                    except Exception as e:
                        logger.error(f"Failed to upsert action {app_name}.{action.get('name')}: {e}")
                        errors.append(f"{app_name}.{(action or {}).get('name')}: {e}")
                        # Rollback this action but continue
                        try:
                            self.db.rollback()
                        except Exception:
                            pass

            # Update action_count per app from actual DB count (matches bulk after orphan cleanup)
            for app_name, app in apps_by_name.items():
                try:
                    cached = (
                        self.db.query(ComposioAppCache)
                        .filter(ComposioAppCache.app_name == app_name)
                        .first()
                    )
                    if not cached:
                        continue
                    actual = (
                        self.db.query(func.count(ComposioActionCache.id))
                        .filter(ComposioActionCache.app_name == app_name)
                        .scalar()
                        or 0
                    )
                    cached.action_count = int(actual)
                    cached.last_synced_at = datetime.utcnow()
                except Exception as e:
                    errors.append(f"{app_name}: failed to update counts: {e}")

            self.db.commit()

            # Backfill parameter schemas for apps that have empty parameters.
            # The v3 bulk API doesn't return parameter schemas, but the SDK
            # per-app endpoint does (OpenAI function-calling format).
            logger.info("Step 5/5: Backfilling parameter schemas for actions with empty params...")
            backfilled = self._backfill_action_parameters(set(actions_by_app.keys()))
            if backfilled:
                logger.info(f"  Backfilled parameters for {backfilled} actions")

            logger.info("Updating stats cache...")
            self._update_stats()
            job.status = "completed"
            logger.info(
                f"✅ Sync completed: {apps_synced} apps, {actions_synced} actions, "
                f"{backfilled} params backfilled, "
                f"{len(errors)} errors in {int((datetime.utcnow() - started_at).total_seconds())}s"
            )
        except Exception as e:
            logger.exception("Full metadata sync failed")
            errors.append(str(e))
            job.status = "failed"
            # Rollback on error to allow retry
            try:
                self.db.rollback()
            except Exception:
                pass
        finally:
            completed_at = datetime.utcnow()
            job.completed_at = completed_at
            job.duration_ms = int((completed_at - started_at).total_seconds() * 1000)
            job.apps_synced = apps_synced
            job.actions_synced = actions_synced
            job.errors_count = len(errors)
            job.error_details = {"errors": errors} if errors else None
            self.db.commit()

        return {
            "job_id": job.id,
            "status": job.status,
            "apps_synced": apps_synced,
            "actions_synced": actions_synced,
            "errors_count": len(errors),
            "duration_ms": job.duration_ms,
        }

    def run_incremental_sync(self) -> Dict[str, Any]:
        """Lightweight sync placeholder (currently full sync)."""
        return self.run_full_sync()

    def backfill_parameters_only(self, app_names: List[str] | None = None) -> Dict[str, Any]:
        """Backfill parameter schemas without a full sync.

        Fetches per-app actions via the SDK and updates the parameters column
        for actions that currently have empty parameters.

        Args:
            app_names: Specific apps to backfill. If None, backfills all apps
                       with empty parameters (capped at 30).

        Returns:
            Dict with backfill results.
        """
        logger.info(f"Starting parameter backfill (apps={app_names or 'auto'})...")
        started_at = datetime.utcnow()

        if app_names:
            target_apps = {n.upper() for n in app_names}
        else:
            # Find all apps with empty parameters
            rows = (
                self.db.query(ComposioActionCache.app_name)
                .filter(
                    or_(
                        ComposioActionCache.parameters == None,  # noqa: E711
                        cast(ComposioActionCache.parameters, String) == '{}',
                        cast(ComposioActionCache.parameters, String) == 'null',
                    )
                )
                .distinct()
                .all()
            )
            target_apps = {row[0] for row in rows}

        if not target_apps:
            return {"status": "no_apps_need_backfill", "backfilled": 0}

        backfilled = self._backfill_action_parameters(target_apps, max_apps=len(target_apps))
        duration_ms = int((datetime.utcnow() - started_at).total_seconds() * 1000)

        logger.info(f"Parameter backfill done: {backfilled} actions in {duration_ms}ms")
        return {
            "status": "completed",
            "backfilled": backfilled,
            "apps_processed": len(target_apps),
            "duration_ms": duration_ms,
        }

    def get_sync_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        jobs = (
            self.db.query(ComposioSyncJob)
            .order_by(ComposioSyncJob.started_at.desc())
            .limit(limit)
            .all()
        )
        return [j.to_dict() for j in jobs]

    def _sync_single_app(self, app: Dict[str, Any]) -> int:
        """Upsert one app and its actions."""
        slug = (app.get("name") or "").strip()
        if not slug:
            return 0

        app_name = slug.upper()
        app_slug = slug.lower()

        cached = self.db.query(ComposioAppCache).filter(ComposioAppCache.app_name == app_name).first()
        if cached:
            cached.app_slug = app_slug
            cached.display_name = app.get("display_name") or cached.display_name
            cached.description = app.get("description")
            cached.logo_url = app.get("logo_url")
            cached.categories = app.get("categories") or []
            cached.auth_schemes = app.get("auth_schemes") or []
            cached.trigger_count = int(app.get("trigger_count") or 0)
            cached.action_count = int(app.get("action_count") or 0)
            cached.status = "ACTIVE"
            # Persist triggers (no schema change): store in JSONB metadata
            app_meta = dict(cached.app_metadata or {})
            app_meta["triggers"] = app.get("triggers") or []
            cached.app_metadata = app_meta
            cached.last_synced_at = datetime.utcnow()
        else:
            cached = ComposioAppCache(
                app_name=app_name,
                app_slug=app_slug,
                display_name=app.get("display_name") or app_name,
                description=app.get("description"),
                logo_url=app.get("logo_url"),
                categories=app.get("categories") or [],
                auth_schemes=app.get("auth_schemes") or [],
                action_count=int(app.get("action_count") or 0),
                trigger_count=int(app.get("trigger_count") or 0),
                status="ACTIVE",
                app_metadata={"triggers": app.get("triggers") or []},
                last_synced_at=datetime.utcnow(),
            )
            self.db.add(cached)
            self.db.flush()

        actions = self.client.get_app_actions(app_name) or []
        actions_count = 0
        for action in actions:
            name = (action.get("name") or "").strip()
            if not name:
                continue
            actions_count += 1
            self._upsert_action(app_name, name, action)

        cached.action_count = actions_count
        self.db.commit()
        return actions_count

    def _upsert_app_only(self, app_name: str, app: Dict[str, Any]) -> None:
        """Upsert one app row without fetching actions."""
        slug = (app.get("name") or "").strip()
        if not slug:
            return

        app_slug = slug.lower()
        cached = self.db.query(ComposioAppCache).filter(ComposioAppCache.app_name == app_name).first()
        if cached:
            cached.app_slug = app_slug
            cached.display_name = app.get("display_name") or cached.display_name
            cached.description = app.get("description")
            cached.logo_url = app.get("logo_url")
            cached.categories = app.get("categories") or []
            cached.auth_schemes = app.get("auth_schemes") or []
            cached.trigger_count = int(app.get("trigger_count") or 0)
            cached.action_count = int(app.get("action_count") or 0)
            cached.status = "ACTIVE"
            # Persist triggers (no schema change): store in JSONB metadata
            app_meta = dict(cached.app_metadata or {})
            app_meta["triggers"] = app.get("triggers") or []
            cached.app_metadata = app_meta
            cached.last_synced_at = datetime.utcnow()
        else:
            cached = ComposioAppCache(
                app_name=app_name,
                app_slug=app_slug,
                display_name=app.get("display_name") or app_name,
                description=app.get("description"),
                logo_url=app.get("logo_url"),
                categories=app.get("categories") or [],
                auth_schemes=app.get("auth_schemes") or [],
                action_count=int(app.get("action_count") or 0),
                trigger_count=int(app.get("trigger_count") or 0),
                status="ACTIVE",
                app_metadata={"triggers": app.get("triggers") or []},
                last_synced_at=datetime.utcnow(),
            )
            self.db.add(cached)
            self.db.flush()

    def _delete_orphaned_actions(self, app_name: str, keep_action_names: set) -> int:
        """Delete actions for this app that are not in keep_action_names. Returns count deleted."""
        if not keep_action_names:
            return 0
        to_delete = (
            self.db.query(ComposioActionCache)
            .filter(
                ComposioActionCache.app_name == app_name,
                ~ComposioActionCache.action_name.in_(list(keep_action_names)),
            )
        )
        n = to_delete.delete(synchronize_session=False)
        return int(n) if isinstance(n, int) else 0

    def _upsert_action(self, app_name: str, action_name: str, action: Dict[str, Any]) -> None:
        cached = (
            self.db.query(ComposioActionCache)
            .filter(
                ComposioActionCache.app_name == app_name,
                ComposioActionCache.action_name == action_name,
            )
            .first()
        )

        action_slug = action_name.lower().replace("_", "-")
        display_name = action.get("display_name") or action_name
        description = action.get("description") or ""
        parameters = action.get("parameters") or {}
        # Extract additional metadata fields (if present)
        response_schema = action.get("response_schema") or {}
        tags = action.get("tags") or []
        category = action.get("category")
        is_premium = bool(action.get("is_premium", False))
        
        # Store any extra metadata in the action_metadata JSONB column
        # IMPORTANT: Sanitize to avoid circular references
        import json
        action_metadata = {}
        raw_metadata = action.get("metadata") or {}
        if isinstance(raw_metadata, dict):
            # Sanitize by testing JSON serialization (breaks circular refs)
            for k, v in raw_metadata.items():
                try:
                    json.dumps(v, default=str)
                    action_metadata[k] = v
                except (TypeError, ValueError):
                    # Skip non-serializable values (likely circular refs)
                    pass

        if cached:
            cached.action_slug = action_slug
            cached.display_name = display_name
            cached.description = description
            cached.parameters = parameters
            cached.response_schema = response_schema
            cached.tags = tags if isinstance(tags, list) else []
            cached.category = category
            cached.is_premium = is_premium
            cached.action_metadata = action_metadata
            cached.last_synced_at = datetime.utcnow()
        else:
            self.db.add(
                ComposioActionCache(
                    app_name=app_name,
                    action_name=action_name,
                    action_slug=action_slug,
                    display_name=display_name,
                    description=description,
                    parameters=parameters,
                    response_schema=response_schema,
                    tags=tags if isinstance(tags, list) else [],
                    category=category,
                    is_premium=is_premium,
                    action_metadata=action_metadata,
                    last_synced_at=datetime.utcnow(),
                )
            )

    def _backfill_action_parameters(
        self, synced_app_names: Set[str], max_apps: int = 30
    ) -> int:
        """Backfill parameter schemas for actions that have empty parameters.

        The Composio v3 bulk tools API doesn't return parameter schemas.
        This method fetches per-app actions via the SDK (which returns full
        OpenAI function-calling schemas) and updates the parameters column.

        Only processes apps that have actions with empty parameters in the cache.
        Capped at ``max_apps`` to avoid adding minutes to the sync; prioritizes
        apps with the most cached actions (most likely to be used).

        Returns:
            Number of actions that had parameters backfilled.
        """
        # Find apps that have actions with empty parameters, ordered by action count
        apps_needing_params = (
            self.db.query(
                ComposioActionCache.app_name,
                func.count(ComposioActionCache.id).label("cnt"),
            )
            .filter(
                ComposioActionCache.app_name.in_(list(synced_app_names)),
                or_(
                    ComposioActionCache.parameters == None,  # noqa: E711
                    cast(ComposioActionCache.parameters, String) == '{}',
                    cast(ComposioActionCache.parameters, String) == 'null',
                ),
            )
            .group_by(ComposioActionCache.app_name)
            .order_by(func.count(ComposioActionCache.id).desc())
            .limit(max_apps)
            .all()
        )
        apps_to_backfill = [row[0] for row in apps_needing_params]
        if not apps_to_backfill:
            return 0

        logger.info(
            f"  Found {len(apps_needing_params)} apps needing parameter backfill "
            f"(capped at {max_apps}): {apps_to_backfill[:10]}..."
        )

        total_backfilled = 0
        for app_name in apps_to_backfill:
            try:
                # Use SDK per-app call which returns full OpenAI schemas
                actions = self.client.get_app_actions(app_name) or []
                if not actions:
                    continue

                # Build lookup: action_name -> parameters
                param_map = {}
                for action in actions:
                    name = (action.get("name") or "").strip()
                    params = action.get("parameters") or {}
                    if name and params and isinstance(params, dict) and params.get("properties"):
                        param_map[name] = params

                if not param_map:
                    continue

                # Update actions with empty parameters
                empty_actions = (
                    self.db.query(ComposioActionCache)
                    .filter(
                        ComposioActionCache.app_name == app_name,
                        or_(
                            ComposioActionCache.parameters == None,  # noqa: E711
                            cast(ComposioActionCache.parameters, String) == '{}',
                            cast(ComposioActionCache.parameters, String) == 'null',
                        ),
                    )
                    .all()
                )

                app_backfilled = 0
                for cached in empty_actions:
                    if cached.action_name in param_map:
                        cached.parameters = param_map[cached.action_name]
                        app_backfilled += 1

                if app_backfilled:
                    self.db.commit()
                    total_backfilled += app_backfilled
                    logger.info(f"  {app_name}: backfilled {app_backfilled}/{len(empty_actions)} action parameters")

            except Exception as e:
                logger.warning(f"  Failed to backfill parameters for {app_name}: {e}")
                try:
                    self.db.rollback()
                except Exception:
                    pass

        return total_backfilled

    def _update_stats(self) -> None:
        # Stats cache must reflect what the Tools UI reads:
        # - exclude internal apps
        # - count only ACTIVE apps
        total_apps = (
            self.db.query(func.count(ComposioAppCache.id))
            .filter(ComposioAppCache.status == "ACTIVE", ~ComposioAppCache.app_name.in_(INTERNAL_APP_NAMES))
            .scalar()
            or 0
        )
        total_actions = (
            self.db.query(func.count(ComposioActionCache.id))
            .filter(~ComposioActionCache.app_name.in_(INTERNAL_APP_NAMES))
            .scalar()
            or 0
        )

        category_counts: Dict[str, int] = {}
        apps = (
            self.db.query(ComposioAppCache.categories)
            .filter(ComposioAppCache.status == "ACTIVE", ~ComposioAppCache.app_name.in_(INTERNAL_APP_NAMES))
            .all()
        )
        for (cats,) in apps:
            for c in (cats or []):
                category_counts[str(c)] = category_counts.get(str(c), 0) + 1

        payloads = {
            "total_apps": {"count": int(total_apps)},
            "total_actions": {"count": int(total_actions)},
            "categories": category_counts,
            "last_full_sync": {"timestamp": datetime.utcnow().isoformat()},
        }

        for key, value in payloads.items():
            row = self.db.query(ComposioStatsCache).filter(ComposioStatsCache.stat_key == key).first()
            if row:
                row.stat_value = value
                row.last_updated_at = datetime.utcnow()
            else:
                self.db.add(ComposioStatsCache(stat_key=key, stat_value=value))

        self.db.commit()

