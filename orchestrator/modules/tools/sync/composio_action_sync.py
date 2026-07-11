"""
Composio Action Sync Service
============================

Syncs Composio actions and classifies them for capability-based filtering.

Runs on:
- App enable: When a workspace enables a new Composio app
- Daily refresh: Cron job to detect new/changed actions
- Manual trigger: Admin can force re-sync

Design Principles:
- Classify once at sync time, use forever at runtime
- Preserve manual overrides (never overwrite)
- Hash-based change detection (only re-classify changed actions)
- Batch processing to reduce API calls
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select

from ..capabilities.classifier import ActionClassifier, ActionClassification
from ..capabilities.models import ComposioActionMetadata

logger = logging.getLogger(__name__)


def check_action_metadata_populated(db: Session) -> tuple[bool, str]:
    """Fail-LOUD assertion for the destructive-gate feeder (PRD-194 S6, P2-13).

    ``composio_action_metadata`` backs the destructive-action gate (F018).
    Its feeder was a silent no-op for months — a hardcoded app list plus an
    ``await`` on a synchronous client method meant the daily 04:00 sync
    never wrote a row, and nothing noticed. This check makes that state
    LOUD: **connected apps with an EMPTY metadata table is a failure**, not
    a quiet degrade to the 8-keyword intent heuristic.

    Returns ``(ok, detail)``. ok when there are no active connections
    (cold start — nothing to classify) or when at least one metadata row
    exists. Callers log ``detail`` at ERROR when not ok; the gate reader's
    fail-closed keyword floor (action_capability_filter) stays in place
    either way.
    """
    from core.models.composio import ComposioConnection

    connected = db.execute(
        select(ComposioConnection.id)
        .where(ComposioConnection.status == "active")
        .limit(1)
    ).first()
    if not connected:
        return True, "no active Composio connections — nothing to classify"

    row = db.execute(select(ComposioActionMetadata.action_id).limit(1)).first()
    if row:
        return True, "composio_action_metadata populated for connected apps"
    return False, (
        "connected Composio apps exist but composio_action_metadata is EMPTY — "
        "the destructive gate (F018) is running blind on the keyword heuristic "
        "and the action sync is a silent no-op (P2-13 §1.4.a)"
    )


@dataclass
class SyncResult:
    """Result of syncing an app's actions."""
    app_id: str
    total_actions: int
    classified: int
    skipped: int
    errors: int
    duration_ms: float


@dataclass
class ComposioAction:
    """Represents a Composio action for classification."""
    id: str
    name: str
    description: Optional[str]
    parameters: Optional[Dict[str, Any]]
    app_id: str


class ComposioActionSyncService:
    """
    Syncs Composio actions and classifies them via heuristics/LLM.

    This service is responsible for:
    1. Fetching actions from Composio API
    2. Detecting changes via content hashing
    3. Classifying new/changed actions
    4. Storing metadata in local database
    5. Preserving manual overrides
    """

    def __init__(
        self,
        db: Session,
        composio_client=None,
        llm_client=None,
        batch_size: int = 10
    ):
        """
        Initialize the sync service.

        Args:
            db: SQLAlchemy session
            composio_client: Composio SDK client
            llm_client: Optional LLM client for fallback classification
            batch_size: Number of actions to classify per batch
        """
        self.db = db
        self.composio_client = composio_client
        self.classifier = ActionClassifier(llm_client=llm_client)
        self.batch_size = batch_size

    async def sync_app(
        self,
        app_id: str,
        force: bool = False
    ) -> SyncResult:
        """
        Sync all actions for a Composio app.

        Args:
            app_id: The Composio app identifier (e.g., "slack", "github")
            force: Force re-classification even if action hasn't changed

        Returns:
            SyncResult with statistics
        """
        start_time = datetime.utcnow()
        classified_count = 0
        skipped_count = 0
        error_count = 0

        try:
            # 1. Fetch actions from Composio
            actions = await self._fetch_app_actions(app_id)
            logger.info(f"Fetched {len(actions)} actions for app {app_id}")

            # 2. Check what needs classification
            to_classify: List[ComposioAction] = []

            for action in actions:
                should_classify, reason = await self._should_classify(
                    action, force
                )

                if should_classify:
                    to_classify.append(action)
                    logger.debug(f"Will classify {action.id}: {reason}")
                else:
                    skipped_count += 1
                    logger.debug(f"Skipping {action.id}: {reason}")

            # 3. Classify in batches
            if to_classify:
                for i in range(0, len(to_classify), self.batch_size):
                    batch = to_classify[i:i + self.batch_size]

                    for action in batch:
                        try:
                            await self._classify_and_save(action)
                            classified_count += 1
                        except Exception as e:
                            logger.error(
                                f"Failed to classify {action.id}: {e}"
                            )
                            error_count += 1

                    # Commit after each batch
                    self.db.commit()

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            result = SyncResult(
                app_id=app_id,
                total_actions=len(actions),
                classified=classified_count,
                skipped=skipped_count,
                errors=error_count,
                duration_ms=duration_ms
            )

            logger.info(
                f"Sync complete for {app_id}: "
                f"{classified_count} classified, {skipped_count} skipped, "
                f"{error_count} errors in {duration_ms:.0f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to sync app {app_id}: {e}")
            raise

    async def sync_all_enabled_apps(self) -> List[SyncResult]:
        """
        Sync all apps that any workspace has enabled.

        Returns:
            List of SyncResult for each app
        """
        results = []

        # Get distinct app IDs from enabled tools
        enabled_apps = await self._get_all_enabled_apps()
        logger.info(f"Found {len(enabled_apps)} enabled apps to sync")

        for app_id in enabled_apps:
            try:
                result = await self.sync_app(app_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to sync {app_id}: {e}")
                results.append(SyncResult(
                    app_id=app_id,
                    total_actions=0,
                    classified=0,
                    skipped=0,
                    errors=1,
                    duration_ms=0
                ))

        return results

    async def _fetch_app_actions(self, app_id: str) -> List[ComposioAction]:
        """Fetch all actions for an app from Composio."""

        if not self.composio_client:
            logger.warning("No Composio client configured, returning empty list")
            return []

        try:
            # get_app_actions is a synchronous `def` (core/composio/client.py)
            # and every other call site calls it synchronously. Awaiting it
            # directly raised TypeError for EVERY app — swallowed as a failed
            # SyncResult, so the daily sync wrote nothing for months (P2-13
            # §1.4.a). Offload to a thread — the in-tree idiom for sync
            # Composio calls (see core/composio/tool_executor.py) — rather
            # than forking the client into a fake-async method.
            raw_actions = await asyncio.to_thread(
                self.composio_client.get_app_actions, app_id
            )

            actions = []
            for raw in raw_actions:
                action = ComposioAction(
                    id=raw.get("name") or raw.get("id"),
                    name=raw.get("display_name") or raw.get("name"),
                    description=raw.get("description"),
                    parameters=raw.get("parameters"),
                    app_id=app_id
                )
                actions.append(action)

            return actions

        except Exception as e:
            logger.error(f"Failed to fetch actions for {app_id}: {e}")
            raise

    async def _should_classify(
        self,
        action: ComposioAction,
        force: bool
    ) -> tuple[bool, str]:
        """
        Determine if an action needs (re-)classification.

        Returns:
            (should_classify, reason)
        """
        # Check if we have existing metadata
        existing = self.db.execute(
            select(ComposioActionMetadata)
            .where(ComposioActionMetadata.action_id == action.id)
        ).scalar_one_or_none()

        if not existing:
            return True, "new action"

        # Never overwrite manual overrides unless forced
        if existing.manually_overridden and not force:
            return False, "manually overridden"

        # Check if action has changed using hash
        current_hash = self._hash_action(action)
        if existing.composio_hash == current_hash and not force:
            return False, "unchanged"

        return True, "action changed"

    async def _classify_and_save(self, action: ComposioAction) -> None:
        """Classify an action and save/update metadata."""

        # Classify the action
        classification = self.classifier.classify(
            action_id=action.id,
            description=action.description,
            parameters=action.parameters
        )

        # Compute hash for change detection
        action_hash = self._hash_action(action)

        # Check if record exists
        existing = self.db.execute(
            select(ComposioActionMetadata)
            .where(ComposioActionMetadata.action_id == action.id)
        ).scalar_one_or_none()

        if existing:
            # Update existing record
            existing.capabilities = classification.capabilities
            existing.intent_keywords = classification.intent_keywords
            existing.risk_level = classification.risk_level
            existing.destructive = classification.destructive
            existing.requires_confirmation = classification.requires_confirmation
            existing.required_params = classification.required_params
            existing.optional_params = classification.optional_params
            existing.action_name_normalized = classification.normalized_name
            existing.description_short = classification.short_description
            existing.description_original = action.description
            existing.classification_method = classification.method
            existing.classification_confidence = classification.confidence
            existing.classification_model = classification.model
            existing.composio_hash = action_hash
            existing.synced_at = datetime.utcnow()
            existing.classified_at = datetime.utcnow()
            # Note: manually_overridden stays False unless admin changes it

            logger.debug(f"Updated metadata for {action.id}")

        else:
            # Create new record
            metadata = ComposioActionMetadata(
                app_id=action.app_id,
                action_id=action.id,
                action_name_normalized=classification.normalized_name,
                capabilities=classification.capabilities,
                intent_keywords=classification.intent_keywords,
                risk_level=classification.risk_level,
                destructive=classification.destructive,
                requires_confirmation=classification.requires_confirmation,
                required_params=classification.required_params,
                optional_params=classification.optional_params,
                description_short=classification.short_description,
                description_original=action.description,
                classification_method=classification.method,
                classification_confidence=classification.confidence,
                classification_model=classification.model,
                composio_hash=action_hash,
                synced_at=datetime.utcnow(),
                classified_at=datetime.utcnow(),
            )
            self.db.add(metadata)

            logger.debug(f"Created metadata for {action.id}")

    def _hash_action(self, action: ComposioAction) -> str:
        """Generate a hash of action content for change detection."""

        content = f"{action.id}|{action.description}|{action.parameters}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _get_all_enabled_apps(self) -> List[str]:
        """Distinct app names with an ACTIVE Composio connection anywhere.

        P2-13 (PRD-194 S6): replaces a hardcoded 8-app placeholder that made
        the daily sync classify apps nobody had connected — and none of the
        apps they actually had. ``composio_connections.app_name`` (stored
        uppercase, matching how ``get_app_actions`` is called everywhere
        else) is the ground truth the destructive gate needs classified.
        """
        from core.models.composio import ComposioConnection

        rows = self.db.execute(
            select(ComposioConnection.app_name)
            .where(ComposioConnection.status == "active")
            .distinct()
        ).all()
        return sorted({str(r[0]) for r in rows if r and r[0]})

    async def override_classification(
        self,
        action_id: str,
        capabilities: Optional[List[str]] = None,
        destructive: Optional[bool] = None,
        risk_level: Optional[str] = None,
        requires_confirmation: Optional[bool] = None
    ) -> bool:
        """
        Manually override classification for an action.

        Args:
            action_id: The action to override
            capabilities: New capability tags
            destructive: Override destructive flag
            risk_level: Override risk level
            requires_confirmation: Override confirmation requirement

        Returns:
            True if override was applied
        """
        metadata = self.db.execute(
            select(ComposioActionMetadata)
            .where(ComposioActionMetadata.action_id == action_id)
        ).scalar_one_or_none()

        if not metadata:
            logger.warning(f"Cannot override unknown action: {action_id}")
            return False

        if capabilities is not None:
            metadata.capabilities = capabilities
        if destructive is not None:
            metadata.destructive = destructive
        if risk_level is not None:
            metadata.risk_level = risk_level
        if requires_confirmation is not None:
            metadata.requires_confirmation = requires_confirmation

        metadata.manually_overridden = True
        metadata.updated_at = datetime.utcnow()

        self.db.commit()

        logger.info(f"Applied manual override to {action_id}")
        return True

    def get_sync_stats(self) -> Dict[str, Any]:
        """Get statistics about synced actions."""

        total = self.db.query(ComposioActionMetadata).count()

        by_method = self.db.execute(
            select(
                ComposioActionMetadata.classification_method,
            ).group_by(ComposioActionMetadata.classification_method)
        ).all()

        by_risk = self.db.execute(
            select(
                ComposioActionMetadata.risk_level,
            ).group_by(ComposioActionMetadata.risk_level)
        ).all()

        overridden = self.db.query(ComposioActionMetadata).filter(
            ComposioActionMetadata.manually_overridden == True
        ).count()

        return {
            "total_actions": total,
            "by_classification_method": dict(by_method),
            "by_risk_level": dict(by_risk),
            "manually_overridden": overridden,
        }
