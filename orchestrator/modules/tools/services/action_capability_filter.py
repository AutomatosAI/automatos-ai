"""
Action Capability Filter
========================

Extends ToolCapabilityMapper with action-level filtering for Composio actions.
Uses locally-stored capability metadata (no API calls at runtime).

Design Principles:
- Capability filtering is deterministic and fast (local DB lookup)
- Semantic ranking can be applied AFTER capability filtering
- Destructive actions are excluded unless explicitly requested
- All filtering uses pre-classified metadata from sync time
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_

from ..capabilities.taxonomy import (
    INTENT_TO_CAPABILITIES,
    DESTRUCTIVE_CAPABILITIES,
    get_capabilities_for_intent,
)
from ..capabilities.models import ComposioActionMetadata

logger = logging.getLogger(__name__)


@dataclass
class ActionRecommendation:
    """A recommended action with relevance info."""
    action_id: str
    app_id: str
    capabilities: List[str]
    relevance_score: float
    description: Optional[str]
    destructive: bool
    requires_confirmation: bool


@dataclass
class ActionFilterResult:
    """Result of filtering actions for an intent."""
    intent: str
    extracted_capabilities: List[str]
    total_available: int
    filtered_count: int
    actions: List[ActionRecommendation]
    fallback_used: bool


class ActionCapabilityFilter:
    """
    Filters Composio actions based on capability metadata.

    This class provides:
    - Intent-to-capability extraction
    - Capability-based action filtering (local DB, no API calls)
    - Destructive action exclusion
    - Fallback handling for zero matches
    """

    def __init__(self, db: Session):
        """
        Initialize the filter.

        Args:
            db: SQLAlchemy session
        """
        self.db = db
        self._capability_cache: Dict[str, List[str]] = {}

    async def get_actions_for_intent(
        self,
        intent: str,
        enabled_apps: List[str],
        include_destructive: bool = False,
        max_actions: int = 15
    ) -> ActionFilterResult:
        """
        Get relevant Composio actions for a user intent.

        Uses locally-stored capability metadata - NO external API calls.

        Args:
            intent: The user's intent text (e.g., "send a message to #general")
            enabled_apps: List of app IDs the user has enabled
            include_destructive: Whether to include destructive actions
            max_actions: Maximum number of actions to return

        Returns:
            ActionFilterResult with filtered and ranked actions
        """
        # 1. Extract capabilities from intent
        required_caps = self._intent_to_capabilities(intent)
        logger.debug(f"Extracted capabilities for '{intent}': {required_caps}")

        # 2. Query local metadata
        query = (
            select(ComposioActionMetadata)
            .where(ComposioActionMetadata.app_id.in_(enabled_apps))
        )

        # Filter by capabilities using array overlap
        if required_caps:
            # PostgreSQL array overlap operator
            query = query.where(
                ComposioActionMetadata.capabilities.overlap(required_caps)
            )

        # Exclude destructive actions unless explicitly allowed
        if not include_destructive:
            query = query.where(ComposioActionMetadata.destructive == False)

        # Execute query
        result = self.db.execute(query)
        all_actions = result.scalars().all()

        total_available = len(all_actions)
        logger.debug(f"Found {total_available} matching actions")

        # 3. Handle fallback if no matches
        fallback_used = False
        if not all_actions and required_caps:
            logger.info(f"No capability matches for '{intent}', using fallback")
            all_actions = await self._get_fallback_actions(enabled_apps, max_actions)
            fallback_used = True

        # 4. Rank by relevance
        ranked_actions = self._rank_actions(all_actions, intent, required_caps)

        # 5. Limit results
        top_actions = ranked_actions[:max_actions]

        # 6. Convert to recommendations
        recommendations = [
            ActionRecommendation(
                action_id=action.action_id,
                app_id=action.app_id,
                capabilities=action.capabilities or [],
                relevance_score=score,
                description=action.description_short,
                destructive=action.destructive,
                requires_confirmation=action.requires_confirmation
            )
            for action, score in top_actions
        ]

        return ActionFilterResult(
            intent=intent,
            extracted_capabilities=required_caps,
            total_available=total_available,
            filtered_count=len(recommendations),
            actions=recommendations,
            fallback_used=fallback_used
        )

    def _intent_to_capabilities(self, intent: str) -> List[str]:
        """
        Map user intent to capability tags.

        Uses the predefined INTENT_TO_CAPABILITIES mapping for deterministic,
        fast capability extraction.
        """
        intent_lower = intent.lower()
        matched_caps: Set[str] = set()

        for keyword, capabilities in INTENT_TO_CAPABILITIES.items():
            if keyword in intent_lower:
                matched_caps.update(capabilities)

        # If no matches, return general query capabilities
        if not matched_caps:
            return ["data.query", "search.general"]

        return list(matched_caps)

    def _rank_actions(
        self,
        actions: List[ComposioActionMetadata],
        intent: str,
        required_caps: List[str]
    ) -> List[tuple]:
        """
        Rank actions by relevance to the intent.

        Scoring factors:
        - Capability match count (how many required caps does action have)
        - Keyword match in intent_keywords
        - Confidence of classification
        """
        intent_lower = intent.lower()
        intent_words = set(intent_lower.split())

        scored_actions = []

        for action in actions:
            score = 0.0

            # Factor 1: Capability match count (0-0.4)
            action_caps = set(action.capabilities or [])
            cap_matches = len(action_caps & set(required_caps))
            score += min(0.4, cap_matches * 0.15)

            # Factor 2: Intent keyword matches (0-0.4)
            action_keywords = set(action.intent_keywords or [])
            keyword_matches = len(action_keywords & intent_words)
            score += min(0.4, keyword_matches * 0.1)

            # Factor 3: Classification confidence (0-0.2)
            confidence = action.classification_confidence or 0.5
            score += confidence * 0.2

            scored_actions.append((action, score))

        # Sort by score descending
        scored_actions.sort(key=lambda x: x[1], reverse=True)

        return scored_actions

    async def _get_fallback_actions(
        self,
        enabled_apps: List[str],
        limit: int
    ) -> List[ComposioActionMetadata]:
        """
        Get safe fallback actions when no capability matches.

        Returns general read/list/search actions that are always safe.
        """
        safe_capabilities = ["data.query", "search.general", "message.read", "file.list"]

        query = (
            select(ComposioActionMetadata)
            .where(ComposioActionMetadata.app_id.in_(enabled_apps))
            .where(ComposioActionMetadata.destructive == False)
            .where(ComposioActionMetadata.capabilities.overlap(safe_capabilities))
            .limit(limit)
        )

        result = self.db.execute(query)
        return result.scalars().all()

    def check_action_eligibility(
        self,
        action_id: str,
        intent: str,
        allow_destructive: bool = False
    ) -> tuple[bool, str]:
        """
        Check if a specific action is eligible for an intent.

        This is called at EXECUTION time as a second validation gate
        (per GPT-5.2 recommendation: enforce at both selection AND execution).

        Args:
            action_id: The action to validate
            intent: The original user intent
            allow_destructive: Whether destructive actions are allowed

        Returns:
            (eligible, reason) tuple
        """
        # Get action metadata
        metadata = self.db.execute(
            select(ComposioActionMetadata)
            .where(ComposioActionMetadata.action_id == action_id)
        ).scalar_one_or_none()

        if not metadata:
            return False, f"Unknown action: {action_id}"

        # Extract capabilities from intent
        required_caps = self._intent_to_capabilities(intent)

        # Check capability alignment
        action_caps = set(metadata.capabilities or [])
        if required_caps and not action_caps.intersection(set(required_caps)):
            return False, (
                f"Action capabilities {list(action_caps)} "
                f"don't match intent capabilities {required_caps}"
            )

        # Check destructive action safeguards
        if metadata.destructive and not allow_destructive:
            destructive_keywords = ['delete', 'remove', 'archive', 'revoke', 'destroy']
            intent_lower = intent.lower()
            if not any(kw in intent_lower for kw in destructive_keywords):
                return False, "Destructive action not allowed for non-destructive intent"

        return True, "Action is eligible"

    def get_actions_by_capability(
        self,
        capabilities: List[str],
        enabled_apps: Optional[List[str]] = None,
        include_destructive: bool = False,
        limit: int = 50
    ) -> List[ComposioActionMetadata]:
        """
        Get actions that have specific capabilities.

        Args:
            capabilities: List of capability tags to match
            enabled_apps: Optional filter by app IDs
            include_destructive: Include destructive actions
            limit: Maximum results

        Returns:
            List of matching action metadata
        """
        query = select(ComposioActionMetadata)

        if capabilities:
            query = query.where(
                ComposioActionMetadata.capabilities.overlap(capabilities)
            )

        if enabled_apps:
            query = query.where(
                ComposioActionMetadata.app_id.in_(enabled_apps)
            )

        if not include_destructive:
            query = query.where(ComposioActionMetadata.destructive == False)

        query = query.limit(limit)

        result = self.db.execute(query)
        return result.scalars().all()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about classified actions."""

        total = self.db.query(ComposioActionMetadata).count()

        destructive = self.db.query(ComposioActionMetadata).filter(
            ComposioActionMetadata.destructive == True
        ).count()

        by_app = self.db.execute(
            select(
                ComposioActionMetadata.app_id,
            ).group_by(ComposioActionMetadata.app_id)
        ).all()

        heuristic = self.db.query(ComposioActionMetadata).filter(
            ComposioActionMetadata.classification_method == "heuristic"
        ).count()

        llm = self.db.query(ComposioActionMetadata).filter(
            ComposioActionMetadata.classification_method == "llm"
        ).count()

        return {
            "total_actions": total,
            "destructive_actions": destructive,
            "apps_with_metadata": len(by_app),
            "classified_by_heuristic": heuristic,
            "classified_by_llm": llm,
        }


# Factory function
def get_action_capability_filter(db: Session) -> ActionCapabilityFilter:
    """Get an ActionCapabilityFilter instance."""
    return ActionCapabilityFilter(db)
