"""
Prompt Registry Service (PRD-58)
================================

Central service for loading system prompts from the database.
Replaces hardcoded prompt strings across the orchestrator.

- Caches prompts in memory with configurable TTL
- Falls back to hardcoded defaults if database is unavailable
- Supports A/B testing via traffic_percentage weights
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class CachedPrompt:
    content: str
    version_id: str
    loaded_at: float

    def is_expired(self, ttl: int = 60) -> bool:
        return (time.time() - self.loaded_at) > ttl


class PromptRegistry:
    """
    Central service for loading system prompts.

    Usage:
        prompt = await PromptRegistry.get("routing-classifier", {
            "content": user_message,
            "agents_block": formatted_agents,
        })
    """

    _cache: Dict[str, CachedPrompt] = {}
    _cache_ttl: int = 60  # seconds

    # Populated by seed_system_prompts at startup
    _hardcoded_defaults: Dict[str, str] = {}

    @classmethod
    def register_default(cls, slug: str, content: str) -> None:
        """Register a hardcoded fallback for a prompt slug."""
        cls._hardcoded_defaults[slug] = content

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the in-memory cache (useful after activating a new version)."""
        cls._cache.clear()

    @classmethod
    async def get(cls, slug: str, variables: Optional[Dict[str, str]] = None) -> str:
        """
        Load a system prompt by slug, apply variable substitution.

        1. Check in-memory cache (TTL-based)
        2. Query database for active version
        3. Fall back to hardcoded default
        """
        cached = cls._cache.get(slug)
        if cached and not cached.is_expired(cls._cache_ttl):
            template = cached.content
        else:
            template = cls._load_from_db(slug)
            if template is None:
                template = cls._hardcoded_defaults.get(slug, "")
                if not template:
                    logger.warning("Prompt '%s' not found in database or defaults", slug)

        if variables and template:
            try:
                template = template.format(**variables)
            except KeyError as e:
                logger.error("Missing variable %s for prompt '%s'", e, slug)

        return template

    @classmethod
    async def get_ab_variant(
        cls, slug: str, variables: Optional[Dict[str, str]] = None
    ) -> Tuple[str, str]:
        """
        For A/B testing: returns (prompt_content, version_id).
        Selects version based on traffic_percentage weights.
        """
        versions = cls._load_testing_versions(slug)
        if not versions:
            content = await cls.get(slug, variables)
            return content, "default"

        # Weighted random selection
        total_weight = sum(v["traffic_percentage"] for v in versions)
        if total_weight <= 0:
            content = await cls.get(slug, variables)
            return content, "default"

        roll = random.randint(1, total_weight)
        cumulative = 0
        selected = versions[0]
        for v in versions:
            cumulative += v["traffic_percentage"]
            if roll <= cumulative:
                selected = v
                break

        template = selected["content"]
        if variables and template:
            try:
                template = template.format(**variables)
            except KeyError as e:
                logger.error("Missing variable %s for prompt '%s' variant", e, slug)

        return template, selected["version_id"]

    # ------------------------------------------------------------------
    # Internal DB helpers (synchronous — called inside async wrappers)
    # ------------------------------------------------------------------

    @classmethod
    def _load_from_db(cls, slug: str) -> Optional[str]:
        """Load the active version content from the database."""
        try:
            from core.database.database import SessionLocal
            from core.models.system_prompts import SystemPrompt, SystemPromptVersion

            db: Session = SessionLocal()
            try:
                prompt = db.query(SystemPrompt).filter(SystemPrompt.slug == slug).first()
                if not prompt or not prompt.active_version_id:
                    return None

                version = (
                    db.query(SystemPromptVersion)
                    .filter(SystemPromptVersion.id == prompt.active_version_id)
                    .first()
                )
                if not version:
                    return None

                content = version.content
                cls._cache[slug] = CachedPrompt(
                    content=content,
                    version_id=str(version.id),
                    loaded_at=time.time(),
                )
                return content
            finally:
                db.close()
        except Exception as e:
            logger.warning("Failed to load prompt '%s' from database: %s", slug, e)
            return None

    @classmethod
    def _load_testing_versions(cls, slug: str) -> List[Dict]:
        """Load all versions with traffic_percentage > 0 for A/B testing."""
        try:
            from core.database.database import SessionLocal
            from core.models.system_prompts import SystemPrompt, SystemPromptVersion

            db: Session = SessionLocal()
            try:
                prompt = db.query(SystemPrompt).filter(SystemPrompt.slug == slug).first()
                if not prompt:
                    return []

                versions = (
                    db.query(SystemPromptVersion)
                    .filter(
                        SystemPromptVersion.prompt_id == prompt.id,
                        SystemPromptVersion.status.in_(["active", "testing"]),
                        SystemPromptVersion.traffic_percentage > 0,
                    )
                    .all()
                )

                return [
                    {
                        "version_id": str(v.id),
                        "content": v.content,
                        "traffic_percentage": v.traffic_percentage,
                    }
                    for v in versions
                ]
            finally:
                db.close()
        except Exception as e:
            logger.warning("Failed to load A/B variants for '%s': %s", slug, e)
            return []
