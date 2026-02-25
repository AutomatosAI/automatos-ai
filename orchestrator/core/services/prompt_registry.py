"""
PRD-58: Prompt Registry Service
================================

Central registry for system prompts with:
- Database-backed storage (SystemPrompt + SystemPromptVersion)
- 60-second TTL in-memory cache
- Hardcoded fallback for bootstrapping (no DB needed)
- Variable interpolation via str.format_map()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.database.database import SessionLocal

logger = logging.getLogger(__name__)


@dataclass
class CachedPrompt:
    content: str
    fetched_at: float = field(default_factory=time.time)
    ttl: float = 60.0  # seconds

    @property
    def is_stale(self) -> bool:
        return (time.time() - self.fetched_at) > self.ttl


class PromptRegistry:
    """
    Singleton prompt registry with DB-backed storage and in-memory cache.

    Usage:
        from core.services.prompt_registry import prompt_registry
        text = prompt_registry.get("chatbot-friendly", agent_name="Atlas", ...)
    """

    _instance: Optional["PromptRegistry"] = None

    def __init__(self) -> None:
        self._cache: Dict[str, CachedPrompt] = {}

    @classmethod
    def get_instance(cls) -> "PromptRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, slug: str, **variables: Any) -> str:
        """
        Get the active prompt content for *slug*, interpolating any {variables}.
        Falls back to hardcoded defaults if the DB is unavailable.
        """
        raw = self._resolve(slug)
        if not raw:
            logger.warning(f"Prompt '{slug}' not found anywhere, returning empty string")
            return ""

        if variables:
            try:
                return raw.format_map(variables)
            except KeyError as e:
                logger.warning(f"Missing variable {e} for prompt '{slug}', returning raw")
                return raw

        return raw

    def get_raw(self, slug: str) -> Optional[str]:
        """Return the raw (un-interpolated) content for a slug."""
        return self._resolve(slug)

    def clear_cache(self, slug: Optional[str] = None) -> None:
        """Clear cache for one slug or all slugs."""
        if slug:
            self._cache.pop(slug, None)
        else:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve(self, slug: str) -> Optional[str]:
        """Try cache -> DB -> hardcoded fallback."""
        # 1. In-memory cache
        cached = self._cache.get(slug)
        if cached and not cached.is_stale:
            return cached.content

        # 2. Database
        try:
            content = self._load_from_db(slug)
            if content is not None:
                self._cache[slug] = CachedPrompt(content=content)
                return content
        except Exception as e:
            logger.debug(f"DB lookup failed for prompt '{slug}': {e}")

        # 3. Hardcoded fallback
        fallback = _HARDCODED_DEFAULTS.get(slug)
        if fallback is not None:
            self._cache[slug] = CachedPrompt(content=fallback)
            return fallback

        return None

    @staticmethod
    def _load_from_db(slug: str) -> Optional[str]:
        """Load the active version content from the database."""
        from core.models.system_prompts import SystemPrompt, SystemPromptVersion

        db = SessionLocal()
        try:
            prompt = db.query(SystemPrompt).filter(
                SystemPrompt.slug == slug,
                SystemPrompt.is_active == True,
            ).first()
            if not prompt:
                return None

            version = (
                db.query(SystemPromptVersion)
                .filter(
                    SystemPromptVersion.prompt_id == prompt.id,
                    SystemPromptVersion.status == "active",
                )
                .order_by(SystemPromptVersion.version_number.desc())
                .first()
            )
            return version.content if version else None
        finally:
            db.close()


# ========================================================================
# Hardcoded fallback defaults (used when DB is unavailable or empty)
# ========================================================================

_HARDCODED_DEFAULTS: Dict[str, str] = {
    "chatbot-friendly": (
        "You are {agent_name}, a friendly and enthusiastic AI assistant.\n"
        "Your role: {agent_role}\n"
        "Available tools: {tools_list}\n"
        "Be warm, approachable, and helpful."
    ),
    "chatbot-professional": (
        "You are {agent_name}, a professional AI assistant.\n"
        "Your role: {agent_role}\n"
        "Available tools: {tools_list}\n"
        "Be precise, data-driven, and efficient."
    ),
    "chatbot-technical": (
        "You are {agent_name}, a technical AI assistant.\n"
        "Your role: {agent_role}\n"
        "Available tools: {tools_list}\n"
        "Use precise terminology and include code examples."
    ),
    "routing-classifier": (
        "You are a request router. Select the best agent for the user's request.\n"
        "Available agents:\n{available_routes}\n"
        "User message: {message}\n\n"
        "Rules:\n"
        "- Pick a specialized agent when its tools/description clearly match.\n"
        '- Pick "Auto" (ID 0) for platform queries, general questions, greetings, or when unsure.\n'
        "- Prefer Auto over a poor-fit specialized agent.\n\n"
        'Return ONLY JSON: {"agent_id": <int>, "confidence": <float 0-1>}'
    ),
    "task-decomposer": (
        "Break this task into executable sub-tasks.\n"
        "Task: {task_description}\n"
        "Available agents:\n{available_agents}\n"
        "Context:\n{context}\n"
        "Return a JSON array of sub-tasks."
    ),
    "quality-assessor": (
        "Evaluate the quality of this agent output.\n"
        "Task: {task_description}\n"
        "Output:\n{output}\n"
        "Criteria:\n{criteria}\n"
        "Return JSON quality assessment."
    ),
    "nl2sql-generator": (
        "Convert the following natural language question into a SQL query.\n"
        "Question: {question}\n"
        "Schema:\n{schema}\n"
        "Dialect: {dialect}\n"
        "Return ONLY the SQL query. Only SELECT queries allowed."
    ),
}


# Module-level singleton for easy import
prompt_registry = PromptRegistry.get_instance()
