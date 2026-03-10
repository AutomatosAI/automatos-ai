"""
Plugin Context Service
======================

Shared service for loading plugin context into agent system prompts.
Used by agent factory prompt builder (single injection point).

Two-tier architecture:
- Tier 1: Plugin summaries from DB fields only (~200 tokens/plugin)
- Tier 2: Full SKILL.md content from S3 via Redis cache (~2000 tokens/plugin)

PRD-71: Skills and plugins now coexist in the same prompt. Plugins that
have been materialized into Skill records are skipped (their content is
already loaded via the skills path). Non-materialized plugins are loaded
via this service.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class PluginContextService:
    """Loads plugin context for agent system prompts."""

    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_assigned_plugins(
        self, agent_id: int
    ) -> List[Tuple[Any, Any]]:
        """Query assigned plugins for an agent, ordered by priority.

        Returns list of (AgentAssignedPlugin, MarketplacePlugin) tuples.
        """
        from core.models.marketplace_plugins import (
            AgentAssignedPlugin,
            MarketplacePlugin,
        )

        rows = (
            self.db.query(AgentAssignedPlugin, MarketplacePlugin)
            .join(MarketplacePlugin, MarketplacePlugin.id == AgentAssignedPlugin.plugin_id)
            .filter(AgentAssignedPlugin.agent_id == agent_id)
            .order_by(AgentAssignedPlugin.priority.asc())
            .all()
        )
        return rows

    def build_tier1_summary(
        self, plugin_rows: List[Tuple[Any, Any]]
    ) -> str:
        """Build lightweight plugin summary from DB fields only (no S3 calls).

        Returns a markdown section listing available plugins with descriptions.
        ~200 tokens per plugin.
        """
        if not plugin_rows:
            return ""

        lines = [
            "## Available Plugins\n",
            "The following plugins are loaded and available for this task:\n",
        ]

        for _aap, plugin in plugin_rows:
            lines.append(f"### {plugin.name} (`{plugin.slug}`)")
            if plugin.description:
                lines.append(plugin.description)
            if plugin.tags:
                lines.append(f"Tags: {', '.join(plugin.tags)}")
            token_hint = f" (~{plugin.token_estimate} tokens)" if plugin.token_estimate else ""
            lines.append(
                f"Skills: {plugin.skills_count or 0} | "
                f"Commands: {plugin.commands_count or 0}{token_hint}"
            )
            lines.append("")  # blank line separator

        return "\n".join(lines)

    async def build_tier2_content(
        self,
        plugin_rows: List[Tuple[Any, Any]],
        task_context: Optional[str] = None,
        max_plugins: int = 2,
    ) -> str:
        """Load full SKILL.md content from S3/Redis cache for most relevant plugins.

        Selects the top `max_plugins` most relevant to `task_context`, then
        fetches their SKILL.md files and extracts tool schemas from manifest.

        Returns a markdown section with full plugin content.
        """
        if not plugin_rows:
            return ""

        selected = self._select_relevant_plugins(plugin_rows, task_context, max_plugins)

        from core.services.plugin_cache import get_plugin_content_cache

        cache = get_plugin_content_cache()
        sections: List[str] = []

        for _aap, plugin in selected:
            plugin_section = [f"## Plugin: {plugin.name} ({plugin.slug})"]

            # Load SKILL.md
            try:
                skill_md = await cache.get_file(plugin.slug, plugin.version, "SKILL.md")
                if skill_md:
                    plugin_section.append(skill_md)
                    logger.info(
                        "Loaded Tier-2 SKILL.md for %s@%s (%d chars)",
                        plugin.slug, plugin.version, len(skill_md),
                    )
            except Exception as e:
                logger.warning(
                    "Could not load SKILL.md for %s@%s: %s",
                    plugin.slug, plugin.version, e,
                )

            # Extract tool schemas from manifest
            try:
                manifest = await cache.get_manifest(plugin.slug, plugin.version)
                if manifest:
                    tools = manifest.get("tools", [])
                    if tools:
                        plugin_section.append("\n### Available Tools\n")
                        for tool in tools:
                            name = tool.get("name", "unknown")
                            desc = tool.get("description", "")
                            plugin_section.append(f"- **{name}**: {desc}")
            except Exception as e:
                logger.warning(
                    "Could not load manifest for %s@%s: %s",
                    plugin.slug, plugin.version, e,
                )

            sections.append("\n".join(plugin_section))

        if not sections:
            return ""

        header = "# Loaded Plugin Skills\n\nFull content for the most relevant plugins:\n"
        return header + "\n\n".join(sections)

    def build_tier2_content_sync(
        self,
        plugin_rows: List[Tuple[Any, Any]],
        task_context: Optional[str] = None,
        max_plugins: int = 2,
    ) -> str:
        """Synchronous wrapper around build_tier2_content for agent_factory.

        If a running event loop exists, schedules as a task on it.
        Otherwise creates a new loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context (e.g. FastAPI) — create a future
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(
                    asyncio.run,
                    self.build_tier2_content(plugin_rows, task_context, max_plugins),
                ).result(timeout=30)
            return result
        else:
            return asyncio.run(
                self.build_tier2_content(plugin_rows, task_context, max_plugins)
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _select_relevant_plugins(
        self,
        plugin_rows: List[Tuple[Any, Any]],
        task_context: Optional[str],
        max_plugins: int = 2,
    ) -> List[Tuple[Any, Any]]:
        """Score plugins against task_context and return top N.

        Mirrors _select_relevant_skills from agent_factory: keyword scoring
        against plugin name, slug, description, and tags.

        If no task context or no matches, returns first `max_plugins` by priority.
        """
        if not plugin_rows:
            return []

        if len(plugin_rows) <= max_plugins:
            return plugin_rows

        if not task_context:
            return plugin_rows[:max_plugins]

        task_lower = task_context.lower()

        scored = []
        for row in plugin_rows:
            _aap, plugin = row
            score = 0

            # Score from name
            name_lower = plugin.name.lower()
            for word in name_lower.replace("-", " ").replace("_", " ").split():
                if word in task_lower:
                    score += len(word)

            # Score from slug
            slug_lower = plugin.slug.lower()
            for word in slug_lower.replace("-", " ").replace("_", " ").split():
                if word in task_lower:
                    score += len(word)

            # Score from description
            if plugin.description:
                desc_lower = plugin.description.lower()
                desc_words = [w for w in desc_lower.split() if len(w) > 3]
                for word in desc_words[:15]:
                    if word in task_lower:
                        score += len(word)

            # Score from tags
            if plugin.tags:
                for tag in plugin.tags:
                    if tag.lower() in task_lower:
                        score += len(tag) * 2  # tags are high-signal

            scored.append((row, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top N with positive scores
        selected = [row for row, s in scored[:max_plugins] if s > 0]

        # Fallback: if no keyword matches, use priority order
        if not selected:
            logger.info(
                "No keyword matches for plugins — using priority order (top %d)",
                max_plugins,
            )
            selected = plugin_rows[:max_plugins]

        logger.info(
            "Selected %d/%d plugins for Tier-2: %s",
            len(selected),
            len(plugin_rows),
            [p.slug for _, p in selected],
        )
        return selected
