"""
PluginsSection -- Plugin tier-1 summary and tier-2 content for the agent.

Priority 5. Renders non-materialized plugin context so the LLM
understands what plugin capabilities are available. Materialized
plugins (those with ``materialized_skill_ids``) are handled by
SkillsSection instead.
"""

from __future__ import annotations

import logging
from typing import Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)


class PluginsSection(BaseSection):
    """Plugin tier-1 + tier-2 content for the agent's prompt.

    Delegates to ``PluginContextService`` for the heavy lifting:
    - ``build_tier1_summary()`` -- lightweight overview (~200 tokens)
    - ``build_tier2_content_sync()`` -- detailed content (~2000 tokens)

    Only includes non-materialized plugins (those without
    ``materialized_skill_ids``).
    """

    name: str = "plugins"
    priority: int = 5
    max_tokens: Optional[int] = 2000

    async def render(self, ctx: SectionContext) -> str:
        """Render plugin tier-1 + tier-2 blocks."""
        try:
            return self._build(ctx)
        except Exception:
            logger.warning("PluginsSection.render failed", exc_info=True)
            return ""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(self, ctx: SectionContext) -> str:
        agent = ctx.agent
        db = ctx.db_session
        if agent is None or db is None:
            return ""

        agent_id = getattr(agent, "id", None)
        if not agent_id:
            return ""

        from core.services.plugin_context_service import PluginContextService

        plugin_svc = PluginContextService(db)
        plugin_rows = plugin_svc.get_assigned_plugins(agent_id)
        if not plugin_rows:
            return ""

        # Filter to non-materialized plugins only
        non_materialized = []
        for row in plugin_rows:
            _aap, plugin = (
                row if isinstance(row, tuple) else (row, getattr(row, "plugin", row))
            )
            materialized_ids = getattr(plugin, "materialized_skill_ids", None) or []
            if not materialized_ids:
                non_materialized.append(row)

        if not non_materialized:
            return ""

        parts: list[str] = []

        tier1 = plugin_svc.build_tier1_summary(non_materialized)
        if tier1:
            parts.append(tier1)

        task_context = ctx.task_description or ""
        tier2 = plugin_svc.build_tier2_content_sync(
            non_materialized, task_context=task_context
        )
        if tier2:
            parts.append(tier2)

        if not parts:
            return ""

        content = "\n\n".join(parts)

        if self.max_tokens:
            content = self.truncate(content, self.max_tokens)
        return content
