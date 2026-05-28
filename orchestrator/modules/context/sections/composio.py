"""
ComposioSection -- Connected external apps (Composio) for the agent.

Priority 5. Renders a markdown block listing the agent's assigned
Composio apps with descriptions, so the LLM knows what external
integrations are available via ``composio_execute``.
"""

from __future__ import annotations

import logging
from typing import Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)


class ComposioSection(BaseSection):
    """Composio external-app descriptions for the agent's prompt.

    Queries ``AgentAppAssignment`` for active EXTERNAL apps, then looks
    up cached descriptions from ``ComposioAppCache``.
    """

    name: str = "composio"
    priority: int = 5
    max_tokens: Optional[int] = None

    def __init__(self) -> None:
        super().__init__()
        from config import config
        self.max_tokens = config.COMPOSIO_SECTION_MAX_TOKENS

    async def render(self, ctx: SectionContext) -> str:
        """Render connected Composio apps block."""
        try:
            return self._build(ctx)
        except Exception:
            logger.warning("ComposioSection.render failed", exc_info=True)
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

        from core.models.composio_cache import AgentAppAssignment, ComposioAppCache

        assignments = (
            db.query(AgentAppAssignment)
            .filter(
                AgentAppAssignment.agent_id == agent_id,
                AgentAppAssignment.is_active.is_(True),
                AgentAppAssignment.app_type == "EXTERNAL",
            )
            .all()
        )

        if not assignments:
            return ""

        # Look up cached app descriptions
        app_names = [a.app_name.upper() for a in assignments if a.app_name]
        if not app_names:
            return ""

        cache = {
            a.app_name: a
            for a in db.query(ComposioAppCache)
            .filter(ComposioAppCache.app_name.in_(app_names))
            .all()
        }

        parts: list[str] = [
            "\n## Connected Apps (Composio)\n",
            "You have access to these external apps via Composio. "
            "Use the `composio_execute` tool with an appropriate action.\n",
        ]

        for assignment in assignments:
            app_name = (assignment.app_name or "").upper()
            if not app_name:
                continue
            app = cache.get(app_name)
            parts.append(f"### {app_name}")
            if app and getattr(app, "description", None):
                parts.append(f"**Description**: {app.description}")

        content = "\n".join(parts)

        if self.max_tokens:
            content = self.truncate(content, self.max_tokens)
        return content
