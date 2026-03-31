"""
PlatformActionsSection — Markdown catalog of available platform_execute actions.

Priority 5. Wraps ActionRegistry.build_prompt_summary() so every code path
gets the same action catalog injected into the system prompt.

Replaces inline injection in:
- smart_orchestrator.py
- agent_factory.py
- heartbeat_service.py
"""

from __future__ import annotations

import logging
from typing import Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)


class PlatformActionsSection(BaseSection):
    """Markdown catalog of available platform_execute actions.

    Delegates to ``ActionRegistry.build_prompt_summary()`` which returns
    a formatted markdown string grouped by category.  If the registry is
    unavailable or raises, the section degrades to an empty string so the
    rest of the prompt is unaffected.
    """

    name: str = "platform_actions"
    priority: int = 5
    max_tokens: Optional[int] = 2000

    async def render(self, ctx: SectionContext) -> str:
        """Return the platform-action catalog for the system prompt."""
        try:
            return self._build()
        except Exception:
            logger.exception(
                "PlatformActionsSection.render failed — skipping action catalog"
            )
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self) -> str:
        from modules.tools.discovery.action_registry import get_action_registry

        registry = get_action_registry()
        content: str = registry.build_prompt_summary(exclude_promoted=True, exclude_admin=True)

        if not content:
            logger.warning(
                "PlatformActionsSection: ActionRegistry returned empty summary"
            )
            return ""

        if self.max_tokens:
            content = self.truncate(content, self.max_tokens)

        return content
