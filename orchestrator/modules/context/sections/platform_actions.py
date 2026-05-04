"""
PlatformActionsSection — Markdown catalog of available platform_execute actions.

Priority 5. Wraps ActionRegistry.build_prompt_summary() so every code path
gets the same action catalog injected into the system prompt.

When ``config.SEMANTIC_TOOL_ROUTING`` is enabled and a ``query`` is supplied
through ``ctx.kwargs``, the section first asks ``ActionSemanticIndex`` to
rank actions by similarity and renders only the top-K via
``ActionRegistry.build_filtered_prompt_summary()`` (PRD-138 US-004). Any
failure in that path falls back to the full catalog so prompt assembly
never breaks because of the index.

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
        """Return the platform-action catalog for the system prompt.

        Decision tree:

        - SEMANTIC_TOOL_ROUTING flag off → full ``_build()`` dump
        - flag on but no/empty query → full ``_build()`` dump
        - flag on + query present → try ``_build_filtered(query)`` and
          fall back to full ``_build()`` if it returns ``None``
        """
        try:
            query = ""
            if ctx.kwargs:
                raw = ctx.kwargs.get("query", "")
                if isinstance(raw, str):
                    query = raw.strip()

            if query and self._semantic_routing_enabled():
                filtered = await self._build_filtered(query)
                if filtered:
                    return filtered

            return self._build()
        except Exception:
            logger.exception(
                "PlatformActionsSection.render failed — skipping action catalog"
            )
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _PREAMBLE = (
        "## Platform Actions\n\n"
        "You can execute these actions via `platform_execute`. Always specify the "
        "action name and include all required parameters. If an action fails, check "
        "the error and retry with corrected parameters — do not guess or fabricate "
        "results.\n\n"
    )

    def _semantic_routing_enabled(self) -> bool:
        """Read the feature flag from the canonical config singleton."""
        from config import config

        return bool(getattr(config, "SEMANTIC_TOOL_ROUTING", False))

    def _top_k(self) -> int:
        """Return the configured top-K with a safe default."""
        from config import config

        try:
            return int(getattr(config, "SEMANTIC_TOOL_ROUTING_TOP_K", 15))
        except (TypeError, ValueError):
            return 15

    def _build(self) -> str:
        from modules.tools.discovery.action_registry import get_action_registry

        registry = get_action_registry()
        catalog: str = registry.build_prompt_summary(exclude_promoted=True, exclude_admin=True)

        if not catalog:
            logger.warning(
                "PlatformActionsSection: ActionRegistry returned empty summary"
            )
            return ""

        content = self._PREAMBLE + catalog

        if self.max_tokens:
            content = self.truncate(content, self.max_tokens)

        return content

    async def _build_filtered(self, query: str) -> Optional[str]:
        """Render only the top-K actions ranked against ``query``.

        Returns the filtered markdown string on success, or ``None`` if any
        step fails so ``render()`` can fall back to the full catalog. Never
        raises.
        """
        try:
            # Lazy imports avoid a circular dep with the action registrar.
            from modules.tools.discovery.action_registry import get_action_registry
            from modules.tools.discovery.action_semantic_index import (
                get_action_semantic_index,
            )

            index = get_action_semantic_index()
            top_k = self._top_k()
            ranked = await index.rank_actions(
                query,
                top_k=top_k,
                exclude_admin=True,
                exclude_promoted=True,
            )
            if not ranked:
                logger.debug(
                    "PlatformActionsSection: semantic index returned no matches "
                    "for query — falling back to full catalog"
                )
                return None

            top_names = [name for name, _score in ranked]
            registry = get_action_registry()
            catalog = registry.build_filtered_prompt_summary(
                top_names,
                exclude_admin=True,
                exclude_promoted=True,
            )
            if not catalog:
                return None

            content = self._PREAMBLE + catalog
            if self.max_tokens:
                content = self.truncate(content, self.max_tokens)
            return content
        except Exception:
            logger.warning(
                "PlatformActionsSection._build_filtered failed — falling back to "
                "full catalog",
                exc_info=True,
            )
            return None
