"""
ToolsSection — Unified tool loading with four strategies.

Priority 3. Does NOT contribute text to the system prompt (render() returns
empty string). Instead, provides ``load_tools()`` which the ContextService
calls to populate ``ContextResult.tools`` and ``ContextResult.tool_choice``.

Strategies:
    FULL           — All assigned tools (core + dispatcher + composio).
    FILTERED       — Intent-based filtering via SmartToolRouter.
    DISPATCHER_ONLY — Only the platform_execute dispatcher schema.
    NONE           — Empty tool list, tool_choice="none".

Replaces:
- get_tools_for_agent() in tool_router.py
- smart_tool_router.route() in chatbot path
- inline to_dispatcher_schema() calls in heartbeat
- Tool assembly in agent_factory.py
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)


class ToolLoadingStrategy(str, Enum):
    """How tools are assembled for a given context mode."""

    FULL = "full"
    FILTERED = "filtered"
    DISPATCHER_ONLY = "dispatcher_only"
    NONE = "none"


class ToolsSection(BaseSection):
    """Unified tool loading across all context modes.

    Tools are NOT part of the system prompt text — they go into
    ``ContextResult.tools``.  The ``render()`` method returns an empty
    string; callers should invoke ``load_tools()`` separately.
    """

    name: str = "tools"
    priority: int = 3
    max_tokens: Optional[int] = None  # Not rendered into system prompt

    async def render(self, ctx: SectionContext) -> str:
        """Tools don't contribute to the system prompt text."""
        return ""

    # ------------------------------------------------------------------
    # Public API — called by ContextService, not by render()
    # ------------------------------------------------------------------

    async def load_tools(
        self,
        agent_id: Optional[int],
        workspace_id: str,
        strategy: ToolLoadingStrategy,
        db_session: Any = None,
        intent_result: Any = None,
        tool_hints: Optional[list[str]] = None,
        query: Optional[str] = None,
        conversation_context: Optional[list[dict]] = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """Load tool schemas and determine tool_choice.

        Returns:
            (tool_schemas, tool_choice) — ready for ContextResult.
        """
        try:
            if strategy == ToolLoadingStrategy.NONE:
                return [], "none"

            if strategy == ToolLoadingStrategy.DISPATCHER_ONLY:
                return self._load_dispatcher_only()

            if strategy == ToolLoadingStrategy.FULL:
                return self._load_full(agent_id, workspace_id, db_session)

            if strategy == ToolLoadingStrategy.FILTERED:
                return await self._load_filtered(
                    agent_id,
                    workspace_id,
                    db_session,
                    intent_result=intent_result,
                    tool_hints=tool_hints,
                    query=query,
                    conversation_context=conversation_context,
                )

            logger.warning("Unknown ToolLoadingStrategy %r — returning empty tools", strategy)
            return [], "none"

        except Exception:
            logger.exception(
                "ToolsSection.load_tools failed (strategy=%s) — returning empty tools",
                strategy,
            )
            return [], "auto"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dispatcher_only(self) -> tuple[list[dict[str, Any]], str]:
        """Return only the platform_execute dispatcher schema."""
        from modules.tools.discovery.action_registry import get_action_registry

        registry = get_action_registry()
        schema = registry.to_dispatcher_schema(exclude_admin=True)
        return [schema], "auto"

    def _load_full(
        self,
        agent_id: Optional[int],
        workspace_id: str,
        db_session: Any = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """Return all assigned tools (core + platform dispatcher + composio)."""
        from modules.tools.tool_router import get_tools_for_agent

        tools = get_tools_for_agent(
            agent_id=agent_id,
            db_session=db_session,
            workspace_id=workspace_id,
        )
        return tools, "auto"

    async def _load_filtered(
        self,
        agent_id: Optional[int],
        workspace_id: str,
        db_session: Any = None,
        intent_result: Any = None,
        tool_hints: Optional[list[str]] = None,
        query: Optional[str] = None,
        conversation_context: Optional[list[dict]] = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """Return intent-filtered subset of tools via SmartToolRouter."""
        from modules.tools.tool_router import get_tools_for_agent

        # Step 1: Load all available tools
        all_tools = get_tools_for_agent(
            agent_id=agent_id,
            db_session=db_session,
            workspace_id=workspace_id,
        )

        if not all_tools:
            return [], "none"

        # Step 2: Apply smart filtering if we have context for it
        if query or conversation_context or tool_hints:
            try:
                from consumers.chatbot.smart_tool_router import get_smart_tool_router

                router = get_smart_tool_router()
                result = await router.route(
                    query=query or "",
                    available_tools=all_tools,
                    conversation_context=conversation_context,
                    tool_hints=tool_hints,
                )

                if not result.should_include_tools:
                    # Even when SmartToolRouter says no tools, always
                    # include platform_* tools so the agent can answer
                    # self-awareness queries (PRD-64).
                    platform_tools = [
                        t for t in all_tools
                        if t.get("function", {}).get("name", "").startswith("platform_")
                    ]
                    if platform_tools:
                        return platform_tools, "auto"
                    return [], "none"

                return result.filtered_tools, result.tool_choice

            except Exception:
                logger.warning(
                    "SmartToolRouter.route() failed — falling back to full tool set",
                    exc_info=True,
                )

        # Fallback: return all tools unfiltered
        return all_tools, "auto"
