"""
MemorySection — User memories and daily logs for system prompt injection.

Priority 6. Wraps SmartMemoryManager.retrieve_memories() and get_daily_logs()
so every code path gets consistent memory formatting.

Replaces memory injection in:
- smart_orchestrator.py (retrieve_memories + daily logs)
- agent_factory.py (string concatenation)
- Missing injection in heartbeat/recipe paths
"""

from __future__ import annotations

import logging
from typing import Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)


class MemorySection(BaseSection):
    """User memories and daily activity logs.

    Retrieves two-tier memories (global + agent-specific) via
    ``SmartMemoryManager.retrieve_memories()`` and appends daily logs
    via ``get_daily_logs()``.  The raw memory text is stashed in
    ``ctx.kwargs['_memory_context']`` so ``ContextService`` can expose it
    on ``ContextResult.memory_context`` for SSE events.

    All failures are caught and logged — memory retrieval must **never**
    crash the prompt build.
    """

    name: str = "memory"
    priority: int = 6
    max_tokens: Optional[int] = 1500

    async def render(self, ctx: SectionContext) -> str:
        """Return formatted memory + daily-log block for the system prompt."""
        try:
            return await self._build(ctx)
        except Exception:
            logger.exception(
                "MemorySection.render failed — skipping memory injection"
            )
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _build(self, ctx: SectionContext) -> str:
        from consumers.chatbot.smart_memory import get_smart_memory_manager

        manager = get_smart_memory_manager()

        # --- Retrieve memories ---------------------------------------------------
        memory_text = await self._retrieve_memories(manager, ctx)

        # --- Retrieve daily logs -------------------------------------------------
        daily_logs = await self._retrieve_daily_logs(manager, ctx)

        # --- Assemble output -----------------------------------------------------
        parts: list[str] = []

        if memory_text:
            parts.append("## What You Know About This User")
            parts.append("")
            parts.append(memory_text)

        if daily_logs:
            parts.append("")
            parts.append("## Recent Activity")
            parts.append("")
            parts.append(daily_logs)

        content = "\n".join(parts).strip()

        # Stash raw memory text for ContextResult.memory_context (SSE events)
        if content:
            ctx.kwargs["_memory_context"] = content

        if self.max_tokens and content:
            content = self.truncate(content, self.max_tokens)

        return content

    async def _retrieve_memories(
        self,
        manager: object,
        ctx: SectionContext,
    ) -> str:
        """Retrieve and format memories, returning empty string on failure."""
        try:
            # Build a query from the latest user message if available
            query = self._extract_query(ctx)
            if not query:
                return ""

            agent_id = getattr(ctx.agent, "id", None) if ctx.agent else None
            widget_mode = ctx.widget_mode

            result = await manager.retrieve_memories(  # type: ignore[attr-defined]
                workspace_id=ctx.workspace_id,
                agent_id=agent_id,
                query=query,
                limit=8,
                widget_mode=widget_mode,
            )

            if not result or not result.formatted_context:
                return ""

            # Stash user name for ContextResult.user_name
            if result.user_context and result.user_context.name:
                ctx.kwargs["_user_name"] = result.user_context.name

            return result.formatted_context

        except Exception:
            logger.warning(
                "MemorySection: memory retrieval failed — continuing without memories",
                exc_info=True,
            )
            return ""

    async def _retrieve_daily_logs(
        self,
        manager: object,
        ctx: SectionContext,
    ) -> str:
        """Retrieve daily logs, returning empty string on failure."""
        try:
            logs: str = await manager.get_daily_logs(  # type: ignore[attr-defined]
                workspace_id=ctx.workspace_id,
                max_chars=2000,
            )
            return logs or ""
        except Exception:
            logger.warning(
                "MemorySection: daily logs retrieval failed — continuing without logs",
                exc_info=True,
            )
            return ""

    @staticmethod
    def _extract_query(ctx: SectionContext) -> str:
        """Extract a search query from the latest user message."""
        if not ctx.messages:
            # Fall back to task description for non-chat modes
            return ctx.task_description or ""

        # Walk messages in reverse to find the latest user message
        for msg in reversed(ctx.messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content.strip()

        return ctx.task_description or ""
