"""
MemorySection — User memories and daily logs for system prompt injection.

Priority 6. Wraps SmartMemoryManager.retrieve_memories() and get_daily_logs()
so every code path gets consistent memory formatting.

For CHATBOT mode, also tries the Context Router (PRD-79) first for richer
context (session summary, temporal results, knowledge awareness) with
SmartMemoryManager as fallback.

Replaces memory injection in:
- smart_orchestrator.py (Context Router + retrieve_memories + daily logs)
- agent_factory.py (string concatenation)
- Missing injection in heartbeat/recipe paths

Supports kwargs:
- ``skip_memory`` (bool) — skip all memory retrieval (chatbot optimisation)
- ``chat_id`` (str) — conversation ID for Context Router session hydration
- ``query`` (str) — override search query for memory retrieval
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

    For chatbot mode, tries the Context Router (PRD-79) first for richer
    context, then falls back to SmartMemoryManager.

    All failures are caught and logged — memory retrieval must **never**
    crash the prompt build.
    """

    name: str = "memory"
    priority: int = 6
    max_tokens: Optional[int] = None

    def __init__(self) -> None:
        super().__init__()
        from config import config
        self.max_tokens = config.MEMORY_SECTION_MAX_TOKENS

    async def render(self, ctx: SectionContext) -> str:
        """Return formatted memory + daily-log block for the system prompt."""
        try:
            # Skip memory if the caller explicitly opted out
            if ctx.kwargs.get("skip_memory"):
                logger.debug("MemorySection: skip_memory=True — returning empty")
                return ""
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
        # --- Try Context Router first (PRD-79) for chatbot mode ---
        chat_id = ctx.kwargs.get("chat_id")
        context_bundle = await self._try_context_router(ctx, chat_id)

        if context_bundle is not None:
            content = self._format_context_bundle(context_bundle, ctx)
        else:
            # --- Fallback: SmartMemoryManager ---
            content = await self._build_from_smart_memory(ctx, chat_id)

        # Recipe memories (Mem0 learnings from previous runs) — step 1 only
        recipe_memories = ctx.kwargs.get("recipe_memories")
        if recipe_memories:
            summary = recipe_memories.get("summary", "")
            if summary and summary != "No relevant memories found":
                recipe_block = f"## Learnings from Previous Runs\n{summary}"
                content = f"{content}\n\n{recipe_block}" if content else recipe_block

        # Stash raw memory text for ContextResult.memory_context (SSE events)
        if content:
            ctx.kwargs["_memory_context"] = content

        if self.max_tokens and content:
            content = self.truncate(content, self.max_tokens)

        return content

    async def _try_context_router(
        self, ctx: SectionContext, chat_id: Optional[str]
    ) -> object | None:
        """Try the unified memory Context Router (PRD-79).

        Returns the context_bundle on success, None on failure or unavailability.
        """
        try:
            from modules.memory.unified_memory_service import get_unified_memory_service

            unified = get_unified_memory_service()
            query = ctx.kwargs.get("query") or self._extract_query(ctx)
            if not query:
                return None

            agent_id = getattr(ctx.agent, "id", None) if ctx.agent else None

            context_bundle = await unified.retrieve_context(
                workspace_id=ctx.workspace_id,
                agent_id=agent_id,
                query=query,
                conversation_id=chat_id,
            )

            logger.info(
                "[MemorySection] Context Router OK: ~%d tokens, signals=%s",
                getattr(context_bundle, "total_tokens_estimate", 0),
                getattr(context_bundle, "signals", []),
            )
            return context_bundle

        except Exception:
            logger.warning(
                "MemorySection: Context Router failed — falling back to SmartMemoryManager",
                exc_info=True,
            )
            return None

    def _format_context_bundle(self, bundle: object, ctx: SectionContext) -> str:
        """Format a Context Router bundle into system prompt sections."""
        parts: list[str] = []

        # Long-term memories
        lt_memories = getattr(bundle, "long_term_memories", [])
        if lt_memories:
            # PRD-206 S7: the Router path gets the same Q7 private-scope rule
            # and composite ranking as the SmartMemoryManager path — one
            # consent model, whichever retrieval lane served the turn.
            from modules.memory.injection_filter import visible_to_viewer
            from modules.memory.recall_ranking import rank_memories

            viewer = ctx.kwargs.get("viewer_subject_id")
            dict_memories = [m for m in lt_memories if isinstance(m, dict)]
            str_memories = [m for m in lt_memories if not isinstance(m, dict)]
            dict_memories = rank_memories(
                [m for m in dict_memories if visible_to_viewer(m, viewer)]
            )
            lt_memories = dict_memories + [{"memory": str(m)} for m in str_memories if m]

            memory_lines: list[str] = []
            raw_memories: list[dict] = []
            for m in lt_memories:
                text = m.get("memory", "") if isinstance(m, dict) else str(m)
                if text:
                    memory_lines.append(f"- {text}")
                    raw_memories.append(m if isinstance(m, dict) else {"memory": text})

            if memory_lines:
                parts.append("## What You Know About This User")
                parts.append("")
                parts.append("\n".join(memory_lines))

            # Stash raw memories for CTO override / SSE compat
            ctx.kwargs["_raw_memories"] = raw_memories

        # Session summary
        session_summary = getattr(bundle, "session_summary", None)
        if session_summary:
            parts.append("")
            parts.append("## Session Context")
            parts.append("")
            parts.append(str(session_summary))

        # Daily logs
        daily_logs = getattr(bundle, "daily_logs", None)
        if daily_logs:
            parts.append("")
            parts.append("## Recent Activity")
            parts.append("")
            parts.append(str(daily_logs))

        # Temporal results
        temporal = getattr(bundle, "temporal_results", None)
        if temporal:
            temporal_lines = []
            for m in temporal:
                text = ""
                if isinstance(m, dict):
                    text = m.get("memory") or m.get("content", "")
                elif isinstance(m, str):
                    text = m
                if text:
                    temporal_lines.append(f"- {text}")
            if temporal_lines:
                parts.append("")
                parts.append("## Relevant Past Context")
                parts.append("")
                parts.append("\n".join(temporal_lines))

        # Knowledge awareness
        awareness = getattr(bundle, "knowledge_awareness", None)
        if awareness:
            parts.append("")
            parts.append(str(awareness))

        # Extract user name if available
        self._extract_user_name_from_bundle(bundle, ctx)

        return "\n".join(parts).strip()

    async def _build_from_smart_memory(
        self, ctx: SectionContext, chat_id: Optional[str]
    ) -> str:
        """Fallback path: SmartMemoryManager + session hydration."""
        from consumers.chatbot.smart_memory import get_smart_memory_manager

        manager = get_smart_memory_manager()

        # --- Retrieve memories ---
        memory_text = await self._retrieve_memories(manager, ctx)

        # --- Retrieve daily logs ---
        daily_logs = await self._retrieve_daily_logs(manager, ctx)

        # --- Session hydration (Redis) — only in fallback path ---
        session_text = await self._hydrate_session(ctx, chat_id)

        # --- Assemble output ---
        parts: list[str] = []

        if memory_text:
            parts.append("## What You Know About This User")
            parts.append("")
            parts.append(memory_text)

        if session_text:
            parts.append("")
            parts.append(session_text)

        if daily_logs:
            parts.append("")
            parts.append("## Recent Activity")
            parts.append("")
            parts.append(daily_logs)

        return "\n".join(parts).strip()

    async def _retrieve_memories(
        self,
        manager: object,
        ctx: SectionContext,
    ) -> str:
        """Retrieve and format memories, returning empty string on failure."""
        try:
            # Build a query from the latest user message if available
            query = ctx.kwargs.get("query") or self._extract_query(ctx)
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
                # PRD-206 S7: Q7 private-scope guard + composite ranking
                # happen inside retrieve_memories; the viewer rides kwargs.
                viewer_subject_id=ctx.kwargs.get("viewer_subject_id"),
            )

            if not result or not result.formatted_context:
                return ""

            # Stash user name for ContextResult.user_name
            if result.user_context and result.user_context.name:
                ctx.kwargs["_user_name"] = result.user_context.name

            # Stash raw memories for CTO override / SSE compat
            if result.memories:
                ctx.kwargs["_raw_memories"] = list(result.memories)

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
            from config import config

            if not getattr(config, "INJECT_DAILY_LOGS", True):
                return ""

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

    async def _hydrate_session(
        self, ctx: SectionContext, chat_id: Optional[str]
    ) -> str:
        """Hydrate session context from Redis (L1) for the fallback path."""
        if not chat_id:
            return ""
        try:
            from modules.memory.unified_memory_service import get_unified_memory_service

            unified = get_unified_memory_service()
            session = await unified.get_session(
                workspace_id=ctx.workspace_id,
                conversation_id=chat_id,
            )
            if session and getattr(session, "summary", None):
                summary_text = str(session.summary)[:2000]
                exchange_count = getattr(session, "exchange_count", 0)
                return (
                    f"## Session Context\n\n"
                    f"Continuing conversation ({exchange_count} prior exchanges).\n"
                    f"Recent context:\n{summary_text}"
                )
        except Exception:
            logger.warning(
                "MemorySection: session hydration failed for chat_id=%s — skipping",
                chat_id,
                exc_info=True,
            )
        return ""

    @staticmethod
    def _extract_user_name_from_bundle(bundle: object, ctx: SectionContext) -> None:
        """Try to extract user name from Context Router bundle."""
        try:
            lt_memories = getattr(bundle, "long_term_memories", [])
            # Some memory entries contain user name metadata
            for m in lt_memories:
                if isinstance(m, dict) and m.get("user_name"):
                    ctx.kwargs["_user_name"] = m["user_name"]
                    return
        except Exception:
            pass

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
