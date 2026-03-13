"""
ContextService — Single entry point for building LLM context.

Orchestrates section composition, parallel rendering, token budget
allocation, tool loading, and message formatting into a ready-to-use
``ContextResult``.

Usage::

    service = ContextService(db_session)
    result = await service.build_context(
        mode=ContextMode.TASK_EXECUTION,
        agent=agent_record,
        workspace_id="ws_123",
        messages=conversation_messages,
        task_description="Search the web and write a report",
    )
    # result.system_prompt, result.tools, result.messages → ready for LLM
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from modules.context.budget import (
    DEFAULT_BUDGETS,
    RenderedSection,
    TokenBudget,
    TokenBudgetManager,
)
from modules.context.estimator import TokenEstimator
from modules.context.modes import MODE_CONFIGS, ContextMode, ModeConfig
from modules.context.result import ContextResult
from modules.context.sections import SECTION_REGISTRY, SectionContext
from modules.context.sections.conversation import ConversationSection
from modules.context.sections.tools import ToolLoadingStrategy, ToolsSection

logger = logging.getLogger(__name__)

_estimator = TokenEstimator()
_budget_manager = TokenBudgetManager()


class ContextService:
    """Single entry point for building LLM context.

    Every code path that calls an LLM should use this service instead of
    building prompts, loading tools, and injecting memory individually.
    """

    def __init__(self, db_session: Any = None) -> None:
        self._db_session = db_session

    async def build_context(
        self,
        mode: ContextMode,
        agent: Any,
        workspace_id: str,
        messages: Optional[list[dict]] = None,
        task_description: Optional[str] = None,
        recipe_step: Optional[dict] = None,
        complexity_assessment: Any = None,
        tool_hints: Optional[list[str]] = None,
        widget_mode: bool = False,
        **kwargs: Any,
    ) -> ContextResult:
        """Build a complete LLM context for the given *mode*.

        Returns an immutable ``ContextResult`` ready for the LLM call.
        Failures in individual sections are caught and logged — the build
        never crashes due to a single section error.
        """
        start = time.perf_counter()

        # --- 1. Look up mode configuration ---
        config = MODE_CONFIGS.get(mode)
        if config is None:
            logger.error(
                "[ContextService] Unknown mode %r — using CHATBOT defaults", mode
            )
            config = MODE_CONFIGS[ContextMode.CHATBOT]

        # --- 2. Build SectionContext ---
        # Pass personality flag so IdentitySection can use it
        extra_kwargs = dict(kwargs)
        extra_kwargs["personality"] = config.personality

        ctx = SectionContext(
            agent=agent,
            workspace_id=workspace_id,
            workspace_name=kwargs.get("workspace_name"),
            db_session=self._db_session,
            messages=messages,
            task_description=task_description,
            recipe_step=recipe_step,
            complexity_assessment=complexity_assessment,
            tool_hints=tool_hints,
            widget_mode=widget_mode,
            kwargs=extra_kwargs,
        )

        # --- 3. Instantiate sections from config ---
        sections = self._instantiate_sections(config)

        # --- 4. Render all sections in parallel ---
        rendered = await self._render_sections(sections, ctx)

        # --- 5. Apply token budget ---
        budget = self._get_budget(mode, config)
        included, trimmed_names = _budget_manager.allocate(rendered, budget)

        # --- 6. Assemble system prompt ---
        system_prompt = self._assemble_prompt(included)
        prompt_token_estimate = _estimator.estimate(system_prompt)

        # --- 7. Load tools ---
        tools, tool_choice = await self._load_tools(config, ctx)

        # --- 8. Format messages ---
        formatted_messages = self._format_messages(
            config, ctx, budget, prompt_token_estimate
        )

        # --- 9. Compute final metadata ---
        sections_included = [s.name for s in included if s.content]
        total_token_estimate = (
            prompt_token_estimate
            + sum(_estimator.estimate(m.get("content", "")) for m in formatted_messages)
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        agent_id = getattr(agent, "id", "?") if agent else "?"
        logger.info(
            "[ContextService] mode=%s agent=%s sections=%s trimmed=%s "
            "token_estimate=%d/%d tools=%d prep_time=%.1fms",
            mode.value if isinstance(mode, ContextMode) else mode,
            agent_id,
            sections_included,
            trimmed_names,
            total_token_estimate,
            budget.available_for_sections,
            len(tools),
            elapsed_ms,
        )

        return ContextResult(
            system_prompt=system_prompt,
            messages=formatted_messages,
            tools=tools,
            tool_choice=tool_choice,
            mode=mode.value if isinstance(mode, ContextMode) else str(mode),
            sections_included=sections_included,
            sections_trimmed=trimmed_names,
            token_estimate=total_token_estimate,
            token_budget=budget.available_for_sections,
            memory_context=ctx.kwargs.get("_memory_context"),
            user_name=ctx.kwargs.get("_user_name"),
            preparation_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _instantiate_sections(config: ModeConfig) -> list[Any]:
        """Create section instances from the mode's section name list."""
        sections = []
        for name in config.sections:
            cls = SECTION_REGISTRY.get(name)
            if cls is None:
                logger.warning(
                    "[ContextService] Section '%s' not found in SECTION_REGISTRY — skipping",
                    name,
                )
                continue
            sections.append(cls())
        return sections

    @staticmethod
    async def _render_sections(
        sections: list[Any],
        ctx: SectionContext,
    ) -> list[RenderedSection]:
        """Render all sections in parallel, catching per-section failures."""

        async def _safe_render(section: Any) -> RenderedSection:
            try:
                content = await section.render(ctx)
            except Exception:
                logger.exception(
                    "[ContextService] Section '%s' render() raised — skipping",
                    section.name,
                )
                content = ""

            return RenderedSection(
                name=section.name,
                priority=section.priority,
                content=content or "",
                token_estimate=_estimator.estimate(content or ""),
                max_tokens=section.max_tokens,
            )

        results = await asyncio.gather(
            *(_safe_render(s) for s in sections),
            return_exceptions=False,
        )
        return list(results)

    @staticmethod
    def _get_budget(mode: ContextMode, config: ModeConfig) -> TokenBudget:
        """Resolve the token budget for this mode."""
        budget = DEFAULT_BUDGETS.get(mode)
        if budget is None:
            # Fallback: generous budget
            budget = TokenBudget(
                total=128_000,
                reserved_for_response=4_096,
                reserved_for_messages=0,
            )

        # If mode has a max_tokens override, use a tighter budget
        if config.max_tokens is not None:
            budget = TokenBudget(
                total=config.max_tokens,
                reserved_for_response=budget.reserved_for_response,
                reserved_for_messages=budget.reserved_for_messages,
            )

        return budget

    @staticmethod
    def _assemble_prompt(sections: list[RenderedSection]) -> str:
        """Concatenate rendered sections into the system prompt.

        Sections with empty content are skipped. Non-empty sections
        are joined by double newlines.
        """
        blocks: list[str] = []
        for section in sections:
            if not section.content:
                continue
            blocks.append(section.content)
        return "\n\n".join(blocks)

    @staticmethod
    async def _load_tools(
        config: ModeConfig,
        ctx: SectionContext,
    ) -> tuple[list[dict[str, Any]], str]:
        """Load tools using the strategy declared by the mode config."""
        strategy_str = config.tool_loading
        try:
            strategy = ToolLoadingStrategy(strategy_str)
        except ValueError:
            logger.warning(
                "[ContextService] Unknown tool_loading strategy '%s' — using NONE",
                strategy_str,
            )
            return [], "none"

        if strategy == ToolLoadingStrategy.NONE:
            return [], "none"

        tools_section = ToolsSection()
        agent_id = getattr(ctx.agent, "id", None) if ctx.agent else None

        return await tools_section.load_tools(
            agent_id=agent_id,
            workspace_id=ctx.workspace_id,
            strategy=strategy,
            db_session=ctx.db_session,
            intent_result=ctx.kwargs.get("intent_result"),
            tool_hints=ctx.tool_hints,
            query=ctx.kwargs.get("query"),
            conversation_context=ctx.messages,
        )

    @staticmethod
    def _format_messages(
        config: ModeConfig,
        ctx: SectionContext,
        budget: TokenBudget,
        prompt_tokens: int,
    ) -> list[dict[str, str]]:
        """Format conversation messages with budget-aware trimming."""
        if not ctx.messages:
            return []

        # Only format messages if conversation is in the mode's sections
        if "conversation" not in config.sections:
            return []

        # Budget for messages = reserved_for_messages (or remaining budget)
        message_budget = budget.reserved_for_messages
        if message_budget <= 0:
            # No explicit reservation — use whatever is left
            message_budget = max(
                0, budget.available_for_sections - prompt_tokens
            )

        conversation = ConversationSection()
        return conversation.format_messages(
            messages=ctx.messages,
            budget_tokens=message_budget if message_budget > 0 else None,
        )
