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
from modules.context.planning import (
    PACK_HEADER,
    PlanningContextPack,
    apply_pack_budget,
    trim_to_tokens,
)
from modules.context.result import ContextResult
from modules.context.sections import SECTION_REGISTRY, SectionContext
from modules.context.sections.conversation import ConversationSection
from modules.context.sections.tools import ToolLoadingStrategy, ToolsSection

# PRD-127: Ephemeral attachments
from modules.attachments.resolver import (
    AttachmentResolver,
    VisionNotSupportedError,
    inject_parts_into_last_user_message,
)
from uuid import UUID

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
        attachment_ids: Optional[list[str]] = None,  # PRD-127
        model_id: Optional[str] = None,  # PRD-127: for vision capability check
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

        # --- 10. PRD-127: Inject attachment parts into messages ---
        if attachment_ids:
            try:
                resolved_model_id = model_id
                if not resolved_model_id and agent:
                    llm_config = getattr(agent, "llm_config", {}) or {}
                    resolved_model_id = llm_config.get("model")

                resolver = AttachmentResolver(db_session=self._db_session)
                attachment_parts = await resolver.resolve(
                    attachment_ids=[UUID(aid) for aid in attachment_ids],
                    workspace_id=UUID(workspace_id),
                    model_id=resolved_model_id or "",
                )
                if attachment_parts:
                    formatted_messages = inject_parts_into_last_user_message(
                        formatted_messages, attachment_parts
                    )
                    logger.info(
                        "[ContextService] Injected %d attachment parts into messages",
                        len(attachment_parts),
                    )
            except VisionNotSupportedError:
                # Re-raise vision errors — caller should handle
                raise
            except Exception as e:
                logger.error(
                    "[ContextService] Attachment resolution failed: %s", e, exc_info=True
                )
                # Continue without attachments rather than failing the entire request

        elapsed_ms = (time.perf_counter() - start) * 1000

        agent_id = getattr(agent, "id", "?") if agent else "?"
        logger.info(
            "[ContextService] mode=%s agent=%s sections=%s trimmed=%s "
            "token_estimate=%d/%d tools=%d attachments=%d prep_time=%.1fms",
            mode.value if isinstance(mode, ContextMode) else mode,
            agent_id,
            sections_included,
            trimmed_names,
            total_token_estimate,
            budget.available_for_sections,
            len(tools),
            len(attachment_ids) if attachment_ids else 0,
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
    # Planning Context Pack (PRD-164 S1, Q61)
    # ------------------------------------------------------------------

    async def build_planning_context(
        self,
        *,
        goal: str,
        workspace_id: str,
        agents: Optional[list] = None,
        include_roster: bool = True,
        max_tokens: Optional[int] = None,
        team: Optional[str] = None,
    ) -> PlanningContextPack:
        """Assemble the ONE token-budgeted planning context pack (Q61).

        Every planner on the platform — MissionPlanner, board ``plan_task``,
        AutoBrain — consumes THIS pack; none assembles planning context on its
        own. Sections come from ``MODE_CONFIGS[ContextMode.PLANNING]``:
        RAG-on-goal (PRD-157 choke point), mission summaries + task failures
        (PRD-159 recall), KG subgraph (PRD-165 graph service), and roster +
        agent performance. The cap reuses the PRD-157 token budgeter.

        ``include_roster=False`` is for callers whose prompt already presents
        a roster surface (MissionPlanner) — same assembler, one section fewer.
        Never raises: failures degrade to an empty pack.
        """
        start = time.perf_counter()
        budget_tokens = max_tokens or DEFAULT_BUDGETS[
            ContextMode.PLANNING
        ].available_for_sections

        try:
            config = MODE_CONFIGS[ContextMode.PLANNING]

            kwargs: dict[str, Any] = {"team": team}
            if include_roster:
                roster = agents if agents is not None else self._fetch_roster(workspace_id)
                if roster is not None:
                    kwargs["roster_agents"] = roster
                    kwargs["agent_performance"] = self._fetch_agent_performance(roster)

            ctx = SectionContext(
                agent=None,
                workspace_id=str(workspace_id),
                db_session=self._db_session,
                messages=None,
                task_description=goal,
                kwargs=kwargs,
            )

            sections = self._instantiate_sections(config)
            rendered = await self._render_sections(sections, ctx)

            # PRD-157 budgeter caps the pack (whole-section selection + hard
            # cap), leaving headroom for the standing header + joins. Counting
            # goes through core.context_guard.count_tokens — the same single
            # definition of "how big is this" the budgeter uses.
            from core.context_guard import count_tokens

            header_tokens = count_tokens(PACK_HEADER)
            section_budget = max(
                0, budget_tokens - header_tokens - (len(rendered) + 1)
            )
            included, trimmed = apply_pack_budget(rendered, section_budget)

            section_map = {s.name: s.content for s in included if s.content}
            if not section_map:
                return PlanningContextPack(token_budget=budget_tokens)

            content = "\n\n".join([PACK_HEADER, *section_map.values()])
            # Belt-and-braces: the budget is a hard guarantee (AC3) — joins
            # and tokenizer boundary wobble must never push the pack over.
            if count_tokens(content) > budget_tokens:
                content = trim_to_tokens(content, budget_tokens)
            token_estimate = count_tokens(content)

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "[ContextService] planning pack ws=%s sections=%s trimmed=%s "
                "tokens=%d/%d prep_time=%.1fms",
                workspace_id,
                list(section_map.keys()),
                trimmed,
                token_estimate,
                budget_tokens,
                elapsed_ms,
            )

            return PlanningContextPack(
                content=content,
                sections=section_map,
                token_estimate=token_estimate,
                token_budget=budget_tokens,
                sections_included=list(section_map.keys()),
                sections_trimmed=trimmed,
            )
        except Exception:
            logger.exception(
                "[ContextService] build_planning_context failed — returning empty pack"
            )
            return PlanningContextPack(token_budget=budget_tokens)

    def _fetch_roster(self, workspace_id: str) -> Optional[list]:
        """Active workspace agents for the pack's roster section.

        Returns None (roster unknown) when no DB session is available, so the
        roster section renders nothing rather than claiming "no agents".
        """
        if self._db_session is None:
            return None
        try:
            from core.models.core import Agent

            return (
                self._db_session.query(Agent)
                .filter(
                    Agent.workspace_id == workspace_id,
                    Agent.status == "active",
                )
                .all()
            )
        except Exception:
            logger.warning(
                "[ContextService] roster fetch failed for planning pack",
                exc_info=True,
            )
            return None

    def _fetch_agent_performance(self, roster: list) -> dict:
        """Recent verification performance per agent (agent_matcher history map).

        Reuses ``_build_history_map`` — the platform's one performance scorer —
        rather than a parallel computation. Empty on any failure.
        """
        if self._db_session is None or not roster:
            return {}
        try:
            from modules.coordination.agent_matcher import _build_history_map

            agent_ids = [
                getattr(a, "id", None) for a in roster
                if getattr(a, "id", None) is not None
            ]
            if not agent_ids:
                return {}
            return _build_history_map(self._db_session, agent_ids)
        except Exception:
            logger.warning(
                "[ContextService] agent performance fetch failed for planning pack",
                exc_info=True,
            )
            return {}

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
