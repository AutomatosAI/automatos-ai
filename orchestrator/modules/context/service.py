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
from core.context_guard import count_tokens, get_context_window
from modules.context.modes import MODE_CONFIGS, ContextMode, ModeConfig
from modules.context.planning import (
    PACK_HEADER,
    PlanningContextPack,
    apply_pack_budget,
    trim_to_tokens,
)
from modules.context.result import ContextResult
from modules.context.sections import SECTION_REGISTRY, SectionContext
from core.observability.tracer import fire_assembly_trace
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

_budget_manager = TokenBudgetManager()

# PRD-201 S4: sections whose rendered content changes per-turn or per-query.
# They render AFTER the cache-stable prefix so a change in them never
# invalidates the cached static blocks (identity/skills/catalog). ``datetime``
# changes every turn (the canonical volatile block); ``memory``/``business_graph``
# are query-dependent; ``conversation`` renders no system text at all. Every
# other section is treated as cache-stable.
VOLATILE_SECTIONS = frozenset(
    {
        "datetime_context",
        "memory",
        "business_graph",
        "field_memory",
        "planning_history",
        "conversation",
    }
)


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

        # Resolve the driving model ONCE — used for the model-aware budget
        # (PRD-201 S3), the persisted assembly trace (PRD-201 S1), and the
        # attachment vision check (PRD-127). The agent's llm_config carries the
        # model; an explicit model_id kwarg (chat) wins.
        resolved_model_id = model_id
        if not resolved_model_id and agent is not None:
            _llm_config = getattr(agent, "llm_config", {}) or {}
            resolved_model_id = _llm_config.get("model")

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

        # --- 5. Apply token budget (PRD-201 S3: sized to the model window) ---
        budget = self._get_budget(mode, config, resolved_model_id, self._db_session)
        included, trimmed_names = _budget_manager.allocate(rendered, budget)

        # --- 6. Assemble system prompt (PRD-201 S4: cache-stable ordering) ---
        system_prompt, cacheable_prefix = self._assemble_prompt(included)
        prompt_token_estimate = count_tokens(system_prompt)

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
            + sum(count_tokens(m.get("content", "")) for m in formatted_messages)
        )

        # --- 10. PRD-127: Inject attachment parts into messages ---
        attachment_failures: list[dict] = []
        if attachment_ids:
            try:
                resolver = AttachmentResolver(db_session=self._db_session)
                attachment_parts, attachment_failures = await resolver.resolve(
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
                if attachment_failures:
                    logger.warning(
                        "[ContextService] %d attachment(s) unavailable: %s",
                        len(attachment_failures),
                        [f.get("filename") or f.get("attachment_id") for f in attachment_failures],
                    )
            except VisionNotSupportedError:
                # Re-raise vision errors — caller should handle
                raise
            except Exception as e:
                logger.error(
                    "[ContextService] Attachment resolution failed: %s", e, exc_info=True
                )
                # PRD-223 S0.3: continue without the attachments, but NEVER
                # silently — the model must be told the content did not
                # arrive, or it will infer/fabricate from the filename.
                from modules.attachments.resolver import build_unavailable_marker
                attachment_failures = [
                    {"attachment_id": str(aid), "filename": None, "reason": "resolution error"}
                    for aid in attachment_ids
                ]
                formatted_messages = inject_parts_into_last_user_message(
                    formatted_messages, [build_unavailable_marker(attachment_failures)]
                )

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

        # PRD-201 S1: the per-section trace the assembler used to discard —
        # every rendered section with its priority, honest token count, whether
        # it rendered anything, and whether the budgeter trimmed it.
        _trimmed_set = set(trimmed_names)
        section_trace = [
            {
                "name": s.name,
                "priority": s.priority,
                "token_estimate": s.token_estimate,
                "rendered_nonempty": bool(s.content),
                "trimmed": s.name in _trimmed_set,
            }
            for s in rendered
        ]

        result = ContextResult(
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
            model=resolved_model_id,
            budget_total=budget.total,
            sections=section_trace,
            injected_memory_ids=list(ctx.kwargs.get("_injected_memory_ids") or []),
            cacheable_prefix=cacheable_prefix,
            attachment_failures=attachment_failures,
        )

        # PRD-201 S1: emit the assembly trace onto the observability seam. The
        # durable per-turn row (messages.context_trace) is written by the turn
        # writer from result.to_assembly_trace() regardless of TRACING_ENABLED;
        # this fire only mirrors it to Langfuse when tracing is ON. Guarded —
        # never fails a build.
        fire_assembly_trace(
            trace=result.to_assembly_trace(),
            workspace_id=workspace_id,
            metadata={"agent_id": agent_id},
        )

        return result

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

            # PRD-201 S1: same assembly-trace shape as build_context, so an
            # approver can later judge a plan against the pack the planner saw
            # (PRD-163 card). Guarded inside fire_assembly_trace — never fails.
            _trimmed_planning = set(trimmed)
            fire_assembly_trace(
                trace={
                    "mode": ContextMode.PLANNING.value,
                    "model": None,
                    "budget_total": budget_tokens,
                    "token_estimate": token_estimate,
                    "token_budget": budget_tokens,
                    "prep_ms": round(elapsed_ms, 1),
                    "sections": [
                        {
                            "name": s.name,
                            "priority": s.priority,
                            "token_estimate": s.token_estimate,
                            "rendered_nonempty": bool(s.content),
                            "trimmed": s.name in _trimmed_planning,
                        }
                        for s in rendered
                    ],
                    "sections_included": list(section_map.keys()),
                    "sections_trimmed": trimmed,
                    "injected_memory_ids": [],
                },
                workspace_id=workspace_id,
                metadata={"planning_pack": True},
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
                token_estimate=count_tokens(content or ""),
                max_tokens=section.max_tokens,
            )

        results = await asyncio.gather(
            *(_safe_render(s) for s in sections),
            return_exceptions=False,
        )
        return list(results)

    @staticmethod
    def _get_budget(
        mode: ContextMode,
        config: ModeConfig,
        model: Optional[str] = None,
        db_session: Any = None,
    ) -> TokenBudget:
        """Resolve the token budget for this mode, sized to the model window.

        PRD-201 S3 — ``total`` comes from the actual model context window via
        ``core.context_guard.get_context_window`` instead of a hardcoded 128k, so
        a small-window model stops silently emitting an over-budget prompt (that
        later gets emergency-compacted) and a 200k/1M-window model stops leaving
        capacity unused. A mode-level ``max_tokens`` override still wins — it is
        a deliberate per-mode cap (the pack rides inside another prompt, the
        heartbeat tick is kept lean), not the model ceiling.

        The per-mode reservations (``reserved_for_response`` /
        ``reserved_for_messages``) were sized for a 128k window; when the
        resolved window is smaller they are scaled down proportionally so
        ``available_for_sections`` never goes negative on a small model. The
        priority-≤2-never-dropped invariant lives in ``TokenBudgetManager`` and
        is untouched.
        """
        base = DEFAULT_BUDGETS.get(mode) or TokenBudget(
            total=128_000,
            reserved_for_response=4_096,
            reserved_for_messages=0,
        )

        reserved_response = base.reserved_for_response
        reserved_messages = base.reserved_for_messages

        if config.max_tokens is not None:
            # A deliberate per-mode cap keeps its designed reservations (e.g. the
            # heartbeat tick's 8000/2048) — no scaling.
            total = config.max_tokens
        else:
            total = get_context_window(model or "", db_session) or base.total
            if 0 < total < base.total:
                # Small MODEL window — shrink the 128k-sized absolute reservations
                # so available_for_sections never goes negative.
                scale = total / base.total
                reserved_response = max(512, int(reserved_response * scale))
                reserved_messages = int(reserved_messages * scale)

        return TokenBudget(
            total=total,
            reserved_for_response=reserved_response,
            reserved_for_messages=reserved_messages,
        )

    @staticmethod
    def _assemble_prompt(sections: list[RenderedSection]) -> tuple[str, Optional[str]]:
        """Concatenate rendered sections into the system prompt, cache-stable.

        PRD-201 S4 — the reordering is the design work: the static, high-value
        blocks (identity, skills, the action catalog, the run's task context)
        render FIRST, and the volatile blocks (memory excerpts, KG snippet,
        datetime) render LAST. Section *content* is unchanged; only the order
        moves, so a change in a volatile block never alters the leading bytes
        that an Anthropic ``cache_control`` breakpoint would cache.

        Returns ``(system_prompt, cacheable_prefix)`` where ``cacheable_prefix``
        is the join of the stable blocks (``None`` when there are none). Every
        consumer reads ``system_prompt`` exactly as before; only the Anthropic
        client seam reads ``cacheable_prefix`` to place the breakpoint.
        """
        stable_blocks: list[str] = []
        volatile_blocks: list[str] = []
        for section in sections:
            if not section.content:
                continue
            if section.name in VOLATILE_SECTIONS:
                volatile_blocks.append(section.content)
            else:
                stable_blocks.append(section.content)

        cacheable_prefix = "\n\n".join(stable_blocks) if stable_blocks else None
        system_prompt = "\n\n".join(stable_blocks + volatile_blocks)
        return system_prompt, cacheable_prefix

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
            # Surface already assembled by the entrypoint (su + PRD-221
            # page-prior threaded in) — ToolsSection uses it instead of
            # rebuilding blind.
            prebuilt_tools=ctx.kwargs.get("prebuilt_tools"),
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
