"""
Smart Tool Router
=================

Intelligent tool routing that decides:
- WHETHER to use tools at all
- WHICH tools to make available
- HOW to prioritize tool selection

Works with the Intent Classifier to route appropriately.
Delegates semantic ranking to GraphRouter (PRD-141 US-014) — the single
tool-selection pipeline — and falls back to keyword/category matching when the
graph is unavailable.

NOTE: This module is consumed by ContextService (ToolLoadingStrategy.FILTERED).
It is the only caller. Future work: absorb filtering logic into
modules/context/sections/tools.py and delete this file.
See PRD-81 Task 5.3 and system audit R1 finding.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass

from .intent_classifier import Intent, IntentResult, get_intent_classifier

logger = logging.getLogger(__name__)


# Intent → ActionRegistry *category* names (PRD-141 US-015).
# Replaces the old hardcoded per-tool category dicts: instead of pinning literal
# tool names, each tool-requiring intent maps to one or more ActionRegistry
# categories, and the matching action names are pulled from the registry at call
# time. An action registered under an already-mapped category is therefore
# auto-discoverable with no edit to this router.
# Values are REAL registry category names (verified against modules/tools/).
# GREETING / CHITCHAT / FACTUAL are intentionally absent — they classify as
# requires_tools=False and never reach the filter.
_INTENT_TO_REGISTRY_CATEGORIES: Dict[Intent, List[str]] = {
    Intent.DATA_QUERY: ["analytics", "database", "graph", "field", "reports"],
    Intent.SEARCH: ["discovery", "documents", "graph", "memory", "workspace_files"],
    Intent.EXTERNAL_ACTION: ["integrations", "notifications", "marketplace", "skills"],
    Intent.CREATION: ["documents", "reports", "blog", "workspace_files", "playbooks"],
    Intent.MEMORY_RECALL: ["memory", "field"],
    Intent.MULTI_STEP: [
        "agents", "missions", "tasks", "playbooks", "workspace", "workspace_files",
        "analytics", "reports", "documents", "graph", "memory", "field",
        "marketplace", "skills", "monitoring", "infrastructure", "integrations",
        "discovery", "scheduling", "governance", "notifications", "blog",
    ],
}


@dataclass
class ToolRoutingResult:
    """Result of tool routing decision."""
    should_include_tools: bool
    filtered_tools: List[Dict[str, Any]]
    priority_tools: List[str]  # Tools to emphasize
    tool_choice: str  # "auto", "required", or "none"
    reasoning: str


class SmartToolRouter:
    """
    Intelligent tool routing for the Automatos assistant.

    Philosophy:
    - Don't overwhelm the LLM with tools it doesn't need
    - Match tools to intent, not just keywords
    - Prefer internal tools for internal data
    - Use Composio for external app actions

    PRD-141 US-014: when SEMANTIC_TOOL_ROUTING=true, ranking is delegated to
    GraphRouter (the single tool-selection pipeline). Falls back to keyword-based
    category matching if the graph is unavailable.
    """

    # Core tools that are almost always useful
    # NOTE: smart_query_database / query_database intentionally excluded —
    # NL2SQL has no workspace scoping and leaks cross-tenant data.
    # Platform tools (platform_list_agents etc.) are the correct path
    # for querying workspace data.  NL2SQL stays in the "data" category
    # so it's still reachable when the intent explicitly asks for SQL.
    CORE_TOOLS = frozenset({
        "search_knowledge",
        "semantic_search",
        "search_codebase",
        "composio_execute",
        "generate_document",
    })

    # Native *signal* tools that MUST survive intent filtering but are NOT
    # registry-promoted (they're gated upstream by validate_tool_access, so they
    # only appear in available_tools when the affordance is enabled).
    #
    # widget_open_callback_form: when present (only when the Site has
    # callback.enabled), it must survive filtering. Without it the LLM,
    # instructed by its skill to open the form, improvises a
    # composio_execute(action="widget_open_callback_form") call that fails with
    # "'WIDGET' is not assigned to agent N". Pinning it keeps the affordance
    # available on every widget turn regardless of the classified intent.
    # Literal (not imported) to keep this hot-path module free of the heavy
    # modules.tools.* import chain; canonical name lives in
    # modules/tools/widget_callback.py::WIDGET_OPEN_CALLBACK_FORM_NAME.
    _SIGNAL_TOOL_PINS = frozenset({
        "widget_open_callback_form",
    })

    # Platform tools that are always useful and must survive intent filtering.
    # Kept static so the hot path knows them without a registry read. These are
    # unioned with every registry-promoted action in _always_include_names().
    _CORE_PLATFORM_PINS = frozenset({
        "platform_list_agents",
        "platform_get_agent",
        "platform_search_memory",
        "platform_store_memory",
        "platform_field_query",
        "platform_field_inject",
    })

    # The platform_execute dispatcher — the single door to every non-promoted
    # platform action via its (narrowed) action enum (~136 actions).
    # It is NOT an ActionDefinition, NOT registry-``promoted``, and NOT a
    # CORE_TOOL, so before PRD-232 US-001 route()'s graph and category keep-sets
    # silently stripped it (C1): a turn whose phrasing missed AutoBrain's phrase
    # map got no ``tool_hints=["platform"]`` substring rescue, so the graph
    # branch removed the dispatcher and the capability became unreachable — the
    # 2026-08-28 VECTOR "close the blocked tickets" failure. Pinning it into the
    # ONE always-include set (`_always_include_names`) keeps it reachable on
    # every branch that ships tools — whenever the agent actually carries it.
    _DISPATCHER_PINS = frozenset({
        "platform_execute",
    })

    def __init__(self):
        self.classifier = get_intent_classifier()

    def _always_include_names(self) -> Set[str]:
        """Tool names that bypass intent filtering.

        Dispatcher pin + signal pins + core platform pins + every
        registry-``promoted`` action. This is the SINGLE always-include
        mechanism — every ``route()`` branch that ships tools (hint, graph,
        category fallback) unions this set into its keep-set, so a name added
        here survives on all of them with no per-branch edit (PRD-232 US-001).

        The ``platform_execute`` dispatcher (`_DISPATCHER_PINS`) is folded in
        here rather than via a second pins pass so there is exactly one door
        list to reason about. Reading ``promoted`` here is what makes a tool
        marked ``promoted=True`` surface in chat with no edit to this router
        (PRD-122 US-010) — e.g. the full-autonomy dial
        (``platform_set/get_autonomy_level``), whose ``settings`` category is
        intentionally unmapped.

        Fail-safe: a registry import/lookup error degrades to the static pins,
        never crashes routing.
        """
        names: Set[str] = (
            set(self._DISPATCHER_PINS)
            | set(self._SIGNAL_TOOL_PINS)
            | set(self._CORE_PLATFORM_PINS)
        )
        try:
            from modules.tools.discovery.action_registry import get_action_registry

            names |= {a.name for a in get_action_registry().get_promoted()}
        except Exception:
            logger.warning(
                "[ToolRouter] promoted-tool lookup failed — using static pins only",
                exc_info=True,
            )
        return names

    async def route(
        self,
        query: str,
        available_tools: List[Dict[str, Any]],
        conversation_context: Optional[List[Dict]] = None,
        tool_hints: Optional[List[str]] = None,
        agent_id: Optional[int] = None,
        workspace_id: Optional[str] = None,
    ) -> ToolRoutingResult:
        """
        Route a query to appropriate tools.

        Args:
            query: The user's message
            available_tools: All tools available to the agent
            conversation_context: Recent conversation history
            tool_hints: PRD-68 hint keywords from AutoBrain (e.g. ["email", "github"])
            agent_id: Owning agent — scopes GraphRouter's per-agent edges/affinities
            workspace_id: Owning workspace — scopes GraphRouter's per-tenant
                edges/affinities (PRD-177 S5). Threaded to ``rank_chains``.

        Returns:
            ToolRoutingResult with filtered tools and guidance
        """
        # ── PRD-68: tool_hints from AutoBrain take priority over regex ──
        if tool_hints and available_tools:
            hint_matched = []
            for tool in available_tools:
                tool_name = tool.get("function", {}).get("name", "").lower()
                tool_desc = tool.get("function", {}).get("description", "").lower()
                for hint in tool_hints:
                    hint_lower = hint.lower()
                    if hint_lower in tool_name or hint_lower in tool_desc:
                        hint_matched.append(tool)
                        break
            if hint_matched:
                # Always include core + promoted/pinned tools alongside hint-matched tools
                must_have = self.CORE_TOOLS | self._always_include_names()
                core = [t for t in available_tools if t.get("function", {}).get("name") in must_have]
                combined = hint_matched + [c for c in core if c not in hint_matched]
                logger.info(f"[ToolRouter] PRD-68 hint match: {len(hint_matched)} tools for hints={tool_hints}")
                return ToolRoutingResult(
                    should_include_tools=True,
                    filtered_tools=combined,
                    priority_tools=[t.get("function", {}).get("name", "") for t in hint_matched[:5]],
                    tool_choice="auto",
                    reasoning=f"Tool hints: {tool_hints}",
                )
            # Hints didn't match anything — fall through to existing logic
            logger.info(f"[ToolRouter] PRD-68 hints {tool_hints} matched 0 tools, falling through")

        # Classify intent
        intent_result = self.classifier.classify(query, conversation_context)

        logger.info(f"[ToolRouter] Intent: {intent_result.primary_intent.value} "
                   f"(confidence: {intent_result.confidence:.2f}, "
                   f"requires_tools: {intent_result.requires_tools})")

        # If no tools needed, return empty
        if not intent_result.requires_tools:
            return ToolRoutingResult(
                should_include_tools=False,
                filtered_tools=[],
                priority_tools=[],
                tool_choice="none",
                reasoning=intent_result.reasoning
            )

        # PRD-141 US-014: delegate semantic ranking to GraphRouter — the single
        # tool-selection pipeline. It wraps ActionSemanticIndex + the tool-routing
        # graph and internally falls back to embedding-only ranking when the graph
        # is empty. An empty result or any failure drops through to category
        # filtering below. CORE_TOOLS / always-include pins / classifier-suggested
        # tools are always kept so the graph path never strips a tool the agent needs.
        from config import config
        if config.SEMANTIC_TOOL_ROUTING:
            try:
                from modules.tools.discovery.graph_router import get_graph_router

                chains = await get_graph_router().rank_chains(
                    query=query, workspace_id=workspace_id, agent_id=agent_id, top_k=30,
                )
                if chains:
                    keep = {name for _, _, chain in chains for name in chain}
                    keep |= self.CORE_TOOLS | self._always_include_names()
                    keep |= set(intent_result.suggested_tools or [])
                    filtered = [
                        t for t in available_tools
                        if t.get("function", {}).get("name", "") in keep
                    ]
                    if filtered:
                        return ToolRoutingResult(
                            should_include_tools=True,
                            filtered_tools=filtered,
                            priority_tools=intent_result.suggested_tools or [],
                            tool_choice=self._determine_tool_choice(intent_result, filtered),
                            reasoning=f"Graph routing: {intent_result.reasoning}",
                        )
            except Exception as e:
                from core.utils.exception_telemetry import record_error
                record_error(
                    subsystem="routing",
                    operation="graph_rank_chains",
                    error=e,
                    agent_id=agent_id,
                )

        # Fallback: ActionRegistry category matching (PRD-141 US-015)
        filtered = self._filter_tools_by_intent(available_tools, intent_result)

        tool_choice = self._determine_tool_choice(intent_result, filtered)
        priority = intent_result.suggested_tools or []

        return ToolRoutingResult(
            should_include_tools=True,
            filtered_tools=filtered,
            priority_tools=priority,
            tool_choice=tool_choice,
            reasoning=intent_result.reasoning
        )

    def _filter_tools_by_intent(
        self,
        all_tools: List[Dict[str, Any]],
        intent_result: IntentResult,
    ) -> List[Dict[str, Any]]:
        """Filter tools to the ActionRegistry categories mapped to the intent.

        PRD-141 US-015: maps the classified intent to ActionRegistry *category
        names* via ``_INTENT_TO_REGISTRY_CATEGORIES`` and pulls the matching
        action names from the registry at call time. The kept set is unioned
        with the classifier's suggested tools plus the always-on CORE_TOOLS and
        the signal/core/promoted pins from ``_always_include_names()``, so an
        action registered under an already-mapped category — or marked
        ``promoted=True`` — is auto-discoverable with no edit to this router.
        """
        categories = _INTENT_TO_REGISTRY_CATEGORIES.get(intent_result.primary_intent, [])
        suggested = set(intent_result.suggested_tools or [])

        # Nothing to narrow on (unmapped intent, no suggestions) — can't filter
        # meaningfully, so keep the full set.
        if not categories and not suggested:
            return all_tools

        relevant_names = suggested | set(self.CORE_TOOLS) | self._always_include_names()

        if categories:
            # Lazy import — the registry pulls in heavy modules.tools.* deps.
            # This is already the fallback-of-last-resort (the graph path failed
            # or is off), so a registry hiccup must not crash routing: degrade
            # to suggested ∪ CORE_TOOLS ∪ always-include pins.
            try:
                from modules.tools.discovery.action_registry import get_action_registry
                registry = get_action_registry()
                for category in categories:
                    for action in registry.get_by_category(category):
                        relevant_names.add(action.name)
            except Exception:
                logger.warning(
                    "[ToolRouter] ActionRegistry lookup failed during category "
                    "fallback — degrading to core/always/suggested tools",
                    exc_info=True,
                )

        filtered = [
            t for t in all_tools
            if isinstance(t, dict)
            and t.get("function", {}).get("name", "") in relevant_names
        ]

        # Limit to a reasonable number, keeping suggested + core + always-include
        # (signal/core/promoted) tools first so the cap never drops a pinned or
        # promoted affordance (e.g. the full-autonomy dial).
        if len(filtered) > 30:
            always = self._always_include_names()
            priority_tools = []
            other_tools = []
            for tool in filtered:
                name = tool.get("function", {}).get("name", "")
                if name in suggested or name in self.CORE_TOOLS or name in always:
                    priority_tools.append(tool)
                else:
                    other_tools.append(tool)
            filtered = priority_tools + other_tools[:max(0, 30 - len(priority_tools))]

        logger.debug(f"[ToolRouter] Filtered {len(all_tools)} tools to {len(filtered)}")
        return filtered

    def _determine_tool_choice(
        self,
        intent: IntentResult,
        filtered_tools: List[Dict]
    ) -> str:
        """
        Determine the tool_choice parameter for the LLM.

        Returns:
            "auto" - LLM decides whether to use tools
            "required" - LLM must use at least one tool
            "none" - Don't send tools at all
        """
        if not filtered_tools:
            return "none"

        # For data queries and external actions, strongly prefer tools
        if intent.primary_intent in [Intent.DATA_QUERY, Intent.EXTERNAL_ACTION, Intent.CREATION]:
            return "required"

        # For search, prefer tools but don't force
        if intent.primary_intent == Intent.SEARCH:
            return "auto"

        # For multi-step, let LLM decide
        if intent.primary_intent == Intent.MULTI_STEP:
            return "auto"

        # Default: auto
        return "auto"

    def should_skip_tools_for_message(self, query: str) -> bool:
        """
        Quick check: should we skip tools entirely for this message?

        Use this before loading all tools to save time on simple messages.
        """
        intent_result = self.classifier.classify(query)
        return not intent_result.requires_tools

    def get_composio_action_hint(self, query: str) -> Optional[str]:
        """
        Get a hint for which Composio action might be needed.

        Returns action keyword hint or None.
        """
        query_lower = query.lower()

        # Email patterns
        if any(w in query_lower for w in ["email", "mail", "inbox", "gmail"]):
            if any(w in query_lower for w in ["send", "write", "compose"]):
                return "send_email"
            elif any(w in query_lower for w in ["read", "check", "list"]):
                return "list_messages"

        # Slack patterns
        if any(w in query_lower for w in ["slack", "message", "channel"]):
            if any(w in query_lower for w in ["send", "post", "write"]):
                return "post_message"
            elif any(w in query_lower for w in ["create"]):
                return "create_channel"

        # GitHub patterns
        if any(w in query_lower for w in ["github", "repo", "repository", "pr", "pull request"]):
            if "create" in query_lower and "repo" in query_lower:
                return "create_repository"
            elif any(w in query_lower for w in ["pr", "pull request"]):
                return "pull_request"
            elif "issue" in query_lower:
                return "create_issue"

        return None


# Module-level singleton
_router = None

def get_smart_tool_router() -> SmartToolRouter:
    """Get the global smart tool router instance."""
    global _router
    if _router is None:
        _router = SmartToolRouter()
    return _router
