"""
Smart Tool Router
=================

Intelligent tool routing that decides:
- WHETHER to use tools at all
- WHICH tools to make available
- HOW to prioritize tool selection

Works with the Intent Classifier to route appropriately.
Supports embedding-based semantic ranking (PRD-64) with fallback to keyword matching.

NOTE: This module is consumed by ContextService (ToolLoadingStrategy.FILTERED).
It is the only caller. Future work: absorb filtering logic into
modules/context/sections/tools.py and delete this file.
See PRD-81 Task 5.3 and system audit R1 finding.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .intent_classifier import Intent, IntentResult, get_intent_classifier

logger = logging.getLogger(__name__)


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

    PRD-64: Supports embedding-based semantic ranking when SEMANTIC_TOOL_ROUTING=true.
    Falls back to keyword-based category matching if embeddings are unavailable.
    """

    # Core tools that are almost always useful
    CORE_TOOLS = frozenset({
        "search_knowledge",
        "semantic_search",
        "search_codebase",
        "smart_query_database",
        "query_database",
        "composio_execute",
        "generate_document",
    })

    # Promoted platform tools that bypass intent filtering —
    # always included regardless of detected intent (PRD-122 US-010)
    ALWAYS_INCLUDE = frozenset({
        "platform_list_agents",
        "platform_get_agent",
        "platform_search_memory",
        "platform_store_memory",
        "platform_field_query",
        "platform_field_inject",
    })

    # Tool categories
    TOOL_CATEGORIES = {
        "data": ["query_database", "smart_query_database", "sql_query"],
        "search": ["search_knowledge", "semantic_search", "search_codebase", "search_multimodal",
                    "search_tables", "search_images", "search_formulas"],
        "web_search": [
            "TAVILY_TAVILY_SEARCH", "COMPOSIO_SEARCH_FETCH_URL_CONTENT",
            "COMPOSIO_SEARCH_SEC_FILINGS", "composio_execute",
        ],
        "files": ["read_file", "write_file", "list_directory", "create_directory", "delete_file"],
        "external": ["composio_execute", "composio_actions"],
        "creation": ["write_file", "create_directory", "generate_document"],
        "document": ["generate_document", "write_file"],
        "code": ["search_codebase", "execute_code", "run_command"],
        # Promoted platform tool categories (PRD-122 US-010)
        "platform_management": [
            "platform_list_agents", "platform_get_agent",
            "platform_create_agent", "platform_update_agent",
        ],
        "marketplace": [
            "platform_browse_marketplace_agents",
            "platform_browse_marketplace_skills",
            "platform_browse_marketplace_plugins",
            "platform_install_skill", "platform_install_plugin",
        ],
        "monitoring": [
            "platform_get_system_health", "platform_get_activity_feed",
        ],
        "memory": [
            "platform_search_memory", "platform_store_memory",
        ],
        "fields": [
            "platform_field_query", "platform_field_inject",
        ],
    }

    # Intent to tool category mapping
    INTENT_TO_TOOLS = {
        Intent.DATA_QUERY: ["data", "search", "web_search", "fields"],
        Intent.SEARCH: ["search", "web_search", "code", "memory"],
        Intent.EXTERNAL_ACTION: ["external", "web_search", "document", "platform_management"],
        Intent.CREATION: ["files", "creation", "document", "external", "platform_management"],
        Intent.MULTI_STEP: [
            "data", "search", "web_search", "files", "external", "document", "code",
            "platform_management", "marketplace", "monitoring", "memory", "fields",
        ],
        Intent.MEMORY_RECALL: ["memory"],  # Memory tools for recall intents
        Intent.GREETING: [],  # No tools needed
        Intent.CHITCHAT: [],  # No tools needed
        Intent.FACTUAL: [],  # Try without tools first
    }

    def __init__(self):
        self.classifier = get_intent_classifier()

        # Embedding-based semantic ranking state (PRD-64)
        self._embedding_manager = None
        self._tool_embeddings: Dict[str, List[float]] = {}  # tool_name -> embedding
        self._embeddings_initialized = False
        self._embeddings_init_lock = asyncio.Lock()

    async def _ensure_embeddings(self, available_tools: List[Dict[str, Any]]) -> bool:
        """
        Lazy-initialize tool embeddings on first route() call.
        Returns True if semantic ranking is available.
        """
        from config import config
        if not config.SEMANTIC_TOOL_ROUTING:
            return False

        async with self._embeddings_init_lock:
            # Collect tool names that need embedding
            tools_needing_embed = []
            for tool in available_tools:
                fn = tool.get("function", {})
                name = fn.get("name", "")
                if name and name not in self._tool_embeddings:
                    desc = fn.get("description", "")
                    tools_needing_embed.append((name, f"{name}: {desc}"))

            if not tools_needing_embed and self._embeddings_initialized:
                return True

            try:
                if self._embedding_manager is None:
                    from core.llm.embedding_manager import get_embedding_manager
                    self._embedding_manager = get_embedding_manager()

                if tools_needing_embed:
                    texts = [t[1] for t in tools_needing_embed]
                    embeddings = await self._embedding_manager.generate_embeddings_batch(texts)
                    for (name, _), embedding in zip(tools_needing_embed, embeddings):
                        self._tool_embeddings[name] = embedding

                    logger.info(
                        f"[ToolRouter] Embedded {len(tools_needing_embed)} new tools "
                        f"(total cached: {len(self._tool_embeddings)})"
                    )

                self._embeddings_initialized = True
                return True

            except Exception as e:
                logger.warning(f"[ToolRouter] Embedding init failed, falling back to keyword matching: {e}")
                return False

    async def _rank_tools_by_similarity(
        self,
        query: str,
        available_tools: List[Dict[str, Any]],
        intent_result: IntentResult,
        max_tools: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Rank tools by cosine similarity between query embedding and tool embeddings.
        Applies intent boost and core tool boost.
        """
        from core.math.vector_operations import VectorOperations

        # Generate query embedding
        query_embedding = await self._embedding_manager.generate_embedding(query)

        # Score each tool
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for tool in available_tools:
            fn = tool.get("function", {})
            name = fn.get("name", "")

            tool_emb = self._tool_embeddings.get(name)
            if not tool_emb:
                continue

            score = float(VectorOperations.cosine_similarity(query_embedding, tool_emb))

            # Boost for intent-suggested tools
            if name in (intent_result.suggested_tools or []):
                score += 0.15

            # Slight boost for core tools
            if name in self.CORE_TOOLS:
                score += 0.05

            scored.append((score, tool))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])

        # Log top 5 for debugging
        top_debug = [(t.get("function", {}).get("name", "?"), f"{s:.3f}") for s, t in scored[:5]]
        logger.info(f"[ToolRouter] Semantic ranking top-5: {top_debug}")

        return [tool for _, tool in scored[:max_tools]]

    async def route(
        self,
        query: str,
        available_tools: List[Dict[str, Any]],
        conversation_context: Optional[List[Dict]] = None,
        tool_hints: Optional[List[str]] = None,
    ) -> ToolRoutingResult:
        """
        Route a query to appropriate tools.

        Args:
            query: The user's message
            available_tools: All tools available to the agent
            conversation_context: Recent conversation history
            tool_hints: PRD-68 hint keywords from AutoBrain (e.g. ["email", "github"])

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
                # Always include core + ALWAYS_INCLUDE tools alongside hint-matched tools
                must_have = self.CORE_TOOLS | self.ALWAYS_INCLUDE
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

        # Try semantic ranking first (PRD-64)
        semantic_available = await self._ensure_embeddings(available_tools)

        if semantic_available:
            try:
                filtered = await self._rank_tools_by_similarity(
                    query, available_tools, intent_result
                )
                # Ensure ALWAYS_INCLUDE tools are present even after semantic ranking
                filtered_names = {t.get("function", {}).get("name") for t in filtered}
                for tool in available_tools:
                    name = tool.get("function", {}).get("name", "")
                    if name in self.ALWAYS_INCLUDE and name not in filtered_names:
                        filtered.append(tool)
                tool_choice = self._determine_tool_choice(intent_result, filtered)
                priority = intent_result.suggested_tools or []

                return ToolRoutingResult(
                    should_include_tools=True,
                    filtered_tools=filtered,
                    priority_tools=priority,
                    tool_choice=tool_choice,
                    reasoning=f"Semantic ranking: {intent_result.reasoning}"
                )
            except Exception as e:
                logger.warning(f"[ToolRouter] Semantic ranking failed, falling back: {e}")

        # Fallback: keyword-based category matching
        relevant_categories = self.INTENT_TO_TOOLS.get(
            intent_result.primary_intent,
            []
        )

        filtered = self._filter_tools_by_categories(
            available_tools,
            relevant_categories,
            intent_result.suggested_tools
        )

        tool_choice = self._determine_tool_choice(intent_result, filtered)
        priority = intent_result.suggested_tools or []

        return ToolRoutingResult(
            should_include_tools=True,
            filtered_tools=filtered,
            priority_tools=priority,
            tool_choice=tool_choice,
            reasoning=intent_result.reasoning
        )

    def _filter_tools_by_categories(
        self,
        all_tools: List[Dict[str, Any]],
        categories: List[str],
        suggested: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter tools to only include relevant categories."""
        if not categories and not suggested:
            # For multi-step or complex queries, include all tools
            return all_tools

        # Build set of relevant tool names
        relevant_names = set(suggested) if suggested else set()
        for category in categories:
            relevant_names.update(self.TOOL_CATEGORIES.get(category, []))

        # Always include core tools and promoted always-include tools
        relevant_names.update(self.CORE_TOOLS)
        relevant_names.update(self.ALWAYS_INCLUDE)

        # Filter
        filtered = []
        for tool in all_tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function", {})
            name = fn.get("name", "")

            if name in relevant_names:
                filtered.append(tool)
            elif self._tool_matches_query(name, fn.get("description", ""), categories):
                filtered.append(tool)

        # Limit to reasonable number
        if len(filtered) > 30:
            # Keep suggested + core + first N others
            priority_tools = []
            other_tools = []
            for tool in filtered:
                name = tool.get("function", {}).get("name", "")
                if name in suggested or name in self.CORE_TOOLS:
                    priority_tools.append(tool)
                else:
                    other_tools.append(tool)
            filtered = priority_tools + other_tools[:30 - len(priority_tools)]

        logger.debug(f"[ToolRouter] Filtered {len(all_tools)} tools to {len(filtered)}")
        return filtered

    def _tool_matches_query(self, name: str, description: str, categories: List[str]) -> bool:
        """Check if a tool might be relevant based on its name/description."""
        text = f"{name} {description}".lower()

        category_keywords = {
            "data": ["database", "query", "sql", "data", "analytics"],
            "search": ["search", "find", "lookup", "knowledge"],
            "files": ["file", "directory", "folder", "write", "read"],
            "external": ["email", "slack", "github", "compose"],
            "document": ["document", "report", "pdf", "docx", "xlsx", "invoice", "export", "generate"],
            "code": ["code", "execute", "run", "command"],
        }

        for category in categories:
            keywords = category_keywords.get(category, [])
            if any(kw in text for kw in keywords):
                return True

        return False

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
