"""
Smart Tool Router
=================

Intelligent tool routing that decides:
- WHETHER to use tools at all
- WHICH tools to make available
- HOW to prioritize tool selection

Works with the Intent Classifier to route appropriately.
"""

import logging
import re
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
    """

    # Core tools that are almost always useful
    CORE_TOOLS = {
        "search_knowledge",
        "semantic_search",
        "smart_query_database",
        "query_database",
    }

    # Tool categories
    TOOL_CATEGORIES = {
        "data": ["query_database", "smart_query_database", "sql_query"],
        "search": ["search_knowledge", "semantic_search", "search_codebase", "search_multimodal"],
        "files": ["read_file", "write_file", "list_directory", "create_directory", "delete_file"],
        "external": ["composio_execute", "composio_actions"],
        "creation": ["write_file", "create_directory"],
        "code": ["search_codebase", "execute_code", "run_command"],
    }

    # Intent to tool category mapping
    INTENT_TO_TOOLS = {
        Intent.DATA_QUERY: ["data", "search"],
        Intent.SEARCH: ["search"],
        Intent.EXTERNAL_ACTION: ["external"],
        Intent.CREATION: ["files", "creation"],
        Intent.MULTI_STEP: ["data", "search", "files", "external"],  # All tools
        Intent.MEMORY_RECALL: [],  # No tools needed
        Intent.GREETING: [],  # No tools needed
        Intent.CHITCHAT: [],  # No tools needed
        Intent.FACTUAL: [],  # Try without tools first
    }

    def __init__(self):
        self.classifier = get_intent_classifier()

    def route(
        self,
        query: str,
        available_tools: List[Dict[str, Any]],
        conversation_context: Optional[List[Dict]] = None
    ) -> ToolRoutingResult:
        """
        Route a query to appropriate tools.

        Args:
            query: The user's message
            available_tools: All tools available to the agent
            conversation_context: Recent conversation history

        Returns:
            ToolRoutingResult with filtered tools and guidance
        """
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

        # Get relevant tool categories for this intent
        relevant_categories = self.INTENT_TO_TOOLS.get(
            intent_result.primary_intent,
            []
        )

        # Filter tools by category
        filtered = self._filter_tools_by_categories(
            available_tools,
            relevant_categories,
            intent_result.suggested_tools
        )

        # Determine tool_choice strategy
        tool_choice = self._determine_tool_choice(intent_result, filtered)

        # Get priority tools (should be tried first)
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

        # Always include core tools
        relevant_names.update(self.CORE_TOOLS)

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
        if len(filtered) > 15:
            # Keep suggested + core + first N others
            priority_tools = []
            other_tools = []
            for tool in filtered:
                name = tool.get("function", {}).get("name", "")
                if name in suggested or name in self.CORE_TOOLS:
                    priority_tools.append(tool)
                else:
                    other_tools.append(tool)
            filtered = priority_tools + other_tools[:15 - len(priority_tools)]

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
