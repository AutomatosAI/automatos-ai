"""Tool-execution dedup + per-tool retry caps.

Extracted from ``consumers/chatbot/service.py`` as the single source of
truth for tool-loop dedup behaviour during PRD-142 W3-S4 (converge the two
tool loops onto one executor). Both the chat surface and the agent
``execute_with_prompt`` inner loop now share this tracker via
``ToolLoopExecutor`` — so dedup, prefix-based caps, and search-spiral
prevention apply identically in chat and in agent tool turns.

Standalone (stdlib-only) so it loads without triggering the heavier
``modules.tools`` import chain in unit-test environments.
"""
from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple


def _normalize_query(query: str) -> str:
    """Normalize a search query for deduplication comparison."""
    if not query:
        return ""
    normalized = re.sub(r"[^\w\s]", "", query.lower())
    return " ".join(normalized.split())


def _queries_are_similar(query1: str, query2: str, threshold: float = 0.75) -> bool:
    """Check if two queries are semantically similar (sequence-matcher ratio)."""
    norm1 = _normalize_query(query1)
    norm2 = _normalize_query(query2)
    if not norm1 or not norm2:
        return False
    if norm1 == norm2:
        return True
    return SequenceMatcher(None, norm1, norm2).ratio() >= threshold


def _extract_query_from_args(tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
    """Extract the search/query parameter from tool arguments."""
    for key in ("query", "search_query", "q", "text", "question", "prompt"):
        if key in tool_args and isinstance(tool_args[key], str):
            return tool_args[key]
    return None


class ToolExecutionTracker:
    """Tracks tool executions within one conversation turn to prevent looping.

    - Exact dedup: same tool + same canonical args → skip.
    - Semantic dedup for SEARCH_TOOLS: similar queries → skip.
    - Per-tool retry limits, with prefix-aware defaults and
      ``platform_execute`` dispatcher awareness.
    """

    SEARCH_TOOLS: Set[str] = {
        "search_knowledge", "semantic_search", "search_codebase",
        "search_tables", "search_images", "search_formulas",
        "search_multimodal", "smart_query_database", "query_database",
    }

    TOOL_RETRY_LIMITS: Dict[str, int] = {
        "composio_execute": 5,
        "search_knowledge": 5,
        "semantic_search": 5,
        "search_codebase": 5,
        "smart_query_database": 5,
        "query_database": 5,
        "list_directory": 5,
        "read_file": 8,
        "write_file": 5,
        "platform_default": 25,
        "workspace_default": 8,
        "default": 5,
    }

    def __init__(self) -> None:
        self.exact_executions: Set[Tuple[str, str]] = set()
        self.search_queries: Dict[str, List[str]] = {}
        self.tool_counts: Dict[str, int] = {}

    def _hash_args(self, tool_args: Dict[str, Any]) -> str:
        return hashlib.md5(json.dumps(tool_args, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _counting_key(tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Return the key used for per-tool call counting.

        For the ``platform_execute`` dispatcher, count by inner action so
        that distinct actions stay distinct.
        """
        if tool_name == "platform_execute":
            action = tool_args.get("action") or tool_args.get("name")
            if action:
                return f"platform_execute:{action}"
        return tool_name

    def _resolve_limit(self, counting_key: str) -> int:
        """Resolve the retry limit for a counting key, honouring prefix defaults."""
        if counting_key in self.TOOL_RETRY_LIMITS:
            return self.TOOL_RETRY_LIMITS[counting_key]
        effective_key = counting_key.split(":", 1)[-1] if ":" in counting_key else counting_key
        if effective_key.startswith("workspace_"):
            return self.TOOL_RETRY_LIMITS.get("workspace_default", self.TOOL_RETRY_LIMITS["default"])
        if effective_key.startswith("platform_") or counting_key.startswith("platform_"):
            return self.TOOL_RETRY_LIMITS.get("platform_default", self.TOOL_RETRY_LIMITS["default"])
        return self.TOOL_RETRY_LIMITS["default"]

    def should_skip_execution(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Decide whether to skip this tool call. Returns (should_skip, reason)."""
        key = self._counting_key(tool_name, tool_args)
        current_count = self.tool_counts.get(key, 0)
        limit = self._resolve_limit(key)

        if current_count >= limit:
            return True, f"Tool '{key}' has reached its execution limit ({limit}) for this turn"

        args_hash = self._hash_args(tool_args)
        exec_key = (tool_name, args_hash)
        if exec_key in self.exact_executions:
            return True, f"Tool '{tool_name}' was already executed with identical parameters"

        if tool_name in self.SEARCH_TOOLS:
            query = _extract_query_from_args(tool_name, tool_args)
            if query:
                for prev_query in self.search_queries.get(tool_name, []):
                    if _queries_are_similar(query, prev_query):
                        return True, f"Tool '{tool_name}' was already executed with a similar query"

        return False, ""

    def record_execution(self, tool_name: str, tool_args: Dict[str, Any]) -> None:
        """Record that a tool was executed (updates dedup + count + query state)."""
        args_hash = self._hash_args(tool_args)
        self.exact_executions.add((tool_name, args_hash))
        key = self._counting_key(tool_name, tool_args)
        self.tool_counts[key] = self.tool_counts.get(key, 0) + 1
        if tool_name in self.SEARCH_TOOLS:
            query = _extract_query_from_args(tool_name, tool_args)
            if query:
                self.search_queries.setdefault(tool_name, []).append(query)

    def get_execution_count(self, tool_name: str) -> int:
        return self.tool_counts.get(tool_name, 0)


__all__ = [
    "ToolExecutionTracker",
    "_normalize_query",
    "_queries_are_similar",
    "_extract_query_from_args",
]
