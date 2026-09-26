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
        # F120 (run 4): a ceiling, not a retry count. Eight questions in one
        # message are eight distinct searches — at 5 the last three went
        # unsearched and came back "not in the documents". Repeats are still
        # caught by the exact and similar-query checks below, whatever the count.
        "search_knowledge": 12,
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
        # F108: the actions that did what they were asked this turn (a result
        # that reports a failure does not count) — what a reply may claim.
        self.succeeded: Set[str] = set()
        # F205: the actions that answered with a failure this turn.
        self.failed: Set[str] = set()
        # F120: how many queries per search tool came from EARLIER model responses;
        # None until a caller marks rounds (then every earlier query counts).
        self._round_start: Optional[Dict[str, int]] = None

    def begin_round(self) -> None:
        """F120: the calls of one model response are one batch — similar-looking
        queries in it are separate questions (eight cafés, one price each), not a
        retry. A similar query is only a repeat of one from an earlier response."""
        self._round_start = {tool: len(queries) for tool, queries in self.search_queries.items()}

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

        # F120: a skipped search names the question it left, so the reply says
        # "not searched" — never "not in the documents".
        query = _extract_query_from_args(tool_name, tool_args) if tool_name in self.SEARCH_TOOLS else None
        if current_count >= limit:
            if query:
                return True, (
                    f'Not searched: "{query}" — this reply already ran {limit} {tool_name} calls, its '
                    "ceiling. Tell the owner this question was not searched; do not say the answer "
                    "is not in the documents."
                )
            return True, f"Tool '{key}' has reached its execution limit ({limit}) for this turn"

        args_hash = self._hash_args(tool_args)
        exec_key = (tool_name, args_hash)
        if exec_key in self.exact_executions:
            if query:
                return True, f'Not searched again: "{query}" — the same search already ran in this reply; use its result.'
            return True, f"Tool '{tool_name}' was already executed with identical parameters"

        if query:
            earlier = self.search_queries.get(tool_name, [])
            if self._round_start is not None:
                earlier = earlier[: self._round_start.get(tool_name, 0)]
            for prev_query in earlier:
                if _queries_are_similar(query, prev_query):
                    return True, (
                        f'Not searched again: "{query}" — a similar search ("{prev_query}") already ran '
                        "in this reply; use its result."
                    )

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

    def record_outcome(self, tool_name: str, tool_args: Dict[str, Any], result: Any) -> None:
        """F108: record an action that succeeded — the inner action for the
        platform_execute dispatcher. A result that says it failed
        (``success: False``, Composio's ``successful: False``) is not recorded."""
        action = self._counting_key(tool_name, tool_args).split(":", 1)[-1]
        if isinstance(result, dict) and (result.get("success") is False or result.get("successful") is False):
            self.failed.add(action)
            return
        self.succeeded.add(action)

    def get_execution_count(self, tool_name: str) -> int:
        return self.tool_counts.get(tool_name, 0)


__all__ = [
    "ToolExecutionTracker",
    "_normalize_query",
    "_queries_are_similar",
    "_extract_query_from_args",
]
