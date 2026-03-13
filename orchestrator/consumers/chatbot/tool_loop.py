"""
Tool Loop Prevention Utilities
==============================

Tracks tool executions within a conversation turn to prevent looping.
Implements exact deduplication, semantic deduplication for search tools,
and per-tool retry limits.
"""

import hashlib
import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple


def _normalize_query(query: str) -> str:
    """Normalize a search query for deduplication comparison."""
    if not query:
        return ""
    normalized = re.sub(r'[^\w\s]', '', query.lower())
    normalized = ' '.join(normalized.split())
    return normalized


def _queries_are_similar(query1: str, query2: str, threshold: float = 0.75) -> bool:
    """Check if two queries are semantically similar using string similarity."""
    norm1 = _normalize_query(query1)
    norm2 = _normalize_query(query2)
    if not norm1 or not norm2:
        return False
    if norm1 == norm2:
        return True
    ratio = SequenceMatcher(None, norm1, norm2).ratio()
    return ratio >= threshold


def _extract_query_from_args(tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
    """Extract the search/query parameter from tool arguments."""
    query_keys = ['query', 'search_query', 'q', 'text', 'question', 'prompt']
    for key in query_keys:
        if key in tool_args and isinstance(tool_args[key], str):
            return tool_args[key]
    return None


class ToolExecutionTracker:
    """
    Tracks tool executions within a conversation turn to prevent looping.
    Implements:
    - Exact deduplication (same tool + same args)
    - Semantic deduplication for search tools (similar queries)
    - Per-tool retry limits
    """

    SEARCH_TOOLS = {
        'search_knowledge', 'semantic_search', 'search_codebase',
        'search_tables', 'search_images', 'search_formulas',
        'search_multimodal', 'smart_query_database', 'query_database'
    }

    TOOL_RETRY_LIMITS = {
        'composio_execute': 2,
        'search_knowledge': 2,
        'semantic_search': 2,
        'search_codebase': 2,
        'smart_query_database': 2,
        'query_database': 2,
        'list_directory': 2,
        'read_file': 3,
        'write_file': 2,
        'default': 3
    }

    def __init__(self):
        self.exact_executions: Set[Tuple[str, str]] = set()
        self.search_queries: Dict[str, List[str]] = {}
        self.tool_counts: Dict[str, int] = {}

    def _hash_args(self, tool_args: Dict[str, Any]) -> str:
        return hashlib.md5(json.dumps(tool_args, sort_keys=True).encode()).hexdigest()

    def should_skip_execution(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check if a tool execution should be skipped. Returns (should_skip, reason)."""
        current_count = self.tool_counts.get(tool_name, 0)
        limit = self.TOOL_RETRY_LIMITS.get(tool_name, self.TOOL_RETRY_LIMITS['default'])

        if current_count >= limit:
            return True, f"Tool '{tool_name}' has reached its execution limit ({limit}) for this turn"

        args_hash = self._hash_args(tool_args)
        exec_key = (tool_name, args_hash)
        if exec_key in self.exact_executions:
            return True, f"Tool '{tool_name}' was already executed with identical parameters"

        if tool_name in self.SEARCH_TOOLS:
            query = _extract_query_from_args(tool_name, tool_args)
            if query:
                previous_queries = self.search_queries.get(tool_name, [])
                for prev_query in previous_queries:
                    if _queries_are_similar(query, prev_query):
                        return True, f"Tool '{tool_name}' was already executed with a similar query"

        return False, ""

    def record_execution(self, tool_name: str, tool_args: Dict[str, Any]) -> None:
        """Record that a tool was executed."""
        args_hash = self._hash_args(tool_args)
        self.exact_executions.add((tool_name, args_hash))
        self.tool_counts[tool_name] = self.tool_counts.get(tool_name, 0) + 1
        if tool_name in self.SEARCH_TOOLS:
            query = _extract_query_from_args(tool_name, tool_args)
            if query:
                if tool_name not in self.search_queries:
                    self.search_queries[tool_name] = []
                self.search_queries[tool_name].append(query)

    def get_execution_count(self, tool_name: str) -> int:
        return self.tool_counts.get(tool_name, 0)
