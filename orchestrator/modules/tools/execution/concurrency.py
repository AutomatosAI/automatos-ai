"""
Tool Concurrency Predicates — Input-Aware Batching
===================================================

Classifies tool calls as read-safe (can run in parallel) or mutating
(must run serially). Safety is determined per-invocation based on the
actual input arguments, not just the tool name.

Inspired by free-code's toolOrchestration.ts partition-then-execute pattern.

Usage:
    from modules.tools.execution.concurrency import is_read_safe, partition_tool_batch

    safe, unsafe = partition_tool_batch(tool_calls_prepared)
    # Run safe tools concurrently, then unsafe tools serially.
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Tools that are ALWAYS read-safe regardless of input
_ALWAYS_READ_SAFE: frozenset[str] = frozenset({
    # Platform read tools
    "platform_execute",  # Only when action is a read — checked below
    "query_database",
    "smart_query_database",
    "search_knowledge",
    "search_documents",
    "semantic_search",
    # Research tools
    "web_search",
    "web_scrape",
    "web_browse",
    # Memory reads
    "platform_search_memory",
    "platform_get_memories",
    # Board reads
    "platform_list_board_tasks",
    "platform_get_board_task",
    # Agent reads
    "platform_list_agents",
    "platform_get_agent",
    # Workspace reads
    "platform_get_workspace",
    "platform_list_workspaces",
})

# Tools that are ALWAYS mutating — never run in parallel
_ALWAYS_MUTATING: frozenset[str] = frozenset({
    "run_command",
    "execute_code",
    "write_file",
    "delete_file",
    "platform_create_mission",
    "platform_create_agent",
    "platform_update_agent",
    "platform_delete_agent",
    "platform_store_memory",
    "platform_create_board_task",
    "platform_update_board_task",
})

# Platform actions that are read-safe (used when tool_name == "platform_execute")
_READ_SAFE_PLATFORM_ACTIONS: frozenset[str] = frozenset({
    "list_agents",
    "get_agent",
    "list_board_tasks",
    "get_board_task",
    "search_memory",
    "get_memories",
    "list_workspaces",
    "get_workspace",
    "list_recipes",
    "get_recipe",
    "list_missions",
    "get_mission",
    "list_documents",
    "get_document",
    "search_knowledge",
})


def is_read_safe(tool_name: str, tool_args: Dict[str, Any]) -> bool:
    """
    Determine if a tool call is read-safe (can run concurrently).

    Args:
        tool_name: The tool being called.
        tool_args: The parsed arguments for this specific invocation.

    Returns:
        True if this invocation is safe to run concurrently with other
        read-safe tools. False if it must run serially.
    """
    # Explicit mutating tools
    if tool_name in _ALWAYS_MUTATING:
        return False

    # Explicit read-safe tools
    if tool_name in _ALWAYS_READ_SAFE:
        # Special case: platform_execute depends on the action
        if tool_name == "platform_execute":
            action = (tool_args.get("action") or "").strip().lower()
            return action in _READ_SAFE_PLATFORM_ACTIONS
        return True

    # Composio tools: classify by prefix convention
    # Most Composio actions follow APP_ACTION pattern
    action_lower = tool_name.lower()
    if any(action_lower.startswith(prefix) for prefix in (
        "gmail_fetch", "gmail_get", "gmail_list",
        "slack_get", "slack_list", "slack_search",
        "github_get", "github_list", "github_search",
        "notion_get", "notion_list", "notion_search",
        "calendar_get", "calendar_list",
        "drive_get", "drive_list", "drive_search",
    )):
        return True

    # Default: assume mutating (safe default)
    return False


def partition_tool_batch(
    tool_calls: List[Tuple[str, str, Dict[str, Any]]],
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], List[Tuple[str, str, Dict[str, Any]]]]:
    """
    Partition a batch of tool calls into read-safe and mutating groups.

    Preserves order within each group. Read-safe tools can be executed
    concurrently; mutating tools must be executed serially.

    Args:
        tool_calls: List of (tool_id, tool_name, tool_call_dict) tuples.

    Returns:
        (read_safe, mutating) — two lists of tool call tuples.
    """
    read_safe = []
    mutating = []

    for tool_id, tool_name, tool_call in tool_calls:
        try:
            args_str = tool_call.get("function", {}).get("arguments", "{}")
            tool_args = __import__("json").loads(args_str) if isinstance(args_str, str) else (args_str or {})
        except Exception:
            tool_args = {}

        if is_read_safe(tool_name, tool_args):
            read_safe.append((tool_id, tool_name, tool_call))
        else:
            mutating.append((tool_id, tool_name, tool_call))

    if read_safe and mutating:
        logger.info(
            "Tool batch partitioned: %d read-safe (parallel), %d mutating (serial)",
            len(read_safe),
            len(mutating),
        )

    return read_safe, mutating
