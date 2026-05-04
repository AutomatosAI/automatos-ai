"""PRD-137 Fix #7: ToolExecutionTracker prefix-based limits and dispatcher awareness.

Extracts ToolExecutionTracker and its helpers from service.py using
targeted line ranges to avoid the full import chain (pgvector etc.).
"""
import hashlib
import json
import pathlib
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Re-implement the standalone helpers (no external deps)
# ---------------------------------------------------------------------------

def _normalize_query(query: str) -> str:
    if not query:
        return ""
    normalized = re.sub(r'[^\w\s]', '', query.lower())
    return ' '.join(normalized.split())


def _queries_are_similar(query1: str, query2: str, threshold: float = 0.75) -> bool:
    norm1 = _normalize_query(query1)
    norm2 = _normalize_query(query2)
    if not norm1 or not norm2:
        return False
    if norm1 == norm2:
        return True
    return SequenceMatcher(None, norm1, norm2).ratio() >= threshold


def _extract_query_from_args(tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
    for key in ['query', 'search_query', 'q', 'text', 'question', 'prompt']:
        if key in tool_args and isinstance(tool_args[key], str):
            return tool_args[key]
    return None


# ---------------------------------------------------------------------------
# Extract ToolExecutionTracker source from service.py, load in isolation
# ---------------------------------------------------------------------------

_SERVICE_PATH = pathlib.Path(__file__).resolve().parents[1] / "consumers" / "chatbot" / "service.py"
_source = _SERVICE_PATH.read_text()

_class_start = _source.index("\nclass ToolExecutionTracker:")
_class_end = _source.index("\n\nclass ", _class_start + 1)
_class_source = _source[_class_start:_class_end]

_ns: dict = {
    "hashlib": hashlib,
    "json": json,
    "Dict": Dict,
    "Any": Any,
    "Set": Set,
    "Tuple": Tuple,
    "List": List,
    "Optional": Optional,
    "_queries_are_similar": _queries_are_similar,
    "_extract_query_from_args": _extract_query_from_args,
}
# Load the class definition without importing the full module
_code = compile(_class_source, str(_SERVICE_PATH), "exec")
# Safe: source is our own service.py, not user input
globals_copy = dict(_ns)
locals_copy: dict = {}
# Using __builtins__ to restrict the namespace is unnecessary since this
# is test code running our own source. We just need to execute the class def.
_fn = type((lambda: None))(_code, globals_copy)
_fn()
ToolExecutionTracker = globals_copy["ToolExecutionTracker"]


# ── Direct tool limits ──────────────────────────────────────────────


def test_platform_tool_capped_at_2():
    tracker = ToolExecutionTracker()
    for i in range(3):
        skip, _ = tracker.should_skip_execution("platform_get_settings", {"key": f"v{i}"})
        if not skip:
            tracker.record_execution("platform_get_settings", {"key": f"v{i}"})

    assert tracker.tool_counts.get("platform_get_settings") == 2
    skip, _ = tracker.should_skip_execution("platform_get_settings", {"key": "v3"})
    assert skip


def test_workspace_tool_capped_at_5():
    tracker = ToolExecutionTracker()
    for i in range(6):
        skip, _ = tracker.should_skip_execution("workspace_grep", {"query": f"q{i}"})
        if not skip:
            tracker.record_execution("workspace_grep", {"query": f"q{i}"})

    assert tracker.tool_counts.get("workspace_grep") == 5


def test_default_tool_capped_at_3():
    tracker = ToolExecutionTracker()
    for i in range(4):
        skip, _ = tracker.should_skip_execution("some_custom_tool", {"x": i})
        if not skip:
            tracker.record_execution("some_custom_tool", {"x": i})

    assert tracker.tool_counts.get("some_custom_tool") == 3


# ── Exact-argument dedup ────────────────────────────────────────────


def test_exact_duplicate_skipped():
    tracker = ToolExecutionTracker()
    args = {"action": "list", "filter": "active"}

    skip1, _ = tracker.should_skip_execution("some_tool", args)
    assert not skip1
    tracker.record_execution("some_tool", args)

    skip2, reason = tracker.should_skip_execution("some_tool", args)
    assert skip2
    assert "identical parameters" in reason


# ── platform_execute dispatcher awareness ───────────────────────────


def test_dispatcher_counts_by_inner_action():
    """Different actions through platform_execute should each get their own count."""
    tracker = ToolExecutionTracker()

    actions = [
        {"action": "platform_list_agents", "params": {}},
        {"action": "platform_get_settings", "params": {}},
        {"action": "platform_update_agent", "params": {"id": 1}},
    ]

    for args in actions:
        skip, _ = tracker.should_skip_execution("platform_execute", args)
        assert not skip, f"Should not skip {args['action']}"
        tracker.record_execution("platform_execute", args)

    assert tracker.tool_counts.get("platform_execute:platform_list_agents") == 1
    assert tracker.tool_counts.get("platform_execute:platform_get_settings") == 1
    assert tracker.tool_counts.get("platform_execute:platform_update_agent") == 1


def test_dispatcher_same_action_capped():
    """Repeated calls to the same dispatcher action should be capped at 2."""
    tracker = ToolExecutionTracker()
    for i in range(3):
        skip, _ = tracker.should_skip_execution(
            "platform_execute", {"action": "platform_list_agents", "params": {"page": i}}
        )
        if not skip:
            tracker.record_execution(
                "platform_execute", {"action": "platform_list_agents", "params": {"page": i}}
            )

    assert tracker.tool_counts.get("platform_execute:platform_list_agents") == 2
    skip, reason = tracker.should_skip_execution(
        "platform_execute", {"action": "platform_list_agents", "params": {"page": 99}}
    )
    assert skip
    assert "platform_execute:platform_list_agents" in reason


def test_dispatcher_exact_args_still_deduped():
    """Exact same dispatcher call (same action + same params) deduped on first repeat."""
    tracker = ToolExecutionTracker()
    args = {"action": "platform_list_agents", "params": {"filter": "active"}}

    skip1, _ = tracker.should_skip_execution("platform_execute", args)
    assert not skip1
    tracker.record_execution("platform_execute", args)

    skip2, reason = tracker.should_skip_execution("platform_execute", args)
    assert skip2
    assert "identical parameters" in reason


def test_counting_key_without_action_field():
    """platform_execute without an action field falls back to raw tool name."""
    tracker = ToolExecutionTracker()
    args = {"some_other_param": "value"}

    skip, _ = tracker.should_skip_execution("platform_execute", args)
    assert not skip
    tracker.record_execution("platform_execute", args)
    assert tracker.tool_counts.get("platform_execute") == 1


def test_mixed_dispatcher_and_direct_counted_separately():
    """Direct platform_list_agents and dispatcher platform_execute:platform_list_agents are separate."""
    tracker = ToolExecutionTracker()

    tracker.record_execution("platform_list_agents", {"workspace": "ws1"})
    tracker.record_execution("platform_execute", {"action": "platform_list_agents", "params": {}})

    assert tracker.tool_counts.get("platform_list_agents") == 1
    assert tracker.tool_counts.get("platform_execute:platform_list_agents") == 1
