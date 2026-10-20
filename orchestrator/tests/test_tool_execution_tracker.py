"""PRD-137 Fix #7: ToolExecutionTracker prefix-based limits and dispatcher awareness.

PRD-256 FX-007: these tests read the live tracker (modules/tools/execution/
tool_execution_tracker.py, the one ToolLoopExecutor uses since PRD-142 W3-S4). The copy
that stayed in consumers/chatbot/service.py, which they used to extract, ran nowhere and
was deleted with it.
"""
import pytest

from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker


# ── Direct tool limits ──────────────────────────────────────────────


def test_platform_tool_capped_at_limit():
    tracker = ToolExecutionTracker()
    cap = tracker.TOOL_RETRY_LIMITS['platform_default']
    for i in range(cap + 2):
        skip, _ = tracker.should_skip_execution("platform_get_settings", {"key": f"v{i}"})
        if not skip:
            tracker.record_execution("platform_get_settings", {"key": f"v{i}"})

    assert tracker.tool_counts.get("platform_get_settings") == cap
    skip, _ = tracker.should_skip_execution("platform_get_settings", {"key": "final"})
    assert skip


def test_workspace_tool_capped_at_limit():
    tracker = ToolExecutionTracker()
    cap = tracker.TOOL_RETRY_LIMITS['workspace_default']
    for i in range(cap + 2):
        skip, _ = tracker.should_skip_execution("workspace_grep", {"query": f"q{i}"})
        if not skip:
            tracker.record_execution("workspace_grep", {"query": f"q{i}"})

    assert tracker.tool_counts.get("workspace_grep") == cap
    skip, _ = tracker.should_skip_execution("workspace_grep", {"query": "final"})
    assert skip


def test_default_tool_capped_at_limit():
    tracker = ToolExecutionTracker()
    cap = tracker.TOOL_RETRY_LIMITS['default']
    for i in range(cap + 2):
        skip, _ = tracker.should_skip_execution("some_custom_tool", {"x": i})
        if not skip:
            tracker.record_execution("some_custom_tool", {"x": i})

    assert tracker.tool_counts.get("some_custom_tool") == cap
    skip, _ = tracker.should_skip_execution("some_custom_tool", {"x": 999})
    assert skip


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
    """Repeated dispatcher calls are capped by the inner action's prefix limit.

    ``platform_execute:platform_list_agents`` resolves through the inner
    ``platform_`` prefix to ``platform_default``, not the dispatcher name.
    """
    tracker = ToolExecutionTracker()
    cap = tracker.TOOL_RETRY_LIMITS['platform_default']
    for i in range(cap + 2):
        skip, _ = tracker.should_skip_execution(
            "platform_execute", {"action": "platform_list_agents", "params": {"page": i}}
        )
        if not skip:
            tracker.record_execution(
                "platform_execute", {"action": "platform_list_agents", "params": {"page": i}}
            )

    assert tracker.tool_counts.get("platform_execute:platform_list_agents") == cap
    skip, reason = tracker.should_skip_execution(
        "platform_execute", {"action": "platform_list_agents", "params": {"page": 999}}
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


# ── P256-FIX-N17-4 (night 17, F447): a read runs again after a write that went through ──

BROWSE = {"action": "platform_browse_memories", "params": {}}
DELETE = {"action": "platform_delete_memory", "params": {"memory_id": "m-1"}}
WAITS = {"success": False, "requires_confirmation": True, "grant_id": 42, "message": "Card raised."}


def _ran(tracker, tool_name, args, result):
    tracker.record_execution(tool_name, args)
    tracker.record_outcome(tool_name, args, result)


def test_a_browse_after_a_delete_that_went_through_runs_again():
    tracker = ToolExecutionTracker()
    _ran(tracker, "platform_execute", BROWSE, {"success": True, "memories": []})
    assert tracker.should_skip_execution("platform_execute", BROWSE)[0] is True
    _ran(tracker, "platform_execute", DELETE, {"success": True})

    assert tracker.should_skip_execution("platform_execute", BROWSE)[0] is False
    assert tracker.should_skip_execution("platform_execute", DELETE)[0] is True     # the write still ran once
    assert tracker.ran_as[("platform_execute", tracker._hash_args(DELETE))] == "platform_delete_memory"


def test_a_direct_read_is_cleared_by_a_write_too():
    tracker = ToolExecutionTracker()
    listing = {"status": "review"}
    _ran(tracker, "platform_list_tasks", listing, {"success": True, "tasks": []})
    _ran(tracker, "platform_execute", DELETE, {"success": True})
    assert tracker.should_skip_execution("platform_list_tasks", listing)[0] is False


@pytest.mark.parametrize("result", [{"success": False, "error": "Memory not found"}, WAITS])
def test_a_write_that_failed_or_waits_for_the_owner_leaves_the_read_skipped(result):
    tracker = ToolExecutionTracker()
    _ran(tracker, "platform_execute", BROWSE, {"success": True, "memories": []})
    _ran(tracker, "platform_execute", DELETE, result)
    assert tracker.should_skip_execution("platform_execute", BROWSE)[0] is True


def test_a_read_after_a_read_and_a_search_after_a_write_are_still_skipped():
    tracker = ToolExecutionTracker()
    search = {"query": "posting rules"}
    _ran(tracker, "search_knowledge", search, {"success": True, "results": []})
    _ran(tracker, "platform_execute", BROWSE, {"success": True, "memories": []})
    _ran(tracker, "platform_list_tasks", {"status": "review"}, {"success": True, "tasks": []})
    assert tracker.should_skip_execution("platform_execute", BROWSE)[0] is True      # a read changes nothing
    _ran(tracker, "platform_execute", DELETE, {"success": True})
    skip, reason = tracker.should_skip_execution("search_knowledge", search)
    assert skip is True and "same search already ran" in reason                     # F085-A / F120 dedupe stands
