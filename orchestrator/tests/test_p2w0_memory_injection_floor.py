"""PRD-185 S11: memory-injection assembly guard.

Sub-floor and noise-typed (heartbeat_log / playbook_summary) memories must never
reach the prompt. The relevance floor is applied at the L3 search boundary
(PRD-159 S3); this asserts the assembly-side guard over the merged candidate set,
which also excludes the noise content-types the search layer does not. Pure —
plain dicts, no DB / network.
"""
import sys
from pathlib import Path

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from modules.memory.injection_filter import (
    EXCLUDED_INJECTION_CONTENT_TYPES,
    filter_injectable_memories,
)


def test_sub_floor_memories_dropped():
    mems = [
        {"memory": "strong", "score": 0.9},
        {"memory": "weak", "score": 0.1},
    ]
    out = filter_injectable_memories(mems, floor=0.3)
    assert [m["memory"] for m in out] == ["strong"]


def test_unscored_memories_kept():
    # No score → cannot judge → kept (same rule as filter_by_relevance_floor).
    out = filter_injectable_memories([{"memory": "fact"}], floor=0.3)
    assert len(out) == 1


def test_noise_types_excluded_across_field_paths():
    # The tag can ride on any of several shapes; the guard must bite on all of
    # them, never silently no-op on a shape mismatch.
    mems = [
        {"memory": "keep", "score": 0.9, "content_type": "exchange"},
        {"memory": "hb-top", "score": 0.9, "content_type": "heartbeat_log"},
        {"memory": "hb-meta", "score": 0.9, "metadata": {"content_type": "heartbeat_log"}},
        {"memory": "pb-metatype", "score": 0.9, "metadata": {"type": "playbook_summary"}},
        {"memory": "pb-legacy", "score": 0.9, "category": "recipe_summary"},
    ]
    out = filter_injectable_memories(mems, floor=0.3)
    assert [m["memory"] for m in out] == ["keep"]


def test_floor_and_type_combined():
    mems = [
        {"memory": "keep", "score": 0.8, "content_type": "exchange"},
        {"memory": "low", "score": 0.05, "content_type": "exchange"},
        {"memory": "noise", "score": 0.95, "content_type": "heartbeat_log"},
    ]
    out = filter_injectable_memories(mems, floor=0.3)
    assert [m["memory"] for m in out] == ["keep"]


def test_floor_zero_disables_score_filter():
    mems = [{"memory": "a", "score": 0.01}]
    assert len(filter_injectable_memories(mems, floor=0.0)) == 1


def test_excluded_set_holds_the_noise_types():
    assert "heartbeat_log" in EXCLUDED_INJECTION_CONTENT_TYPES
    assert "playbook_summary" in EXCLUDED_INJECTION_CONTENT_TYPES


def test_smart_memory_wires_the_guard():
    # Guard the wiring: retrieve_memories must apply the filter, else the guard
    # is dead code — the failure class this wave exists to kill.
    src = (
        Path(_orchestrator_root) / "consumers" / "chatbot" / "smart_memory.py"
    ).read_text()
    assert "filter_injectable_memories(" in src, (
        "smart_memory.retrieve_memories must call the injection guard"
    )
