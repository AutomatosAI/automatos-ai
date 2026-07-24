"""
PRD-141 US-012: Adaptive context-guard thresholds.
===================================================

``_thresholds_for_model(context_window)`` returns a
``(compact_threshold, keep_recent_turns)`` tuple that scales with the model's
context window — large windows compact later and keep more turns; small
windows compact earlier and keep fewer, avoiding context_length_exceeded
(provider 400s) and runaway memory.

Tiers:
    >=200K -> (0.90, 12)
    >=100K -> (0.85, 8)
    >= 32K -> (0.80, 6)
    >=  8K -> (0.75, 4)
    else   -> (0.70, 3)

An unknown / non-positive window falls back to the static COMPACT_THRESHOLD /
KEEP_RECENT_TURNS constants.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Ensure orchestrator package is importable
_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

from core.context_guard import (
    ContextGuard,
    COMPACT_THRESHOLD,
    KEEP_RECENT_TURNS,
    _thresholds_for_model,
)


# ---------------------------------------------------------------------------
# Pure threshold resolution
# ---------------------------------------------------------------------------

def test_thresholds_exact_tiers():
    """Each documented tier returns its exact (threshold, keep_turns) tuple."""
    assert _thresholds_for_model(200_000) == (0.90, 12)
    assert _thresholds_for_model(128_000) == (0.85, 8)
    assert _thresholds_for_model(100_000) == (0.85, 8)
    assert _thresholds_for_model(32_000) == (0.80, 6)
    assert _thresholds_for_model(8_000) == (0.75, 4)
    assert _thresholds_for_model(4_000) == (0.70, 3)


def test_tier_boundaries_are_inclusive_lower_bounds():
    """A window exactly on a boundary takes that tier; one below drops a tier."""
    assert _thresholds_for_model(200_000) == (0.90, 12)
    assert _thresholds_for_model(199_999) == (0.85, 8)
    assert _thresholds_for_model(100_000) == (0.85, 8)
    assert _thresholds_for_model(99_999) == (0.80, 6)
    assert _thresholds_for_model(32_000) == (0.80, 6)
    assert _thresholds_for_model(31_999) == (0.75, 4)
    assert _thresholds_for_model(8_000) == (0.75, 4)
    assert _thresholds_for_model(7_999) == (0.70, 3)


def test_compact_threshold_adapts_to_context():
    """The compaction threshold rises monotonically with the context window."""
    windows = [4_000, 8_000, 32_000, 100_000, 200_000]
    thresholds = [_thresholds_for_model(w)[0] for w in windows]

    assert thresholds == sorted(thresholds)            # non-decreasing
    assert thresholds[0] < thresholds[-1]              # genuinely adapts
    assert _thresholds_for_model(200_000)[0] > _thresholds_for_model(8_000)[0]


def test_keep_recent_turns_adapts():
    """Kept-turns rises monotonically with the context window."""
    windows = [4_000, 8_000, 32_000, 100_000, 200_000]
    kept = [_thresholds_for_model(w)[1] for w in windows]

    assert kept == sorted(kept)                        # non-decreasing
    assert kept[0] < kept[-1]                          # genuinely adapts
    assert _thresholds_for_model(200_000)[1] > _thresholds_for_model(8_000)[1]


def test_thresholds_fallback_on_unknown_window():
    """An unknown / non-positive window falls back to the static constants."""
    fallback = (COMPACT_THRESHOLD, KEEP_RECENT_TURNS)
    assert _thresholds_for_model(None) == fallback
    assert _thresholds_for_model(0) == fallback
    assert _thresholds_for_model(-1) == fallback


def test_small_window_compacts_more_aggressively_than_large():
    """Safety property: a small window must NOT use a larger budget than a big one."""
    small_thr, small_keep = _thresholds_for_model(8_000)
    large_thr, large_keep = _thresholds_for_model(200_000)
    assert small_thr <= large_thr
    assert small_keep <= large_keep


# ---------------------------------------------------------------------------
# keep_recent_turns actually flows into compaction
# ---------------------------------------------------------------------------

def _conversation(n_turns: int):
    """A system message + n alternating user/assistant turns."""
    msgs = [{"role": "system", "content": "You are a helpful agent."}]
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"message {i}"})
    return msgs


@pytest.mark.asyncio
async def test_compact_respects_keep_recent_turns():
    """_compact keeps exactly keep_recent_turns recent messages and compacts the rest."""
    guard = ContextGuard()
    guard._summarize = AsyncMock(return_value="SUMMARY")
    messages = _conversation(20)  # 1 system + 20 turns

    compacted = await guard._compact(
        messages=messages,
        llm_manager=MagicMock(),
        workspace_id=None,  # skip memory flush
        keep_recent_turns=4,
    )

    # result = system_msgs + [tombstone] + recent_turns
    recent = [m for m in compacted if not m.get("_compact_tombstone") and m.get("role") != "system"]
    assert len(recent) == 4
    assert recent[-1]["content"] == "message 19"
    tombstone = next(m for m in compacted if m.get("_compact_tombstone"))
    assert tombstone["_compact_tombstone"]["compacted_count"] == 16  # 20 - 4


@pytest.mark.asyncio
async def test_larger_keep_recent_turns_compacts_fewer():
    """A larger keep_recent_turns preserves more turns → compacts fewer."""
    guard = ContextGuard()
    guard._summarize = AsyncMock(return_value="SUMMARY")
    messages = _conversation(20)

    compacted = await guard._compact(
        messages=messages,
        llm_manager=MagicMock(),
        workspace_id=None,
        keep_recent_turns=12,
    )

    recent = [m for m in compacted if not m.get("_compact_tombstone") and m.get("role") != "system"]
    assert len(recent) == 12
    tombstone = next(m for m in compacted if m.get("_compact_tombstone"))
    assert tombstone["_compact_tombstone"]["compacted_count"] == 8  # 20 - 12
