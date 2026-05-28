"""
PRD-141 US-011: Model-proportional context budgets.
====================================================

``ContextRouter._compute_budgets(context_window)`` resolves per-section token
budgets. When the model context window is known, each section is a fixed
proportion of the *usable* window (80% of the raw window). When the window is
unknown, the static ``CONTEXT_BUDGET_*`` config values are the fallback —
never the primary source when a window is available.
"""
import sys
from pathlib import Path

# Ensure orchestrator package is importable
_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

from config import config
from modules.memory.context_router import (
    ContextRouter,
    _CONTEXT_BUDGET_WEIGHTS,
    _USABLE_WINDOW_FRACTION,
)

_SECTIONS = ("session", "long_term", "temporal", "daily", "awareness", "tools", "system_prompt")


def test_context_budgets_scale_with_model():
    """A 128K model gets strictly larger section budgets than an 8K model."""
    small = ContextRouter._compute_budgets(8_000)
    large = ContextRouter._compute_budgets(128_000)

    for section in _SECTIONS:
        assert large[section] > small[section], (
            f"{section}: 128K budget ({large[section]}) should exceed "
            f"8K budget ({small[section]})"
        )


def test_context_budgets_are_proportions_of_usable_window():
    """Each section equals its weight × usable window (usable = 80% of raw)."""
    window = 128_000
    usable = int(window * _USABLE_WINDOW_FRACTION)

    budgets = ContextRouter._compute_budgets(window)

    for section, weight in _CONTEXT_BUDGET_WEIGHTS.items():
        assert budgets[section] == int(usable * weight)


def test_context_budgets_fallback_to_defaults():
    """An unknown window falls back to exactly the static config values."""
    budgets = ContextRouter._compute_budgets(None)

    assert budgets["session"] == config.CONTEXT_BUDGET_SESSION
    assert budgets["long_term"] == config.CONTEXT_BUDGET_LONG_TERM
    assert budgets["temporal"] == config.CONTEXT_BUDGET_TEMPORAL
    assert budgets["daily"] == config.CONTEXT_BUDGET_DAILY
    assert budgets["awareness"] == config.CONTEXT_BUDGET_AWARENESS
    assert budgets["tools"] == config.CONTEXT_BUDGET_TOOLS
    assert budgets["system_prompt"] == config.CONTEXT_BUDGET_SYSTEM_PROMPT


def test_context_budgets_nonpositive_window_falls_back():
    """Zero / negative windows are treated as unknown → config fallback."""
    fallback = ContextRouter._compute_budgets(None)

    assert ContextRouter._compute_budgets(0) == fallback
    assert ContextRouter._compute_budgets(-1) == fallback


def test_context_budgets_always_cover_all_sections():
    """Both the proportional and fallback paths return all seven sections."""
    proportional = ContextRouter._compute_budgets(32_000)
    fallback = ContextRouter._compute_budgets(None)

    assert set(proportional) == set(_SECTIONS)
    assert set(fallback) == set(_SECTIONS)


def test_context_budget_weights_leave_response_slack():
    """Weights sum to 0.80 of usable — the rest is slack, never over-allocated."""
    total = sum(_CONTEXT_BUDGET_WEIGHTS.values())
    assert abs(total - 0.80) < 1e-9
    assert total < 1.0  # must never claim the whole usable window
