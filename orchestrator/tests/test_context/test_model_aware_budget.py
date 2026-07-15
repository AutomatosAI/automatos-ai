"""PRD-201 S3 — model-aware budgets.

Pure — the DB window lookup is mocked. ``TokenBudget.total`` is sized to the
model context window instead of a hardcoded 128k; a mode ``max_tokens`` override
still wins; small windows don't drive ``available_for_sections`` negative.
"""

import sys
from pathlib import Path
from unittest.mock import patch

_ORCH = Path(__file__).resolve().parent.parent.parent
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from modules.context.modes import MODE_CONFIGS, ContextMode
from modules.context.service import ContextService

_CHATBOT = MODE_CONFIGS[ContextMode.CHATBOT]  # max_tokens=None → uses the window
_HEARTBEAT_ORCH = MODE_CONFIGS[ContextMode.HEARTBEAT_ORCHESTRATOR]  # max_tokens=8000 override


def _budget(model, window):
    with patch("modules.context.service.get_context_window", return_value=window):
        return ContextService._get_budget(ContextMode.CHATBOT, _CHATBOT, model, db_session=None)


def test_budget_total_from_model_window():
    small = _budget("small-32k", 32_000)
    large = _budget("large-200k", 200_000)
    assert small.total == 32_000
    assert large.total == 200_000
    assert small.total < large.total  # window drives the ceiling, not a fixed 128k


def test_large_window_uses_full_capacity():
    huge = _budget("million", 1_000_000)
    assert huge.total == 1_000_000
    # A 1M window leaves far more section space than the old fixed 128k.
    assert huge.available_for_sections > 128_000


def test_budget_falls_back_when_model_unknown():
    # get_context_window returns its safe default (128k) for an unregistered model.
    with patch("modules.context.service.get_context_window", return_value=128_000):
        b = ContextService._get_budget(ContextMode.CHATBOT, _CHATBOT, "who-knows", db_session=None)
    assert b.total == 128_000  # no crash, sane default


def test_mode_max_tokens_override_wins_over_window():
    # HEARTBEAT_ORCHESTRATOR pins max_tokens=8000; the model window must not raise it.
    with patch("modules.context.service.get_context_window", return_value=1_000_000):
        b = ContextService._get_budget(
            ContextMode.HEARTBEAT_ORCHESTRATOR, _HEARTBEAT_ORCH, "million", db_session=None
        )
    assert b.total == 8_000


def test_small_window_never_goes_negative():
    # CHATBOT reserves 60k for messages at 128k; on a 32k window that reservation
    # must scale down so section space stays non-negative.
    b = _budget("small-32k", 32_000)
    assert b.available_for_sections >= 0
    assert b.reserved_for_messages < 32_000
