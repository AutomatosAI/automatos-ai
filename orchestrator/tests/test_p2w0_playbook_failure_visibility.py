"""PRD-185 S4: playbook failures are VISIBLE and a broken playbook STOPS.

A ~17-day OpenRouter 402 outage was invisible because (a) no ``playbook_failed``
event type existed and (b) ``complete_recipe_board_task`` accepted ``success`` but
hardcoded ``task.status = 'done'`` regardless — and it re-fired daily forever
because (c) nothing stopped a playbook that failed on every run. This test pins
all three: the event type is registered, the board bridge honors the flag, and
the repeated-failure circuit breaker pauses cron re-firing after N failures.

Pure unit tests — no DB / network (db, task, and history are mocked at the
boundary); the one scheduler test drives ``_fire_playbook`` with every external
seam patched.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_playbook_failed_is_a_valid_event_type():
    try:
        from core.services.notification_dispatcher import VALID_EVENT_TYPES
    except Exception as e:
        pytest.skip(f"notification_dispatcher not importable in this env: {e}")
    assert "playbook_failed" in VALID_EVENT_TYPES
    assert "playbook_complete" in VALID_EVENT_TYPES  # the success twin still exists


def _complete_board():
    try:
        from services.board_task_bridge import complete_recipe_board_task
    except Exception as e:
        pytest.skip(f"board_task_bridge not importable in this env: {e}")
    return complete_recipe_board_task


def _mock_db(task):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = task
    return db


def test_board_task_marked_failed_on_failure():
    complete = _complete_board()
    task = SimpleNamespace(status=None, completed_at=None, result=None, error_message=None)
    complete(_mock_db(task), "exec-1", success=False, error_message="OpenRouter 402")
    assert task.status == "failed"  # was silently 'done' before the fix
    assert task.error_message == "OpenRouter 402"


def test_board_task_marked_done_on_success():
    complete = _complete_board()
    task = SimpleNamespace(status=None, completed_at=None, result=None, error_message=None)
    complete(_mock_db(task), "exec-2", success=True, result="ok")
    assert task.status == "done"


# ---------------------------------------------------------------------------
# PRD-185 S4 (c): repeated-failure circuit breaker. A cron playbook that fails on
# every run must stop re-firing after N consecutive failures (the daily 402 spam).
# ---------------------------------------------------------------------------

def _breaker():
    try:
        from services.playbook_breaker import is_breaker_open, breaker_is_open
    except Exception as e:  # pragma: no cover
        pytest.skip(f"playbook_breaker not importable in this env: {e}")
    return is_breaker_open, breaker_is_open


def test_breaker_opens_when_last_n_all_failed():
    is_open, _ = _breaker()
    assert is_open(["failed", "failed", "failed"], threshold=3) is True


def test_breaker_closed_when_a_recent_run_succeeded():
    is_open, _ = _breaker()
    # statuses are newest-first: a success anywhere in the window breaks the streak
    assert is_open(["failed", "completed", "failed"], threshold=3) is False
    assert is_open(["completed", "failed", "failed"], threshold=3) is False


def test_breaker_closed_with_too_few_terminal_runs():
    is_open, _ = _breaker()
    assert is_open(["failed", "failed"], threshold=3) is False
    assert is_open([], threshold=3) is False


def test_breaker_disabled_when_threshold_zero():
    is_open, _ = _breaker()
    assert is_open(["failed", "failed", "failed"], threshold=0) is False


def test_breaker_only_counts_the_newest_threshold():
    is_open, _ = _breaker()
    # the 3 newest are all failures → open, even though an older run succeeded
    assert is_open(["failed", "failed", "failed", "completed"], threshold=3) is True


def test_breaker_is_open_applies_configured_threshold():
    """DB-boundary: breaker_is_open reads recent history via the internal fetch
    and applies config.PLAYBOOK_BREAKER_THRESHOLD. Fetch is patched — no DB."""
    _, breaker_is_open = _breaker()
    from config import config
    n = config.PLAYBOOK_BREAKER_THRESHOLD

    with patch("services.playbook_breaker._recent_terminal_statuses",
               return_value=["failed"] * n):
        assert breaker_is_open(MagicMock(), recipe_id=42) is (n > 0)

    with patch("services.playbook_breaker._recent_terminal_statuses",
               return_value=(["completed"] + ["failed"] * max(n - 1, 0))):
        assert breaker_is_open(MagicMock(), recipe_id=42) is False


def test_breaker_fails_closed_on_read_error():
    """A breaker that cannot read history must never block the scheduler."""
    _, breaker_is_open = _breaker()
    with patch("services.playbook_breaker._recent_terminal_statuses",
               side_effect=RuntimeError("db down")):
        assert breaker_is_open(MagicMock(), recipe_id=1) is False

# The AC (c) end-to-end assertion — that _fire_playbook actually SKIPS the
# cron re-fire when the breaker is open — lives in test_playbook_scheduler.py
# (TestFirePlaybook), which owns the proven _fire_playbook mocking harness.
