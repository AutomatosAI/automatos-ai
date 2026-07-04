"""PRD-185 S4: playbook failures are VISIBLE — a notification event + a board
status of 'failed', not a silent 'done'.

A ~17-day OpenRouter 402 outage was invisible because (a) no ``playbook_failed``
event type existed and (b) ``complete_recipe_board_task`` accepted ``success`` but
hardcoded ``task.status = 'done'`` regardless. This test pins both: the event type
is registered, and the board bridge honors the flag.

Pure unit test — no DB / network (db + task are mocked).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

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
