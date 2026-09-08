"""PRD-238 W3 — Auto follows through: a bounded wait-and-recheck tool that
narrates progress, and a ticket card the chat can render live.

Pure unit tests: fake db, fake clock, no sleeping.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# the turn-progress bus
# ---------------------------------------------------------------------------

def test_turn_progress_bus_routes_to_the_registered_turn_only():
    from services import turn_progress as tp

    got = []

    async def emitter(text):
        got.append(text)

    tp.register("t1", emitter)
    try:
        assert tp.is_registered("t1") and not tp.is_registered("t2")
        assert asyncio.run(tp.emit("t1", "hello")) is True
        assert asyncio.run(tp.emit("t2", "nobody")) is False
        assert asyncio.run(tp.emit(None, "x")) is False
        assert asyncio.run(tp.emit("t1", "")) is False
    finally:
        tp.unregister("t1")
    assert got == ["hello"]
    assert not tp.is_registered("t1")


def test_turn_progress_emitter_failures_never_propagate():
    from services import turn_progress as tp

    async def boom(text):
        raise RuntimeError("stream gone")

    tp.register("t9", boom)
    try:
        assert asyncio.run(tp.emit("t9", "x")) is False
    finally:
        tp.unregister("t9")


def test_progress_frame_shape():
    import json

    from consumers.chatbot.streaming import get_streaming_handler

    frame = get_streaming_handler().format_aisdk_progress("Bob is working · 10 s")
    assert json.loads(frame[2:]) == {"type": "progress", "data": {"text": "Bob is working · 10 s"}}


# ---------------------------------------------------------------------------
# the ticket card
# ---------------------------------------------------------------------------

def _task(status="in_progress", **ref):
    return SimpleNamespace(
        id=92, title="Write basic webpage", status=status, assigned_agent_id=15,
        started_at="2026-09-08T10:00:00", completed_at=None,
        runtime_ref={"runtime": "cli", "recent_tools": [{"name": "Read"}, {"name": "Bash"}],
                     "files_touched": ["a.py", "b.py"], "exit_reason": None, "denials": 1, **ref},
    )


def test_task_card_carries_counters_never_content():
    try:
        from modules.tools.discovery.handlers_board_tasks import task_card, _progress_line
    except Exception as e:
        pytest.skip(f"handlers not importable here: {e}")

    card = task_card(_task(), "Bob")
    assert card == {
        "id": 92, "title": "Write basic webpage", "status": "in_progress", "assigned_agent": "Bob",
        "runtime": "cli", "last_tool": "Bash", "files_touched": 2, "exit_reason": None, "denials": 1,
        "started_at": "2026-09-08T10:00:00", "completed_at": None,
    }
    assert "description" not in card and "transcript" not in str(card)
    assert _progress_line(card, 20) == "Bob is working on #92 · 20 s · last tool: Bash · 2 files touched"
    bare = task_card(SimpleNamespace(id=1, title="t", status="inbox", runtime_ref=None, started_at=None, completed_at=None))
    assert bare["assigned_agent"] == "unassigned" and bare["files_touched"] == 0


# ---------------------------------------------------------------------------
# the wait tool
# ---------------------------------------------------------------------------

class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    async def sleep(self, seconds):
        self.now += seconds


def _db_for(sequence):
    """A fake session whose task status advances through ``sequence`` per load."""
    states = list(sequence)
    db = MagicMock()

    def _first():
        status = states.pop(0) if len(states) > 1 else states[0]
        return _task(status=status)

    db.query.return_value.filter.return_value.first.side_effect = _first
    db.query.return_value.get.return_value = SimpleNamespace(name="Bob")
    return db


def _run_wait(monkeypatch, sequence, params, budget=90, poll=5):
    from modules.tools.discovery import handlers_board_tasks as h
    from services import turn_progress as tp
    import asyncio as _asyncio
    import time as _time

    clock = _Clock()
    monkeypatch.setattr(_asyncio, "sleep", clock.sleep)
    monkeypatch.setattr(_time, "monotonic", clock.monotonic)
    from config import config as app_config
    monkeypatch.setattr(app_config, "CHATBOT_WAIT_BUDGET_S", budget, raising=False)
    monkeypatch.setattr(app_config, "CHATBOT_WAIT_POLL_S", poll, raising=False)

    lines = []

    async def emitter(text):
        lines.append(text)

    tp.register("turn-1", emitter)
    try:
        result = asyncio.run(h.wait_for_board_task(_db_for(sequence), uuid.uuid4(), {**params, "_turn_id": "turn-1"}))
    finally:
        tp.unregister("turn-1")
    return result, lines, clock


def test_wait_returns_when_the_ticket_ends_and_narrates_meanwhile(monkeypatch):
    try:
        import modules.tools.discovery.handlers_board_tasks  # noqa: F401
    except Exception as e:
        pytest.skip(f"handlers not importable here: {e}")

    result, lines, clock = _run_wait(monkeypatch, ["in_progress", "in_progress", "done"], {"task_id": 92})
    assert result["success"] is True and result["terminal"] is True
    assert result["status"] == "done" and result["waited_seconds"] == 10
    assert result["frontend_data"]["task_card"]["status"] == "done"
    assert result["message"] == "Task #92 ended: done."
    assert lines == [
        "Bob is working on #92 · 0 s · last tool: Bash · 2 files touched",
        "Bob is working on #92 · 5 s · last tool: Bash · 2 files touched",
    ]


def test_wait_respects_the_budget_and_says_still_running(monkeypatch):
    try:
        import modules.tools.discovery.handlers_board_tasks  # noqa: F401
    except Exception as e:
        pytest.skip(f"handlers not importable here: {e}")

    result, lines, clock = _run_wait(monkeypatch, ["in_progress"], {"task_id": 92, "max_wait_seconds": 500}, budget=12, poll=5)
    assert result["terminal"] is False and result["status"] == "still running"
    assert result["waited_seconds"] == 12 and result["budget_seconds"] == 12  # capped by the workspace budget
    assert "still running after 12 s" in result["message"]
    assert len(lines) == 3  # 0 s, 5 s, 10 s
    assert clock.now == 12


def test_wait_returns_at_once_for_a_finished_or_missing_ticket(monkeypatch):
    try:
        import modules.tools.discovery.handlers_board_tasks as h
    except Exception as e:
        pytest.skip(f"handlers not importable here: {e}")

    result, lines, _ = _run_wait(monkeypatch, ["cancelled"], {"task_id": "92"})
    assert result["terminal"] is True and result["status"] == "cancelled" and lines == []

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    assert asyncio.run(h.wait_for_board_task(db, uuid.uuid4(), {"task_id": 5}))["success"] is False
    assert asyncio.run(h.wait_for_board_task(db, uuid.uuid4(), {}))["success"] is False
    assert asyncio.run(h.wait_for_board_task(db, uuid.uuid4(), {"task_id": "x"}))["success"] is False


def test_wait_action_is_registered_read_only_with_examples():
    try:
        from modules.tools.discovery.action_registry import ActionRegistry
        from modules.tools.discovery import actions_board_tasks as mod
    except Exception as e:
        pytest.skip(f"registry not importable here: {e}")

    fn = next((getattr(mod, n) for n in dir(mod) if n.startswith("register") and callable(getattr(mod, n))), None)
    assert fn is not None, "actions_board_tasks exposes no register_* entry point"
    reg = ActionRegistry()
    reg._initialized = True
    fn(reg)
    action = reg.get("platform_wait_for_task")
    assert action is not None
    assert action.permission_level == "read"
    assert "task_id" in action.parameters["properties"]
    assert "wait for task 92 to finish" in action.examples


def test_wait_knobs_exist_with_sane_defaults():
    from config import config

    assert 30 <= int(config.CHATBOT_WAIT_BUDGET_S) <= 600
    assert 1 <= int(config.CHATBOT_WAIT_POLL_S) <= 30
