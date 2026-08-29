"""PRD-227 US-002 — mission lifecycle narration into the launching thread.

The coordinator narrates a mission's lifecycle (approved/started → each task
done/failed → completed/failed/cancelled) back into the chat that launched it,
reusing the PRD-205 background→chat seam (``deliver_background_message``) — never
a parallel send path. Target = the run's originating chat
(``run.config['origin_chat_id']``, captured server-side at create time) else the
creator's per-(workspace,user) Auto thread. Task-level lines are throttled above
``MISSION_NARRATION_TASK_CAP``; run-level lines always send. Source label
``"Auto · mission"`` + ``link_type="mission"`` so the bell/badge deep-link.

Seam-level tests mock the messenger + DB (the test_prd204_watch_hooks idiom);
the real-DB coordinator/messenger round-trip is the CI test.yml gate.
"""
from __future__ import annotations

import asyncio
import types
import uuid
from unittest.mock import MagicMock

import pytest

import services.coordinator_service as cs
from config import Config
from core.models.orchestration_enums import RunState, TaskState


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _run(**kw):
    """A minimal OrchestrationRun-shaped object carrying the attrs narration reads."""
    base = dict(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        goal="Ship the Q3 report",
        created_by="user_abc",
        state=RunState.RUNNING.value,
        config={"origin_chat_id": str(uuid.uuid4())},
        plan={"tasks": [{"id": 1}, {"id": 2}]},
        stop_detail=None,
        stop_reason=None,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _capture_messenger(monkeypatch):
    """Intercept every mission send at the PRD-205 seam. ``_narrate_mission``
    re-imports ``deliver_background_message`` at call time, so patching the module
    attribute catches it."""
    import services.chat_messenger as cm
    calls = []
    monkeypatch.setattr(cm, "deliver_background_message", lambda db, **kw: calls.append(kw))
    return calls


async def _async_noop(*a, **k):
    return None


# ---------------------------------------------------------------------------
# A. Narration helper units — target, throttle, provenance, fail-soft
# ---------------------------------------------------------------------------

def test_narrate_mission_targets_origin_chat(monkeypatch):
    """A run with a captured origin chat narrates INTO that chat, with the
    provenance the bell deep-links on."""
    calls = _capture_messenger(monkeypatch)
    run = _run()

    cs._narrate_mission(MagicMock(), run, "Mission approved — starting 2 tasks",
                        level="run", event="run_started")

    assert len(calls) == 1
    kw = calls[0]
    assert kw["text"] == "Mission approved — starting 2 tasks"
    assert kw["chat_id"] == run.config["origin_chat_id"]
    assert kw["clerk_user_id"] == "user_abc"
    assert kw["link_type"] == "mission"
    assert kw["link_id"] == str(run.id)
    assert kw["source"]["label"] == "Auto · mission"
    assert kw["source"]["origin"] == "mission"


def test_narrate_mission_no_origin_delegates_auto_thread(monkeypatch):
    """No origin chat → chat_id=None + clerk_user_id set, so the messenger's
    Auto-thread fallback (find_or_create_auto_chat) targets the creator."""
    calls = _capture_messenger(monkeypatch)
    run = _run(config={})  # wizard/scheduled: no originating chat

    cs._narrate_mission(MagicMock(), run, "hi", level="run", event="run_started")

    assert calls[0]["chat_id"] is None
    assert calls[0]["clerk_user_id"] == "user_abc"


def test_task_level_suppressed_over_cap(monkeypatch):
    """A run with more than the cap of tasks suppresses task-level lines."""
    calls = _capture_messenger(monkeypatch)
    n = Config.MISSION_NARRATION_TASK_CAP + 1
    run = _run(plan={"tasks": [{"id": i} for i in range(n)]})

    cs._narrate_mission(MagicMock(), run, "task line", level="task", event="task_completed")

    assert calls == [], "task-level lines must be suppressed above the cap"


def test_task_level_sends_at_cap(monkeypatch):
    """Boundary: a run with exactly the cap of tasks still narrates task lines."""
    calls = _capture_messenger(monkeypatch)
    n = Config.MISSION_NARRATION_TASK_CAP
    run = _run(plan={"tasks": [{"id": i} for i in range(n)]})

    cs._narrate_mission(MagicMock(), run, "task line", level="task", event="task_completed")

    assert len(calls) == 1


def test_run_level_always_sends_over_cap(monkeypatch):
    """Run-level lines are NEVER throttled, however large the plan."""
    calls = _capture_messenger(monkeypatch)
    n = Config.MISSION_NARRATION_TASK_CAP + 5
    run = _run(plan={"tasks": [{"id": i} for i in range(n)]})

    cs._narrate_mission(MagicMock(), run, "run line", level="run", event="run_started")

    assert len(calls) == 1


@pytest.mark.parametrize("state,prefix", [
    (RunState.COMPLETED.value, "Mission complete"),
    (RunState.FAILED.value, "Mission failed"),
    (RunState.CANCELLED.value, "Mission cancelled"),
])
def test_narrate_run_terminal_texts(monkeypatch, state, prefix):
    """Each terminal state produces a distinct, lifecycle-framed line (kept
    distinguishable from a PRD-224 watch verdict) with the mission deep-link."""
    calls = _capture_messenger(monkeypatch)
    run = _run(state=state, stop_detail="stalled" if state == RunState.FAILED.value else None)

    cs._narrate_run_terminal(MagicMock(), run)

    assert len(calls) == 1
    assert calls[0]["text"].startswith(prefix)
    assert calls[0]["link_type"] == "mission"
    assert calls[0]["link_id"] == str(run.id)
    assert calls[0]["source"]["label"] == "Auto · mission"


def test_narrate_run_terminal_ignores_non_terminal(monkeypatch):
    """A still-running run produces no terminal line."""
    calls = _capture_messenger(monkeypatch)
    cs._narrate_run_terminal(MagicMock(), _run(state=RunState.RUNNING.value))
    assert calls == []


def test_narrate_mission_is_fail_soft(monkeypatch):
    """A raising messenger must NOT propagate out of the narration helpers."""
    import services.chat_messenger as cm

    def _boom(db, **kw):
        raise RuntimeError("chat down")

    monkeypatch.setattr(cm, "deliver_background_message", _boom)
    # Neither call may raise.
    cs._narrate_mission(MagicMock(), _run(), "x", level="run", event="run_started")
    cs._narrate_run_terminal(MagicMock(), _run(state=RunState.FAILED.value))


# ---------------------------------------------------------------------------
# B. Hook sites — approve_plan (start) and _record_task_result (task terminal)
# ---------------------------------------------------------------------------

def test_approve_plan_narrates_start(monkeypatch):
    """approve_plan produces a run-level start line via the seam."""
    calls = _capture_messenger(monkeypatch)
    monkeypatch.setattr(cs, "transition_run", lambda **kw: None)
    monkeypatch.setattr(cs, "emit_event", lambda **kw: None)

    coordinator = cs.CoordinatorService()
    run = _run(state=RunState.RUNNING.value, goal="Do the thing")
    monkeypatch.setattr(coordinator, "_get_run", lambda db, rid: run)
    monkeypatch.setattr(coordinator, "_queue_initial_tasks", lambda db, r: None)

    result = coordinator.approve_plan(MagicMock(), run.id, "user_abc")

    assert result is run
    assert len(calls) == 1
    assert "approved" in calls[0]["text"].lower()
    assert calls[0]["source"]["label"] == "Auto · mission"
    assert calls[0]["link_id"] == str(run.id)


def test_approve_plan_survives_raising_messenger(monkeypatch):
    """A narration failure must not fail approve_plan (fail-soft)."""
    import services.chat_messenger as cm

    def _boom(db, **kw):
        raise RuntimeError("chat down")

    monkeypatch.setattr(cm, "deliver_background_message", _boom)
    monkeypatch.setattr(cs, "transition_run", lambda **kw: None)
    monkeypatch.setattr(cs, "emit_event", lambda **kw: None)

    coordinator = cs.CoordinatorService()
    run = _run()
    monkeypatch.setattr(coordinator, "_get_run", lambda db, rid: run)
    monkeypatch.setattr(coordinator, "_queue_initial_tasks", lambda db, r: None)

    # Must return the run despite the raising messenger.
    assert coordinator.approve_plan(MagicMock(), run.id, "user_abc") is run


def test_record_task_result_narrates_completed_task(monkeypatch):
    """_record_task_result narrates a settled task's terminal outcome (task-level)."""
    calls = _capture_messenger(monkeypatch)
    monkeypatch.setattr(cs.MissionDispatcher, "record_task_completion",
                        lambda db, task, result: None)
    monkeypatch.setattr(cs, "_dispatch_mission_event", _async_noop)

    coordinator = cs.CoordinatorService()
    monkeypatch.setattr(coordinator, "_inject_task_output_into_field", _async_noop)

    run = _run()  # 2 tasks — under the cap, so task lines are NOT suppressed
    task = types.SimpleNamespace(id=1, title="Write intro", state=TaskState.COMPLETED.value)
    result = {"status": "success", "output": "done"}

    asyncio.run(coordinator._record_task_result(MagicMock(), run, task, 5, result))

    task_lines = [c for c in calls if c["source"]["event"] == "task_completed"]
    assert len(task_lines) == 1
    assert "complete" in task_lines[0]["text"].lower()
    assert "Write intro" in task_lines[0]["text"]
    assert task_lines[0]["link_type"] == "mission"


def test_record_task_result_survives_raising_messenger(monkeypatch):
    """A narration failure must not fail _record_task_result (fail-soft)."""
    import services.chat_messenger as cm

    def _boom(db, **kw):
        raise RuntimeError("chat down")

    monkeypatch.setattr(cm, "deliver_background_message", _boom)
    monkeypatch.setattr(cs.MissionDispatcher, "record_task_completion",
                        lambda db, task, result: None)
    monkeypatch.setattr(cs, "_dispatch_mission_event", _async_noop)

    coordinator = cs.CoordinatorService()
    monkeypatch.setattr(coordinator, "_inject_task_output_into_field", _async_noop)

    run = _run()
    task = types.SimpleNamespace(id=1, title="t", state=TaskState.FAILED.value)
    # Must not raise.
    asyncio.run(
        coordinator._record_task_result(MagicMock(), run, task, 5, {"status": "error"})
    )


# ---------------------------------------------------------------------------
# C. Auto-thread fallback reaches find_or_create_auto_chat (AC #3, explicit)
# ---------------------------------------------------------------------------

def test_no_origin_reaches_find_or_create_auto_chat(monkeypatch):
    """End-to-end through the REAL messenger: a mission with no originating chat
    lands on the creator's Auto thread — find_or_create_auto_chat is invoked."""
    import services.chat_messenger as cm

    captured = {}
    fake_chat = types.SimpleNamespace(id=uuid.uuid4(), user_id=42)
    monkeypatch.setattr(cm, "_resolve_user_int_id", lambda db, cid: 42)

    def _foac(db, ws, uid):
        captured["uid"] = uid
        return fake_chat

    monkeypatch.setattr(cm, "find_or_create_auto_chat", _foac)

    class _FakeChatService:
        def __init__(self, db):
            pass

        def save_message(self, **kw):
            return types.SimpleNamespace(id=uuid.uuid4())

    monkeypatch.setattr("consumers.chatbot.service.ChatService", _FakeChatService)
    monkeypatch.setattr("services.board_events.notify_chat_event", lambda db, **kw: None)

    run = _run(config={})  # no origin_chat_id
    cs._narrate_mission(MagicMock(), run, "Mission complete", level="run", event="run_completed")

    assert captured.get("uid") == 42, "no-origin mission must reach find_or_create_auto_chat"


# ---------------------------------------------------------------------------
# D. Config knob
# ---------------------------------------------------------------------------

def test_mission_narration_task_cap_defined():
    """MISSION_NARRATION_TASK_CAP lives in config.py (default 8, Gerard 2026-08-27)."""
    assert isinstance(Config.MISSION_NARRATION_TASK_CAP, int)
    assert Config.MISSION_NARRATION_TASK_CAP == 8
