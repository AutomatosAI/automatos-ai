"""PRD-205 S6 -- scheduled-task output delivered, not discarded (PRD-77 fix).

Covers: non-empty agent output lands as a background chat delivery with
source.origin='scheduled_task' targeting the captured origin chat / creator
Auto thread; empty output posts nothing; the factory-error path keeps the
existing contract (raise -> caller records last_error) and delivers nothing;
the 200-char logger trace is retained; and the fire path (execute_task)
passes the task row's captured origin_chat_id/created_by through.

Pure unit tests: AgentFactory is patched at its module seam (no LLM), the
messenger at its seam (no session), SessionLocal at its module for the
fire-path wiring test (the agent_scheduled_tasks table is migration-only).
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def capture_chat(monkeypatch):
    import services.chat_messenger as cm

    delivered = []

    def _capture(**kwargs):
        delivered.append(kwargs)
        return None

    monkeypatch.setattr(cm, "deliver_background_message", _capture)
    return delivered


def _patch_factory(monkeypatch, result):
    import modules.agents.factory.agent_factory as af

    class _FakeFactory:
        def __init__(self, db_session=None):
            self.db_session = db_session

        async def execute_with_prompt(self, agent, prompt, context=None, use_memory=True):
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setattr(af, "AgentFactory", _FakeFactory)


def _trigger(**overrides):
    from services.scheduled_task_service import ScheduledTaskService

    kwargs = {
        "workspace_id": str(uuid.uuid4()),
        "agent_id": 9,
        "message": "[Scheduled Task #5] daily digest",
        "db": MagicMock(),
        "task_id": 5,
        "origin_chat_id": None,
        "created_by": None,
    }
    kwargs.update(overrides)
    return asyncio.run(ScheduledTaskService._trigger_agent_chat(**kwargs))


def test_output_is_delivered_with_scheduled_task_source(monkeypatch, capture_chat):
    _patch_factory(
        monkeypatch,
        {"status": "success", "result": "Here is the digest you asked for."},
    )
    origin = str(uuid.uuid4())
    ws = str(uuid.uuid4())

    _trigger(workspace_id=ws, origin_chat_id=origin, created_by="user_gerard")

    assert len(capture_chat) == 1
    sent = capture_chat[0]
    assert sent["workspace_id"] == ws
    assert sent["text"] == "Here is the digest you asked for."  # result['result'], verbatim
    assert sent["source"]["origin"] == "scheduled_task"
    assert sent["source"]["label"]
    assert sent["chat_id"] == origin
    assert sent["clerk_user_id"] == "user_gerard"
    assert sent["link_type"] == "scheduled_task"
    assert sent["link_id"] == "5"


def test_no_origin_targets_creator_auto_thread(monkeypatch, capture_chat):
    _patch_factory(monkeypatch, {"status": "success", "result": "output"})
    _trigger(origin_chat_id=None, created_by="user_creator")
    assert capture_chat[0]["chat_id"] is None
    assert capture_chat[0]["clerk_user_id"] == "user_creator"


@pytest.mark.parametrize(
    "result",
    [
        {"status": "success", "result": ""},
        {"status": "success", "result": None},
        {"status": "success", "result": "   "},
        "not-a-dict",
    ],
)
def test_empty_output_posts_nothing(monkeypatch, capture_chat, result):
    _patch_factory(monkeypatch, result)
    _trigger()
    assert capture_chat == []


def test_response_fallback_key_still_delivers(monkeypatch, capture_chat):
    """The existing result/response/output extraction chain is untouched."""
    _patch_factory(monkeypatch, {"status": "success", "response": "via response key"})
    _trigger()
    assert capture_chat[0]["text"] == "via response key"


def test_error_path_raises_and_delivers_nothing(monkeypatch, capture_chat, caplog):
    _patch_factory(monkeypatch, RuntimeError("LLM exploded"))
    with pytest.raises(RuntimeError):
        _trigger()
    assert capture_chat == []  # error path -> existing flow only, no post


def test_completion_trace_log_is_kept(monkeypatch, capture_chat, caplog):
    import logging

    _patch_factory(monkeypatch, {"status": "success", "result": "traced output"})
    with caplog.at_level(logging.INFO, logger="services.scheduled_task_service"):
        _trigger()
    assert any("completed task" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Fire-path wiring: execute_task passes the row's captured columns through
# ---------------------------------------------------------------------------


def test_execute_task_passes_row_origin_and_creator(monkeypatch):
    import core.database.database as dbmod
    from services.scheduled_task_service import ScheduledTaskService

    origin = uuid.uuid4()
    row = SimpleNamespace(
        id=5,
        workspace_id=uuid.uuid4(),
        target_agent_id=9,
        description="daily digest",
        task_type="one_shot",
        max_runs=None,
        run_count=0,
        origin_chat_id=origin,
        created_by="user_gerard",
    )

    select_result = MagicMock()
    select_result.fetchone.return_value = row

    fake_db = MagicMock()
    calls = {"n": 0}

    def _execute(*a, **k):
        calls["n"] += 1
        return select_result if calls["n"] == 1 else MagicMock()

    fake_db.execute.side_effect = _execute
    monkeypatch.setattr(dbmod, "SessionLocal", lambda: fake_db)

    captured = []

    async def _fake_trigger(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(ScheduledTaskService, "_trigger_agent_chat", _fake_trigger)

    asyncio.run(ScheduledTaskService.execute_task(5))

    assert len(captured) == 1
    sent = captured[0]
    assert sent["task_id"] == 5
    assert sent["origin_chat_id"] == str(origin)
    assert sent["created_by"] == "user_gerard"
    assert sent["workspace_id"] == str(row.workspace_id)
    assert sent["agent_id"] == 9
    assert fake_db.close.called  # session released even on the happy path
