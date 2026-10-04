"""F310 (night 9): "Please put my Monday Stock Check playbook on a timer: every Monday at 8am."

Three tries in two chats sent platform_schedule_playbook {playbook_name: "Monday Stock
Check", cron_expression: "0 8 * * 1", timezone: "Europe/London", enabled: true}, and the
playbook's schedule_config stayed null. The timer's lookup answered a name only one
playbook has with the playbook alone, the tool raised unpacking it, and the executor said
"Action 'platform_schedule_playbook' failed". Switching the timer off by name failed the
same way; only platform_update_playbook by id worked. The repeat in the same reply was
skipped as "already executed with identical parameters", which Auto read as "it ran, but
the schedule isn't applied".
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

UK = "Europe/London"
STOCK_STEP = [{"order": 1, "agent_id": None, "output_key": "stock_report",
               "prompt_template": "Report every coffee's green stock from the shop system and flag anything under 50 kg."}]
MONDAY_8 = {"type": "cron", "cron_expression": "0 8 * * 1", "timezone": UK, "enabled": True}
FAILED = {"success": False, "error": "Action 'platform_schedule_playbook' failed"}
NIGHT_9_CALL = {"action": "platform_schedule_playbook", "params": {
    "timezone": UK, "playbook_name": "Monday Stock Check", "enabled": True, "cron_expression": "0 8 * * 1"}}


@pytest.fixture
def roastery(db_session, seed_workspace):
    from core.models.core import WorkflowTemplate

    ws = UUID(seed_workspace())
    stock = WorkflowTemplate(template_id=f"f310-{uuid4().hex[:8]}", name="Monday Stock Check", workspace_id=ws,
                             description="Report green coffee stock and flag anything under 50 kg.",
                             template_definition={"steps": STOCK_STEP}, steps=STOCK_STEP, created_by="f310")
    db_session.add(stock)
    db_session.flush()
    return NS(db=db_session, ws=ws, stock=stock)


def _schedule(roastery, **params):
    from modules.tools.discovery.handlers_playbooks import schedule_playbook

    return asyncio.run(schedule_playbook(roastery.db, roastery.ws, params))


def _saved(roastery):
    roastery.db.refresh(roastery.stock)
    return roastery.stock.schedule_config


def test_the_timer_night_9_asked_for_by_name_is_saved(roastery):
    out = _schedule(roastery, **NIGHT_9_CALL["params"])                      # night 9: "… failed", three times

    assert out["success"] is True
    assert _saved(roastery) == MONDAY_8


def test_every_weekday_at_9_by_name_works_first_time_in_any_case(roastery):
    out = _schedule(roastery, playbook_name="monday stock check", cron_expression="0 9 * * 1-5", timezone=UK)

    assert out["success"] is True
    assert _saved(roastery)["cron_expression"] == "0 9 * * 1-5" and _saved(roastery)["timezone"] == UK


def test_switching_it_off_by_name_keeps_its_time(roastery):
    """Night 9, chat 046788d5: "turn that timer off" by name failed; by id it worked."""
    roastery.stock.schedule_config = dict(MONDAY_8)
    roastery.db.flush()

    out = _schedule(roastery, playbook_name="Monday stock check", enabled=False)

    assert out["success"] is True and "is kept" in out["message"]
    assert _saved(roastery) == {**MONDAY_8, "enabled": False}


def test_a_repeat_of_a_call_that_failed_says_it_failed():
    from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker

    tracker = ToolExecutionTracker()
    tracker.record_execution("platform_execute", NIGHT_9_CALL)
    tracker.record_outcome("platform_execute", NIGHT_9_CALL, FAILED)

    skipped, reason = tracker.should_skip_execution("platform_execute", NIGHT_9_CALL)

    assert skipped and "identical parameters" in reason
    assert "it failed: Action 'platform_schedule_playbook' failed." in reason   # night 9: read as "not applied"
    assert "it was not done" in reason


def test_a_repeat_of_a_call_that_worked_is_skipped_as_before():
    from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker

    tracker = ToolExecutionTracker()
    tracker.record_execution("platform_execute", NIGHT_9_CALL)
    tracker.record_outcome("platform_execute", NIGHT_9_CALL, {"success": True, "playbook_id": 114})

    skipped, reason = tracker.should_skip_execution("platform_execute", NIGHT_9_CALL)

    assert skipped and reason == "Tool 'platform_execute' was already executed with identical parameters"
