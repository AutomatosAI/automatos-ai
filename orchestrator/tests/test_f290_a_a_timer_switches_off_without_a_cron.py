"""F290 (night 8): "switch the timer off" through platform_schedule_playbook needs no cron.

Auto sent {enabled: false} with no cron and was refused for a missing cron_expression,
then asked the owner "Could you please provide the cron expression?" and sent "None",
null and "" as the cron. The timer already had a time. It is kept now, switched off,
and switching it back on needs no cron either.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

UK = "Europe/London"
MONDAY_7 = {"type": "cron", "cron_expression": "0 7 * * 1", "timezone": UK, "enabled": True}


@pytest.fixture
def roastery(db_session, seed_workspace):
    from core.models.core import WorkflowTemplate

    ws = UUID(seed_workspace())

    def playbook(name, schedule=None):
        made = WorkflowTemplate(template_id=f"f290a-{uuid4().hex[:8]}", name=name, workspace_id=ws,
                                description=name, template_definition={"steps": []}, steps=[],
                                created_by="f290", schedule_config=schedule)
        db_session.add(made)
        db_session.flush()
        return made

    return NS(db=db_session, ws=ws, timed=playbook("Tom's Monday Dispatch Checklist", dict(MONDAY_7)),
              untimed=playbook("Weekly Instagram posts"))


def _schedule(roastery, **params):
    from modules.tools.discovery.handlers_playbooks import schedule_playbook

    return asyncio.run(schedule_playbook(roastery.db, roastery.ws, params))


@pytest.mark.parametrize("sent", [{}, {"cron_expression": None}, {"cron_expression": ""},
                                  {"cron_expression": "None"}, {"enabled": "false"}],
                         ids=["no-cron", "null", "empty", "None-text", "false-as-text"])
def test_the_timer_goes_off_and_keeps_its_time(roastery, sent):
    out = _schedule(roastery, playbook_name="Tom's Monday Dispatch Checklist", **{"enabled": False, **sent})

    assert out["success"] is True
    roastery.db.refresh(roastery.timed)
    assert roastery.timed.schedule_config == {**MONDAY_7, "enabled": False}
    assert "Mondays at 07:00 (Europe/London)" in out["message"] and "is kept" in out["message"]


def test_switching_it_back_on_needs_no_cron_either(roastery):
    roastery.timed.schedule_config = {**MONDAY_7, "enabled": False}
    roastery.db.flush()

    out = _schedule(roastery, playbook_id=roastery.timed.id, enabled=True)

    assert out["success"] is True and "switched back on at its own time" in out["message"]
    roastery.db.refresh(roastery.timed)
    assert roastery.timed.schedule_config == MONDAY_7


def test_a_playbook_with_no_timer_says_so_and_nothing_is_written(roastery):
    out = _schedule(roastery, playbook_name="Weekly Instagram posts", enabled=False)

    assert out["success"] is False
    assert out["error"].startswith(f"Playbook #{roastery.untimed.id} 'Weekly Instagram posts' has no timer")
    roastery.db.refresh(roastery.untimed)
    assert roastery.untimed.schedule_config is None


def test_a_new_timer_with_no_time_is_never_a_question_for_the_owner(roastery):
    out = _schedule(roastery, playbook_name="Weekly Instagram posts")

    assert out["success"] is False and "never ask the owner for a cron expression" in out["error"]
    assert "'0 9 * * 1-5'" in out["error"]


def test_a_new_time_still_changes_the_timer_in_the_owners_zone(roastery):
    out = _schedule(roastery, playbook_id=roastery.timed.id, cron_expression="0 9 * * 1-5", timezone=UK)

    assert out["success"] is True
    roastery.db.refresh(roastery.timed)
    assert roastery.timed.schedule_config == {**MONDAY_7, "cron_expression": "0 9 * * 1-5"}


def test_the_tool_no_longer_requires_a_cron():
    from modules.tools.discovery import get_action_registry

    action = get_action_registry().get("platform_schedule_playbook")
    sent = {"enabled": False, "playbook_name": "Weekly Instagram posts"}           # night 8's first call
    assert [p for p in action.parameters.get("required", []) if p not in sent] == []
