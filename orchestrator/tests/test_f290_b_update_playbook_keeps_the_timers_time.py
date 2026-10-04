"""F290 (night 8): platform_update_playbook switches a timer off and keeps its time.

{schedule_config: {enabled: false}} was refused for having no "type", and Auto gave
up; {type: "manual", enabled: false} was saved and the weekday-9 setting was gone.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

UK = "Europe/London"
WEEKDAY_9 = {"type": "cron", "cron_expression": "0 9 * * 1-5", "timezone": UK, "enabled": True}


@pytest.fixture
def roastery(db_session, seed_workspace):
    from core.models.core import WorkflowTemplate

    ws = UUID(seed_workspace())

    def playbook(name, schedule=None):
        made = WorkflowTemplate(template_id=f"f290b-{uuid4().hex[:8]}", name=name, workspace_id=ws,
                                description=name, template_definition={"steps": []}, steps=[],
                                created_by="f290", schedule_config=schedule)
        db_session.add(made)
        db_session.flush()
        return made

    return NS(db=db_session, ws=ws, timed=playbook("Weekly Instagram posts", dict(WEEKDAY_9)),
              untimed=playbook("Tom's checklist"))


def _update(roastery, playbook, **params):
    from modules.tools.discovery.handlers_playbooks import update_playbook

    return asyncio.run(update_playbook(roastery.db, roastery.ws, {"playbook_id": playbook.id, **params}))


@pytest.mark.parametrize("given", [{"enabled": False}, {"type": "manual", "enabled": False}, {"type": "manual"},
                                   {}, {"type": "cron", "enabled": False}],
                         ids=["enabled-false", "manual-off", "manual", "empty", "cron-without-time"])
def test_switching_the_timer_off_keeps_its_time(roastery, given):
    out = _update(roastery, roastery.timed, schedule_config=given)

    assert out["success"] is True and "weekdays at 09:00 (Europe/London), is kept" in out["message"]
    roastery.db.refresh(roastery.timed)
    assert roastery.timed.schedule_config == {**WEEKDAY_9, "enabled": False}


def test_enabled_true_switches_it_back_on_at_its_own_time(roastery):
    roastery.timed.schedule_config = {**WEEKDAY_9, "enabled": False}
    roastery.db.flush()

    out = _update(roastery, roastery.timed, schedule_config={"enabled": True})

    assert out["success"] is True and "back on at its own time" in out["message"]
    roastery.db.refresh(roastery.timed)
    assert roastery.timed.schedule_config == WEEKDAY_9


def test_a_complete_new_cron_still_replaces_the_timer(roastery):
    new = {"type": "cron", "cron_expression": "0 10 * * *", "timezone": UK, "enabled": True}
    out = _update(roastery, roastery.timed, schedule_config=new)

    assert out["success"] is True
    roastery.db.refresh(roastery.timed)
    assert roastery.timed.schedule_config == new


def test_a_playbook_with_no_timer_says_so_and_nothing_changes(roastery):
    out = _update(roastery, roastery.untimed, schedule_config={"enabled": False})

    assert out["success"] is False
    assert out["error"].startswith(f"Playbook #{roastery.untimed.id} 'Tom's checklist' has no timer")
    roastery.db.refresh(roastery.untimed)
    assert roastery.untimed.schedule_config is None


def test_a_null_schedule_still_changes_nothing_of_the_timer(roastery):
    out = _update(roastery, roastery.timed, description="Captions for the week", schedule_config=None)

    assert out["success"] is True
    roastery.db.refresh(roastery.timed)
    assert roastery.timed.schedule_config == WEEKDAY_9


def test_schedule_is_read_as_schedule_config():
    """Night 8 sent {"name": …, "schedule": {"enabled": false}} once."""
    from modules.tools.discovery import get_action_registry
    from modules.tools.execution.unified_executor import map_optional_aliases, undeclared_params_refusal

    action = get_action_registry().get("platform_update_playbook")
    params = map_optional_aliases("platform_update_playbook", action,
                                  {"playbook_id": 113, "schedule": {"enabled": False}}, "t")

    assert params == {"playbook_id": 113, "schedule_config": {"enabled": False}}
    assert undeclared_params_refusal("platform_update_playbook", action, params, "t") is None
