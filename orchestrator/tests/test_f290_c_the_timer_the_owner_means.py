"""F290 (night 8): of two playbooks named alike, the timer goes off on the one that has it.

"Weekly Social Posts" (#110, no steps) and "Weekly social posts" (#111, steps and the
weekday-9 timer). "Switch off the timer on my Weekly social posts playbook" was
refused for the shared name and Auto asked the owner for an id; told "only one of
them has a timer", it switched 110 "off" twice and said it was done, while 111 kept
running.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

UK = "Europe/London"
WEEKDAY_9 = {"type": "cron", "cron_expression": "0 9 * * 1-5", "timezone": UK, "enabled": True}
A_STEP = [{"prompt_template": "Draft the week's posts.", "agent_id": None}]


@pytest.fixture
def roastery(db_session, seed_workspace):
    from core.models.core import WorkflowTemplate

    ws = UUID(seed_workspace())

    def playbook(name, *, schedule=None, steps=()):
        made = WorkflowTemplate(template_id=f"f290c-{uuid4().hex[:8]}", name=name, workspace_id=ws,
                                description=name, template_definition={"steps": list(steps)}, steps=list(steps),
                                created_by="f290", schedule_config=schedule)
        db_session.add(made)
        db_session.flush()
        return made

    empty = playbook("Weekly Social Posts")
    staffed = playbook("Weekly social posts", schedule=dict(WEEKDAY_9), steps=A_STEP)
    return NS(db=db_session, ws=ws, empty=empty, staffed=staffed)


def _schedule(roastery, **params):
    from modules.tools.discovery.handlers_playbooks import schedule_playbook

    return asyncio.run(schedule_playbook(roastery.db, roastery.ws, params))


def _update(roastery, **params):
    from modules.tools.discovery.handlers_playbooks import update_playbook

    return asyncio.run(update_playbook(roastery.db, roastery.ws, params))


def _schedules(roastery):
    roastery.db.refresh(roastery.empty)
    roastery.db.refresh(roastery.staffed)
    return roastery.empty.schedule_config, roastery.staffed.schedule_config


def test_off_by_the_shared_name_takes_the_one_whose_timer_is_on(roastery):
    out = _schedule(roastery, playbook_name="Weekly social posts", enabled=False)       # night 8, 06:18

    assert out["success"] is True and out["playbook_id"] == roastery.staffed.id
    assert f"#{roastery.staffed.id} is the one whose timer was on" in out["chosen_because"]
    assert _schedules(roastery) == (None, {**WEEKDAY_9, "enabled": False})


def test_off_by_the_empty_ones_id_is_refused_naming_the_other(roastery):
    out = _schedule(roastery, playbook_id=roastery.empty.id, enabled=False, cron_expression="0 9 * * 1-5")

    assert out["success"] is False
    assert f"#{roastery.staffed.id} 'Weekly social posts' has its timer on (weekdays at 09:00 (Europe/London))" \
        in out["error"]
    assert _schedules(roastery) == (None, WEEKDAY_9)


def test_update_playbook_off_on_the_empty_one_is_refused_too(roastery):
    off = {**WEEKDAY_9, "enabled": False}
    out = _update(roastery, playbook_id=roastery.empty.id, schedule_config=off)          # night 8, 06:19

    assert out["success"] is False and f"#{roastery.staffed.id} 'Weekly social posts'" in out["error"]
    assert _schedules(roastery) == (None, WEEKDAY_9)


def test_an_off_timer_written_by_mistake_on_the_empty_one_does_not_hide_the_live_one(roastery):
    roastery.empty.schedule_config = {**WEEKDAY_9, "enabled": False}
    roastery.db.flush()

    out = _schedule(roastery, playbook_id=roastery.empty.id, enabled=False)

    assert out["success"] is False and f"#{roastery.staffed.id}" in out["error"]
    assert _schedules(roastery)[1] == WEEKDAY_9


def test_a_new_timer_by_the_shared_name_goes_on_the_one_with_steps(roastery):
    roastery.staffed.schedule_config = None
    roastery.db.flush()

    out = _schedule(roastery, playbook_name="Weekly social posts", cron_expression="0 9 * * 1-5", timezone=UK)

    assert out["success"] is True and out["playbook_id"] == roastery.staffed.id
    assert _schedules(roastery) == (None, WEEKDAY_9)


def test_a_new_timer_on_the_empty_ones_id_is_refused_naming_the_other(roastery):
    out = _schedule(roastery, playbook_id=roastery.empty.id, cron_expression="0 9 * * 1-5", timezone=UK)

    assert out["success"] is False and "has no steps" in out["error"]
    assert f"#{roastery.staffed.id} 'Weekly social posts' has 1 step(s)" in out["error"]
    assert _schedules(roastery)[0] is None
