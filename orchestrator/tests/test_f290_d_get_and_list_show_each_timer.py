"""F290 (night 8): reading a playbook shows its timer.

Auto said Tom's Monday Dispatch Checklist (#101) "does not currently have a schedule
configured at all" while its Monday 07:00 timer was on: "The platform_get_playbook
call did not return any schedule_config". The list named neither timers nor steps,
so of two namesakes Auto picked the empty one.
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

    def playbook(name, schedule=None, steps=()):
        made = WorkflowTemplate(template_id=f"f290d-{uuid4().hex[:8]}", name=name, workspace_id=ws,
                                description=name, template_definition={"steps": list(steps)}, steps=list(steps),
                                created_by="f290", schedule_config=schedule)
        db_session.add(made)
        db_session.flush()
        return made

    timed = playbook("Tom's Monday Dispatch Checklist", dict(MONDAY_7), [{"prompt_template": "Pack the boxes."}])
    return NS(db=db_session, ws=ws, timed=timed, untimed=playbook("Weekly Instagram posts"))


def _get(roastery, playbook):
    from modules.tools.discovery.handlers_playbooks import get_playbook

    return asyncio.run(get_playbook(roastery.db, roastery.ws, {"playbook_id": playbook.id}))["playbook"]


def test_get_playbook_shows_the_timer_that_is_on(roastery):
    assert _get(roastery, roastery.timed)["timer"] == {
        "on": True, "when": "Mondays at 07:00 (Europe/London)", "cron": "0 7 * * 1", "timezone": UK}


def test_a_timer_switched_off_is_still_shown_with_its_time(roastery):
    roastery.timed.schedule_config = {**MONDAY_7, "enabled": False}
    roastery.db.flush()

    timer = _get(roastery, roastery.timed)["timer"]
    assert timer["on"] is False and timer["when"] == "Mondays at 07:00 (Europe/London)"


def test_a_playbook_with_no_timer_says_none(roastery):
    assert _get(roastery, roastery.untimed)["timer"] is None


def test_the_list_shows_each_ones_timer_and_steps(roastery):
    from modules.tools.discovery.handlers_playbooks import list_playbooks

    listed = asyncio.run(list_playbooks(roastery.db, roastery.ws, {}))["playbooks"]

    seen = {p["id"]: (p["timer"] and p["timer"]["when"], p["step_count"]) for p in listed}
    assert seen == {roastery.timed.id: ("Mondays at 07:00 (Europe/London)", 1), roastery.untimed.id: (None, 0)}


@pytest.mark.parametrize("cron, said", [
    ("0 9 * * 1-5", "weekdays at 09:00"), ("25 20 * * *", "daily at 20:25"),
    ("0 8 * * 1,4", "Mondays and Thursdays at 08:00"), ("0 9 * * 6,0", "weekends at 09:00"),
    ("*/15 * * * *", "cron '*/15 * * * *'"),
])
def test_a_time_is_said_in_words(cron, said):
    from modules.tools.discovery.cron_when import plain_cron

    assert plain_cron(cron, UK) == f"{said} ({UK})"
