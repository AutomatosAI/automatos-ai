"""F271 (night 7b, B14, friction 19) — 'none' is not a schedule type: saving it is
refused, and the refusal says how to turn a timer off.

At 20:25 the owner approved #0195, the card playbook 113's timer had just made, and
saved the playbook's schedule as {type: 'none'} to stop the timer. The schedule stayed
{type: 'cron', enabled: true, '25 20 * * *', Europe/London}, so the timer would have
fired again the next evening; only enabled: false turned it off. 'none' had been
refused all along (400 "type must be 'manual', 'cron', or 'trigger'"), but the night's
call threw the response away, and the refusal never said what does turn a timer off,
so the save read as done. A type the scheduler does not know is now a 422 that names
the two ways, enabled: false (the timer keeps its time) and type manual (the playbook
runs only when started), on create and on update. Both ways work, and the refused
save leaves the stored schedule and the scheduler's job as they were.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from tests import test_f116_a_cancelled_run_stops_its_sessions as f116
from tests import test_f132_schedules_fire_and_say_when_they_dont as f132
from tests.test_f116_a_cancelled_run_stops_its_sessions import _recipe

# F116's Postgres and workspace, and F132's worker that holds the scheduler, as fixtures here too.
engine = f116.engine
workspace = f116.workspace
leader = f132.leader

TIMER = {"type": "cron", "cron_expression": "25 20 * * *", "timezone": "Europe/London"}
REFUSAL = (
    "A schedule's type is manual, cron or trigger; there is no 'none', so nothing was saved. "
    "To turn a timer off, save its schedule with \"enabled\": false (it keeps its time for when "
    "you turn it back on), or with \"type\": \"manual\" to run the playbook only when you start it."
)


def _owner(ws):
    return NS(workspace_id=uuid.UUID(ws), user_id="2", user=None)


def _save(ws, new_session, template_id, data):
    from api.workflow_recipes import update_workflow_recipe

    return asyncio.run(update_workflow_recipe(template_id, ctx=_owner(ws), recipe_data=data, db=new_session()))


def _stored(new_session, recipe_id):
    return new_session().execute(
        text("SELECT schedule_config FROM workflow_recipes WHERE id = :i"), {"i": recipe_id}).scalar()


def _job(scheduler_host, recipe_id):
    return scheduler_host._scheduler.get_job(f"playbook_cron_{recipe_id}")


def _timed_playbook(ws, new_session, scheduler_host):
    """Playbook 113 as the night had it: its timer saved through the route, and running."""
    s = new_session()
    playbook = _recipe(s, ws)
    s.commit()
    _save(ws, new_session, playbook.template_id, {"schedule_config": dict(TIMER)})
    assert _stored(new_session, playbook.id) == TIMER and _job(scheduler_host, playbook.id) is not None
    return playbook


def test_a_schedule_of_type_none_is_refused_and_says_how_to_turn_the_timer_off(workspace, new_session, leader):
    playbook = _timed_playbook(workspace, new_session, leader)
    trigger = repr(_job(leader, playbook.id).trigger)

    with pytest.raises(HTTPException) as refused:
        _save(workspace, new_session, playbook.template_id, {"schedule_config": {"type": "none"}})

    assert refused.value.status_code == 422                                   # night: a 400 nobody read
    assert refused.value.detail == REFUSAL
    assert _stored(new_session, playbook.id) == TIMER                         # nothing was saved
    assert repr(_job(leader, playbook.id).trigger) == trigger                 # and the timer is as it was


@pytest.mark.parametrize("off", [{**TIMER, "enabled": False}, {"type": "manual"}], ids=["enabled-false", "manual"])
def test_each_way_the_refusal_names_turns_the_timer_off(workspace, new_session, leader, off):
    playbook = _timed_playbook(workspace, new_session, leader)

    out = _save(workspace, new_session, playbook.template_id, {"schedule_config": off})

    assert out["recipe"]["schedule_config"] == off                            # the reply shows it off
    assert _stored(new_session, playbook.id) == off                           # enabled: false keeps its time
    assert _job(leader, playbook.id) is None                                  # and nothing will fire


def test_a_new_playbook_with_schedule_type_none_is_refused_the_same_way(workspace, new_session):
    from api.workflow_recipes import create_workflow_recipe

    template_id = f"f271-{uuid.uuid4().hex[:8]}"
    with pytest.raises(HTTPException) as refused:
        asyncio.run(create_workflow_recipe(ctx=_owner(workspace), db=new_session(), recipe_data={
            "template_id": template_id, "name": "Weekly Instagram posts", "description": "f271",
            "template_definition": {"steps": []}, "steps": [], "schedule_config": {"type": "none"}}))

    assert (refused.value.status_code, refused.value.detail) == (422, REFUSAL)
    made = new_session().execute(
        text("SELECT COUNT(*) FROM workflow_recipes WHERE template_id = :t"), {"t": template_id}).scalar()
    assert made == 0


@pytest.mark.parametrize("schedule, verdict", [
    (None, (True, None)),
    ({}, (True, None)),
    ({"type": "manual"}, (True, None)),
    ({"type": "cron", "cron_expression": "25 20 * * *"}, (True, None)),
    ("daily", (False, "schedule_config must be an object")),
    ({"cron_expression": "25 20 * * *"}, (False, "schedule_config must have 'type' field")),
    ({"type": "none"}, (False, "type must be 'manual', 'cron', or 'trigger'")),
    ({"type": "cron"}, (False, "cron type requires cron_expression field")),
    ({"type": "trigger"}, (False, "trigger type requires trigger_config field")),
])
def test_the_models_own_check_reads_the_same_rules_as_before(schedule, verdict):
    """The seeds check their playbooks with the model's method; it answers as it did."""
    from core.models.core import WorkflowTemplate

    assert WorkflowTemplate(schedule_config=schedule).validate_schedule_config() == verdict
