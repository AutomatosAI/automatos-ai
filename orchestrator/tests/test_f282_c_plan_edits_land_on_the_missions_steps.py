"""F261 and F282 (night 8): Auto's plan edits land on the mission's real steps.

"I've updated the mission plan" came five times with nothing changed: the edits named
steps that aren't in the plan (a made-up id, "email_draft", "0410-1", 3), and the plan
skipped them without a word. "Switch on the check for each step" came as approval gates
on each step and as {"plan_updates": {"steps.*.approval_required": true}}, which the
mission never reads; #0410 and #0454 ran start to finish unseen.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from core.models.orchestration_enums import RunState
from tests.test_f245_cancel_stops_a_mission import _mission


@pytest.fixture
def plan(db_session, seed_workspace):
    from types import SimpleNamespace as NS

    from services.ticket_numbers import ticket_number

    ws = UUID(seed_workspace())
    run, card, steps = _mission(db_session, ws, state=RunState.AWAITING_APPROVAL)
    ordered = sorted(steps.values(), key=lambda pair: pair[0].sequence_number)
    return NS(db=db_session, ws=ws, run=run, steps=[task for task, _ in ordered],
              numbers=[ticket_number(db_session, step_card) for _, step_card in ordered])


def _edit(plan, **params):
    """The decorator over a handler that records what reached it."""
    from modules.tools.discovery.plan_edits import reads_the_plan_edits

    reached = []

    async def handler(db, workspace_id, params):
        reached.append(params)
        return {"success": True, "mission_id": params["mission_id"], "message": "plan updated"}

    out = asyncio.run(reads_the_plan_edits(handler)(plan.db, plan.ws, {"mission_id": str(plan.run.id), **params}))
    return out, reached


def test_steps_named_by_card_number_place_or_auto_style_reach_the_plan_by_id(plan):
    step_two = plan.numbers[1].lstrip("#").replace(".", "-")          # Auto's "0352-2" for #0352.2
    out, reached = _edit(plan, task_edits=[
        {"task_id": plan.numbers[0], "changes": {"description": "Last order Thursday 10 December."}},
        {"task_id": step_two, "assigned_agent_name": "Shopify Operations Manager"},
        {"task_id": 3, "title": "Email the cafés"},
    ])

    assert out["success"] is True
    assert reached[0]["task_edits"] == [
        {"task_id": str(plan.steps[0].id), "description": "Last order Thursday 10 December."},
        {"task_id": str(plan.steps[1].id), "agent_role": "Shopify Operations Manager"},
        {"task_id": str(plan.steps[2].id), "title": "Email the cafés"},
    ]


def test_an_edit_naming_no_step_changes_nothing_and_lists_the_steps(plan):
    out, reached = _edit(plan, task_edits=[{"task_id": "4b6110f2-e25f-4638-a28a-77e8006d9a93",
                                            "description": "Thursday 10 December."}])

    assert out["success"] is False and reached == []
    assert "No step" in out["error"] and plan.numbers[0] in out["error"]


@pytest.mark.parametrize("asked", [
    {"task_edits": [{"task_id": "0410-1", "add_approval_gate": True}, {"task_id": "0410-2", "add_approval_gate": True}]},
    {"plan_updates": {"steps.*.approval_required": True}},
    {"check_each_step": True},
])
def test_asking_each_step_to_wait_switches_on_the_missions_check(plan, asked):
    from modules.tools.discovery.plan_edits import CHECKS_ON

    out, reached = _edit(plan, **asked)

    plan.db.refresh(plan.run)
    assert out == {"success": True, "mission_id": str(plan.run.id), "state": plan.run.state, "message": CHECKS_ON}
    assert plan.run.config["check_each_step"] is True and reached == []


def test_a_finished_mission_has_no_steps_left_to_check(plan):
    from modules.tools.discovery.plan_edits import CHECKS_TOO_LATE

    plan.run.state = RunState.COMPLETED.value
    plan.db.flush()
    out, _ = _edit(plan, check_each_step=True)

    assert out == {"success": False, "error": CHECKS_TOO_LATE}
    assert "check_each_step" not in (plan.run.config or {})


def test_the_tool_takes_check_each_step():
    from modules.tools.discovery import get_action_registry

    action = get_action_registry().get("platform_update_mission_plan")
    assert action.parameters["properties"]["check_each_step"]["type"] == "boolean"
    assert action.parameters["required"] == ["mission_id"]
    assert "check_each_step: true" in action.misplaced["plan_updates"]
