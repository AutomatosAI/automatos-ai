"""F291 (night 8) — a mission's card waiting for its plan's approval is approved from the board.

Needs you listed the mission as an approval under its card's number, yet neither of
the card's controls gave it: a drag into In progress said "Assign an agent first: a
ticket with no agent cannot be in progress." (11 of 11), and Approve was refused
("This is a mission's card: approve or change its plan on the mission's page…").
#0214's Approve note, "Use our real Thursday delivery day and keep the email short.",
went nowhere. On the real schema.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException

from core.models.orchestration import OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import RunState, TaskState
from services.orchestration_board_bridge import create_mission_board_task
from tests import test_f170_a_mission_waiting_behind_a_session_says_so as f170

narrated = f170.narrated       # F170's quiet narration and lane, a fixture

GOAL = "Welcome email for The Lantern Room"
NOTE = "Plan is fine. Use our real Thursday delivery day and keep the email short."


class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture
def plan(db_session, seed_workspace, narrated):
    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal=GOAL, state=RunState.AWAITING_APPROVAL.value,
                           created_by="user_test", config={})
    db_session.add(run)
    db_session.flush()
    step = OrchestrationTask(run_id=run.id, title="Draft the email", description="Short, signed Gerard.",
                             sequence_number=1, state=TaskState.PENDING.value, state_type="initial")
    db_session.add(step)
    db_session.flush()
    card = create_mission_board_task(db_session, run)
    card.status = "review"                       # how the board shows a plan waiting for the owner
    db_session.flush()
    ctx = NS(workspace_id=ws, user=NS(id="owner@cafe.test", clerk_user_id=None, email="owner@cafe.test"))
    return NS(db=db_session, run=run, step=step, card=card, ctx=ctx)


def _fresh(plan):
    plan.db.expire_all()
    return plan.db.get(OrchestrationRun, plan.run.id), plan.card


def _notes(card):
    return [n["note"] for n in (card.runtime_ref or {}).get("session_notes") or []]


def test_approve_on_the_card_approves_the_plan_and_keeps_the_note_for_every_step(plan):
    from api.board_tasks import approve_task
    from modules.coordination.dispatcher import MissionDispatcher

    answer = asyncio.run(approve_task(plan.card.id, _Req({"note": NOTE}), ctx=plan.ctx, db=plan.db))

    run, card = _fresh(plan)
    assert run.state == RunState.RUNNING.value                         # night 8: 409, the mission never started
    assert answer["success"] is True and answer["mission_id"] == str(run.id)
    assert "Approved the plan" in answer["message"] and "every step" in answer["message"]
    assert card.status == "in_progress"
    assert f"Approved the plan: {NOTE}" in _notes(card)
    prompt = MissionDispatcher.build_task_prompt(plan.db.get(OrchestrationTask, plan.step.id), goal=run.goal)
    assert NOTE in prompt                                              # the steps are told, in the owner's words


@pytest.mark.parametrize("door", ["drag", "PATCH", "run now"])
def test_moving_the_card_into_in_progress_approves_the_plan(plan, door):
    import api.board_tasks as bt

    if door == "run now":
        answer = bt.run_task_now(plan.card.id, ctx=plan.ctx, db=plan.db)
    else:
        route = bt.update_task_status if door == "drag" else bt.update_task
        answer = asyncio.run(route(plan.card.id, _Req({"status": "in_progress"}), ctx=plan.ctx, db=plan.db))
    answer = asyncio.run(answer) if asyncio.iscoroutine(answer) else answer

    run, _card = _fresh(plan)
    assert run.state == RunState.RUNNING.value                         # night 8: "Assign an agent first"
    assert "Assign an agent" not in str(answer) and "Approved the plan" in str(answer)


def test_a_reject_on_the_card_names_the_mission_and_changes_nothing(plan):
    from api.board_tasks import reject_task

    with pytest.raises(HTTPException) as refused:
        asyncio.run(reject_task(plan.card.id, _Req({"feedback": "Wrong day"}), ctx=plan.ctx, db=plan.db))

    run, _card = _fresh(plan)
    assert refused.value.status_code == 409
    assert f"(/missions/{plan.run.id})" in refused.value.detail and GOAL in refused.value.detail
    assert run.state == RunState.AWAITING_APPROVAL.value


def test_a_running_missions_card_is_pointed_to_its_mission_never_to_an_agent(plan):
    import api.board_tasks as bt

    plan.run.state = RunState.RUNNING.value
    plan.card.status = "in_progress"
    plan.db.flush()

    with pytest.raises(HTTPException) as approve:
        asyncio.run(bt.approve_task(plan.card.id, _Req({}), ctx=plan.ctx, db=plan.db))
    plan.card.status = "blocked"
    plan.db.flush()
    with pytest.raises(HTTPException) as drag:
        asyncio.run(bt.update_task_status(plan.card.id, _Req({"status": "in_progress"}), ctx=plan.ctx, db=plan.db))

    for refused in (approve.value, drag.value):
        assert refused.status_code == 409 and f"/missions/{plan.run.id}" in refused.detail
        assert "Assign an agent" not in refused.detail


def test_the_missions_own_approve_takes_the_note_too(plan):
    from api.missions import MissionApproveRequest, approve_plan
    from modules.coordination.owner_note import OWNER_NOTE_KEY

    asyncio.run(approve_plan(mission_id=plan.run.id, body=MissionApproveRequest(note=NOTE), ctx=plan.ctx, db=plan.db))

    run, _card = _fresh(plan)
    assert run.state == RunState.RUNNING.value and run.config[OWNER_NOTE_KEY] == NOTE
