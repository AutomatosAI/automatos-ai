"""F308 (night 9): Auto's brief edit to step 2 ("85 and 95 words") never reached card #1876.

"Mission 27: approve it, with two corrections … the Harbour Log intro is 85 to 95 words,
not 100 to 150" became platform_update_mission_plan {mission_id: 27, task_edits:
[{task_id: "harbour_log_intro", description: "Draft a compelling intro for the Harbour
Log, between 85 and 95 words, …"}]}. The plan took it; card #1876 kept "approximately
100-150 words". Once a mission had started, any edit was refused as "expected
'awaiting_approval'", with no word of what to do.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from core.models.orchestration_enums import RunState, TaskState
from tests.test_f245_cancel_stops_a_mission import _mission

BRIEF = ("Draft a compelling intro for the Harbour Log, between 85 and 95 words, adhering to the brand voice "
         "guidelines.")


@pytest.fixture
def mission(db_session, seed_workspace):
    from types import SimpleNamespace as NS

    from services.ticket_numbers import ticket_number

    def make(state):
        ws = UUID(seed_workspace())
        run, _card, steps = _mission(db_session, ws, state=state)
        ordered = sorted(steps.values(), key=lambda pair: pair[0].sequence_number)
        return NS(db=db_session, ws=ws, run=run, steps=[task for task, _ in ordered],
                  cards=[card for _, card in ordered],
                  numbers=[ticket_number(db_session, card) for _, card in ordered])
    return make


def _edit(m, edits):
    """reads_the_plan_edits over a handler that records what reached it (the coordinator's)."""
    from modules.tools.discovery.plan_edits import reads_the_plan_edits

    reached = []

    async def handler(db, workspace_id, params):
        reached.append(params)
        return {"success": True, "mission_id": params["mission_id"], "message": "plan updated"}

    out = asyncio.run(reads_the_plan_edits(handler)(m.db, m.ws, {"mission_id": str(m.run.id), "task_edits": edits}))
    return out, reached


def _reload(m, n):
    m.db.refresh(m.steps[n])
    m.db.refresh(m.cards[n])
    return m.steps[n], m.cards[n]


def test_a_plan_edit_puts_the_new_brief_on_the_steps_card(mission):
    m = mission(RunState.AWAITING_APPROVAL)

    out, reached = _edit(m, [{"task_id": m.numbers[1], "description": BRIEF}])

    _step, card = _reload(m, 1)
    assert out["success"] is True and len(reached) == 1
    assert card.description == BRIEF                                    # night 9: #1876 kept "100-150 words"


def test_a_started_missions_step_that_has_not_started_takes_the_new_brief(mission):
    m = mission(RunState.RUNNING)                                         # step 1 pending, step 2 queued

    out, reached = _edit(m, [{"task_id": m.numbers[1], "description": BRIEF}])

    step, card = _reload(m, 1)
    assert out["success"] is True and m.numbers[1] in out["message"]     # night 9: "expected 'awaiting_approval'"
    assert reached == []                                                  # never the plan-time edit
    assert step.description == BRIEF and card.description == BRIEF      # the step works from it, the card shows it


def test_a_step_that_has_started_is_refused_naming_its_redo(mission):
    m = mission(RunState.RUNNING)
    running = next(n for n, step in enumerate(m.steps) if step.state == TaskState.RUNNING.value)

    out, _reached = _edit(m, [{"task_id": m.numbers[running], "description": BRIEF},
                              {"task_id": m.numbers[0], "description": BRIEF}])

    step, card = _reload(m, running)
    first, _first_card = _reload(m, 0)
    assert out["success"] is False and "has started (running)" in out["error"]
    assert "platform_reject_mission" in out["error"] and "Nothing was changed" in out["error"]
    assert step.description != BRIEF and card.description != BRIEF and first.description != BRIEF


def test_a_started_steps_agent_is_not_swapped(mission):
    m = mission(RunState.RUNNING)

    out, _reached = _edit(m, [{"task_id": m.numbers[0], "agent_role": "Content Creator"}])

    assert out["success"] is False and "agent can't change" in out["error"]
