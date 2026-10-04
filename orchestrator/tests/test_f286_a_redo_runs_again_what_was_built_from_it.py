"""F286 (night 8) — a step sent back runs again what was built from it, the synthesis included.

#0282.1 and #0282.2 were sent back while the mission's synthesis was running: it finished
from the drafts the owner had rejected and became the mission card's result ("Dear Club
Member… delightful…!"). #0267.1 was redone, and the reminder built from it pasted the
old "Dear Valued Cafe Partner" letter back in. #0446.4's summary quoted drafts the owner
had sent back.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from core.models.orchestration_enums import RunState, TaskState

REJECTED = "Dear Club Member, what a delightful December!"
NOTE = "Start with Hi, not Dear Club Member. No exclamation marks."
REMINDER_NOTE = "Shorter: two lines, then my name."


def _owner(ws):
    return NS(workspace_id=uuid.UUID(str(ws)), user_id="2", auth_type="anonymous", user=NS(id="2"))


def _body(payload):
    async def _json():
        return payload
    return NS(json=_json)


@pytest.fixture
def mission(db_session, seed_workspace):
    """The letter (1) and price list (2) passed; the synthesis of both (3) is running; the
    reminder built from the letter (4) passed; the order form (5) builds on nothing; a
    summary (6) declares no inputs."""
    from core.models.orchestration import OrchestrationRun, OrchestrationTask, OrchestrationTaskDependency
    from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task

    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal="The Christmas club letter", state=RunState.RUNNING.value,
                           created_by="user_test", config={}, max_concurrent=3)
    db_session.add(run)
    db_session.flush()
    create_mission_board_task(db_session, run)
    plan = [("Draft the club letter", "llm_generation", TaskState.VERIFIED, REJECTED),
            ("Price list", "llm_generation", TaskState.VERIFIED, "Five coffees, per kilo."),
            ("Synthesize", "synthesis", TaskState.RUNNING, None),
            ("Reminder email", "llm_generation", TaskState.VERIFIED, f"Reminder: {REJECTED}"),
            ("Order form", "llm_generation", TaskState.VERIFIED, "The order form."),
            ("Summary", "synthesis", TaskState.VERIFIED, f"Summary: {REJECTED}")]
    steps, cards = [], []
    for n, (title, kind, state, output) in enumerate(plan, start=1):
        step = OrchestrationTask(run_id=run.id, title=title, description="Do it.", sequence_number=n, task_type=kind,
                                 state=state.value, max_retries=3, attempt_number=0, output=output)
        db_session.add(step)
        db_session.flush()
        steps.append(step)
        cards.append(create_task_board_task(db_session, run, step))
        cards[-1].result = output                                                   # what the card shows
    for task, builds_on in ((2, 0), (2, 1), (3, 0)):
        db_session.add(OrchestrationTaskDependency(task_id=steps[task].id, depends_on_task_id=steps[builds_on].id))
    steps[3].input_context = {"upstream_results": [{"title": "Draft the club letter", "output": REJECTED}],
                              "field_digest": REJECTED, "verification_requeues": 1}
    db_session.flush()
    return NS(db=db_session, ws=ws, run=run, steps=steps, cards=cards)


def _reject(mission, n, note=NOTE):
    from api.board_tasks import reject_task

    asyncio.run(reject_task(mission.cards[n].id, _body({"feedback": note}), ctx=_owner(mission.ws), db=mission.db))
    for row in (*mission.steps, *mission.cards):
        mission.db.refresh(row)


def test_what_was_built_from_a_step_sent_back_waits_for_its_redo(mission):
    letter, prices, synthesis, reminder, order_form, summary = mission.steps

    _reject(mission, 0)

    assert letter.state == TaskState.RETRYING.value
    assert reminder.state == TaskState.PENDING.value and reminder.output is None    # night: pasted the old letter
    assert summary.state == TaskState.PENDING.value                                 # merges the mission's steps
    for key in ("upstream_results", "field_digest", "verification_requeues"):
        assert key not in (reminder.input_context or {})                            # never told the old letter again
    reminder_card = mission.cards[3]
    assert (reminder_card.status, reminder_card.result) == ("inbox", None)
    assert reminder_card.planning_data["previous_runs"][-1]["result"] == f"Reminder: {REJECTED}"
    assert synthesis.state == TaskState.RUNNING.value and synthesis.input_context["rerun_after_redo"]
    assert (prices.state, order_form.state) == (TaskState.VERIFIED.value, TaskState.VERIFIED.value)


def test_a_synthesis_running_when_its_input_was_sent_back_runs_again_after_the_redo(mission):
    from modules.coordination.dispatcher import MissionDispatcher
    from services.orchestration_deps import DependencyResolver

    letter, _prices, synthesis, *_ = mission.steps
    _reject(mission, 0)

    MissionDispatcher.record_task_completion(mission.db, synthesis, {"status": "success", "result": REJECTED})

    assert synthesis.state == TaskState.PENDING.value and synthesis.output is None  # night: became the result
    assert synthesis not in DependencyResolver.get_ready_tasks(mission.db, mission.run.id)
    letter.output, letter.state = "Hi, here is December's coffee.", TaskState.VERIFIED.value
    mission.db.flush()
    assert synthesis in DependencyResolver.get_ready_tasks(mission.db, mission.run.id)  # runs from the redo


def test_a_step_already_sent_back_waits_for_its_inputs_redo_with_the_owners_words(mission):
    letter, _prices, _synthesis, reminder, *_ = mission.steps

    _reject(mission, 3, note=REMINDER_NOTE)
    assert reminder.state == TaskState.RETRYING.value
    _reject(mission, 0)

    assert letter.state == TaskState.RETRYING.value
    assert reminder.state == TaskState.PENDING.value                                # waits for the letter's redo
    assert REMINDER_NOTE in reminder.input_context["verification_feedback"]["reasoning"]
    assert reminder.input_context["previous_output"] == f"Reminder: {REJECTED}"
    assert mission.cards[3].status == "assigned"                                    # its redo is still to come


def test_a_redo_whose_input_is_being_redone_waits_for_it(mission):
    letter, _prices, _synthesis, reminder, *_ = mission.steps
    letter.state = TaskState.RETRYING.value                                         # the letter is being redone
    mission.db.flush()

    _reject(mission, 3, note=REMINDER_NOTE)

    assert reminder.state == TaskState.PENDING.value                                # night: redone from the old letter
    assert REMINDER_NOTE in reminder.input_context["verification_feedback"]["reasoning"]
