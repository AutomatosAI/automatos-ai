"""F286 (night 8): mission results built from the wrong pieces.

- #0400 needed Gmail, which was not connected. Its steps answered "Now let me inject my
  findings…:" and "This information is still missing.", both passed the check, and the
  mission "completed" with that sentence as its result. An answer that is a note, or
  says it could not do the work, now fails the mission's check.
- #0433 asked for "a reusable welcome template with gaps", and its steps failed the
  check for the gaps. A placeholder the step's brief (or the mission's goal) asked for
  is not a slot left in.
- #0383.3, the summary step, could not see its own mission's approved steps; its redo
  had only its last answer. #0446.4 quoted the drafts the owner had sent back. A step
  that pulls the mission together gets the approved results of the steps before it,
  its redo gets them again, and its field digest points to them instead of carrying
  the field's copies.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from core.models.orchestration_enums import RunState, TaskState

NOTE = "Now let me inject my findings into the mission field and provide the complete list with draft replies:"
MISSING = 'I need the list of club members who paused from "task_1". This information is still missing.'
CANNOT_SEE = ("I cannot locate the specific approved content from cards #0383.1 and #0383.2 in the workspace "
              "documents. Could you please provide the exact table and line you approved?")
TEMPLATE = "Hi [Owner first name],\n\nWelcome to Harbourline. Your first delivery comes on [Delivery day].\n\nGerard"
APPROVED_LINE = "Blackcurrant, lime and brown sugar, from the Gakenke washing station at 1,850 metres."
REJECTED_LINE = "Experience a truly captivating cup that offers a vibrant journey."
MARGIN = "| Price ex VAT | £9.50 |\n| Margin | 67.47% |"


def _check(output):
    from modules.coordination.deterministic_checks import DeterministicChecker

    return DeterministicChecker().check(output, None)


def test_a_note_or_an_answer_that_could_not_do_the_work_fails_the_check():
    from modules.coordination.non_answers import A_NOTE, CANNOT

    for answer, kind in ((NOTE, A_NOTE), (MISSING, CANNOT), (CANNOT_SEE, CANNOT),
                         ("The Gmail integration is not connected, so I could not read the replies.", CANNOT)):
        result = _check(answer)
        assert (result.passed, result.short_circuited) == (False, True), answer
        assert result.failures[0].description.startswith(kind), answer
    assert _check(CANNOT_SEE).failures[0].description == CANNOT + (
        '"I cannot locate the specific approved content from cards #0383.1 and #0383.2 in the workspace documents."')


def test_findings_drafts_and_long_work_are_the_work():
    long_work = "Here is the reconciliation, café by café. " + "Lantern Yard paid £120.40 on time. " * 25 + (
        "I don't have access to the bank feed, so the October figures come from the invoice sheet.")
    for answer in ("I couldn't find any pause requests in October.",
                   "Hi Ruth,\n\nI couldn't find your order HL-1047 on our van list, so Tom is checking.\n\nGerard",
                   "Bright, sweet and floral…", "Gmail is connected: the five replies are below.", long_work):
        assert _check(answer).passed is True, answer


def test_a_template_the_brief_asks_for_keeps_its_gaps():
    from core.services.placeholders import UNFINISHED
    from modules.coordination.verification import VERDICT_FAIL, VERDICT_PASS, VerificationResult, VerificationService

    svc = VerificationService()
    svc._run_llm_judge = AsyncMock(return_value=VerificationResult(verdict=VERDICT_PASS, reasoning="fine"))
    asked = asyncio.run(svc.verify_task("Draft the welcome email",
                                        "A reusable welcome template with gaps for the owner's name", TEMPLATE, None))
    plain = asyncio.run(svc.verify_task("Draft the welcome email for Priya", "Welcome Priya to Harbourline.",
                                        TEMPLATE, None))

    assert asked.verdict == VERDICT_PASS                                    # night 8: failed for its gaps
    assert plain.verdict == VERDICT_FAIL and plain.deterministic_failures[0].startswith(UNFINISHED)


def test_a_goal_asking_for_gaps_lets_them_stand_and_one_naming_a_template_does_not():
    from modules.coordination.what_was_asked import asking

    with asking(goal="A reusable welcome template with gaps, from the board's mission call"):
        assert _check(TEMPLATE).passed is True
    with asking(goal="Use our welcome template for Priya at Larder & Loaf"):
        assert _check(TEMPLATE).passed is False


@pytest.fixture
def mission(db_session, seed_workspace):
    """#0383's shape: the margin and the shop line, both approved (the line after a
    send-back), and a summary step that declares no steps it builds on."""
    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task

    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal="The Burundi Kayanza in the shop: margin and a front-page line",
                           state=RunState.RUNNING.value, created_by="user_test", config={})
    db_session.add(run)
    db_session.flush()
    create_mission_board_task(db_session, run)
    rows = (("Work out the margin", "llm_generation", TaskState.VERIFIED, MARGIN),
            ("Write the front-page line", "llm_generation", TaskState.VERIFIED, APPROVED_LINE),
            ("Synthesize the margin and the line", "synthesis", TaskState.RUNNING, None))
    steps = []
    for n, (title, kind, state, output) in enumerate(rows, start=1):
        step = OrchestrationTask(run_id=run.id, title=title, description="Do it.", sequence_number=n,
                                 task_type=kind, state=state.value, state_type="active", output=output,
                                 input_context={})
        db_session.add(step)
        db_session.flush()
        create_task_board_task(db_session, run, step)
        steps.append(step)
    return NS(db=db_session, run=run, steps=steps)


def test_a_summary_gets_its_missions_approved_steps(mission):
    from services.coordinator_service import CoordinatorService

    margin, line, summary = mission.steps
    results = CoordinatorService._collect_upstream_outputs(mission.db, summary)

    assert [r["output"] for r in results] == [MARGIN, APPROVED_LINE]       # night 8: nothing reached it
    assert results[0]["title"].endswith("Work out the margin") and results[0]["title"].startswith("#")


def test_a_step_sent_back_is_not_built_on_until_it_is_approved_again(mission):
    from services.coordinator_service import CoordinatorService

    margin, line, summary = mission.steps
    line.state, line.output = TaskState.RETRYING.value, REJECTED_LINE         # sent back, being redone
    mission.db.flush()

    outputs = [r["output"] for r in CoordinatorService._collect_upstream_outputs(mission.db, summary)]
    assert outputs == [MARGIN] and REJECTED_LINE not in outputs


def test_a_summary_steps_prompt_carries_the_approved_steps(mission):
    from modules.coordination.dispatcher import MissionDispatcher
    from modules.coordination.step_inputs import EARLIER_HEADING

    summary = mission.steps[2]
    summary.task_type, summary.title = "llm_generation", "Put the margin and the line together"
    mission.db.flush()

    prompt = MissionDispatcher.build_task_prompt(summary, goal=mission.run.goal)
    assert EARLIER_HEADING in prompt and MARGIN in prompt and APPROVED_LINE in prompt


def test_a_summarys_redo_is_given_the_approved_steps_again(mission):
    from modules.coordination.step_inputs import APPROVED_HEADING, a_summary_keeps_its_approved_inputs
    from services.coordinator_service import CoordinatorService

    summary = mission.steps[2]
    summary.input_context = {"previous_output": "I have saved the document as burundi_kayanza_analysis.md",
                             "verification_feedback": {"reasoning": "Put the two pieces on this card"}}
    mission.db.flush()

    async def prepare(self, db, run, task, agent_id):
        return {"prompt": "# Revision Request: Synthesize the margin and the line"}

    prep = asyncio.run(a_summary_keeps_its_approved_inputs(prepare)(None, mission.db, mission.run, summary, 1))
    assert APPROVED_HEADING in prep["prompt"] and MARGIN in prep["prompt"] and APPROVED_LINE in prep["prompt"]
    assert CoordinatorService._prepare_task.__wrapped__.__name__ == "_prepare_task"     # the tick runs through it


def test_the_summarys_digest_never_carries_a_sent_back_draft(mission):
    from core.models.orchestration import OrchestrationTaskDependency
    from modules.coordination.step_inputs import PINNED_BELOW, RESULTS_KEY
    from services.coordinator_service import CoordinatorService

    margin, line, summary = mission.steps
    summary.task_type = "llm_generation"
    mission.db.add_all([OrchestrationTaskDependency(task_id=summary.id, depends_on_task_id=margin.id),
                        OrchestrationTaskDependency(task_id=summary.id, depends_on_task_id=line.id)])
    mission.db.flush()

    class Field:                       # the field holds every answer as it finished, the rejected one too
        async def query(self, **kwargs):
            return [{"key": line.title, "value": REJECTED_LINE}, {"key": "Roast days", "value": "Tuesdays"}]

    svc = CoordinatorService.__new__(CoordinatorService)
    svc._field, svc._get_field = Field(), lambda: svc._field
    rows = CoordinatorService._collect_upstream_digest_rows(mission.db, summary)
    asyncio.run(svc._attach_field_digest(mission.db, mission.run, summary, "field-1", 1, upstream_rows=rows))

    digest = summary.input_context["field_digest"]
    assert REJECTED_LINE not in digest and PINNED_BELOW in digest and "Tuesdays" in digest   # night 8: #0446.4
    assert APPROVED_LINE in [r["output"] for r in summary.input_context[RESULTS_KEY]]


def test_a_step_whose_answer_is_not_the_work_fails_once_its_attempts_are_spent(mission):
    from modules.coordination.non_answers import CANNOT
    from modules.coordination.reconciler import MissionReconciler
    from modules.coordination.verification import VerificationService

    step = mission.steps[1]
    step.state, step.output, step.attempt_number = TaskState.VERIFYING.value, MISSING, 2
    step.max_retries, step.input_context = 3, {"verification_requeues": 1}
    mission.db.flush()
    result = asyncio.run(VerificationService().verify_task(step.title, step.description, MISSING, None))

    assert asyncio.run(MissionReconciler._apply_verdict(mission.db, step, result)) is True
    assert step.state == TaskState.FAILED.value and step.failure_detail.startswith(CANNOT)   # night 8: verified
