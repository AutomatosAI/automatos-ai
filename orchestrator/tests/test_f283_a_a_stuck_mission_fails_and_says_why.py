"""F283 (night 8): a mission never reached "failed", so nothing recovered it.

#0352.2 failed the mission's own check ("It still has placeholders where its content
belongs: [Cafe Name].", attempt 1 of 3) and nothing ran it again. The summary waited
for it, the mission said "running" for 20 minutes, Pause → Resume changed nothing and
Replan wanted a failed mission. Now the step uses its attempts, each a revision with
the check's words; when they are spent and nothing else can move, the mission fails
naming the step and why, the steps that waited are skipped for the failure, Resume
runs them again and Replan is offered. A mission that failed while planning (26
September, no steps) is not "resumed" into running with nothing to run.
"""
from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from core.models.orchestration_enums import RunState, TaskState

PLACEHOLDERS = "It still has placeholders where its content belongs: [Cafe Name]."
EMAIL = "Dear [Cafe Name],\n\nPlease place your Christmas order by Thursday 10 December.\n\nGerard"
STOPPED = re.compile(r"^Step #\d{4}\.2 failed the mission's check: " + re.escape(PLACEHOLDERS) + "$")


@pytest.fixture
def mission(db_session, seed_workspace):
    """#0352's shape: the margin (approved), the email (being checked), the summary
    (waiting on both), each with its card."""
    from core.models.orchestration import OrchestrationRun, OrchestrationTask, OrchestrationTaskDependency
    from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task

    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal="Christmas wholesale: the margin and one email to the cafes",
                           state=RunState.RUNNING.value, created_by="user_test", config={}, max_retries=3)
    db_session.add(run)
    db_session.flush()
    card = create_mission_board_task(db_session, run)
    rows = (("Work out the margin", TaskState.VERIFIED, "| £10.31 | 49.1% |"),
            ("Draft the email to the cafes", TaskState.VERIFYING, EMAIL),
            ("Pull the margin and the email together", TaskState.PENDING, None))
    steps, cards = [], []
    for n, (title, state, output) in enumerate(rows, start=1):
        step = OrchestrationTask(run_id=run.id, title=title, description="Do it.", sequence_number=n,
                                 state=state.value, state_type="active", output=output, max_retries=3,
                                 attempt_number=0, input_context={})
        db_session.add(step)
        db_session.flush()
        steps.append(step)
        cards.append(create_task_board_task(db_session, run, step))
    margin, email, summary = steps
    db_session.add_all([OrchestrationTaskDependency(task_id=summary.id, depends_on_task_id=margin.id),
                        OrchestrationTaskDependency(task_id=summary.id, depends_on_task_id=email.id)])
    db_session.flush()
    return NS(db=db_session, run=run, card=card, margin=margin, email=email, summary=summary, cards=cards)


def _check_says_unfinished(mission, *, attempt):
    """The mission's check finds the email still holds a placeholder, on its ``attempt``."""
    from modules.coordination.reconciler import MissionReconciler
    from modules.coordination.verification import VERDICT_FAIL, VerificationResult

    email = mission.email
    email.state, email.attempt_number = TaskState.VERIFYING.value, attempt
    email.input_context = {"verification_requeues": 1}          # PRD-200's one revision is spent
    mission.db.flush()
    result = VerificationResult(verdict=VERDICT_FAIL, reasoning="Deterministic must_pass check failed",
                                deterministic_passed=False, deterministic_failures=[PLACEHOLDERS])
    return asyncio.run(MissionReconciler._apply_verdict(mission.db, email, result))


def _reconcile(mission, monkeypatch):
    from modules.coordination.reconciler import MissionReconciler

    told = []

    async def notify(db, run):
        told.append(run.state)

    monkeypatch.setattr(MissionReconciler, "_notify_run_failed", staticmethod(notify))
    result = asyncio.run(MissionReconciler.reconcile(mission.db, mission.run))
    return result, told


def test_a_step_the_check_finds_unfinished_is_revised_while_it_has_attempts(mission):
    assert _check_says_unfinished(mission, attempt=1) is True          # night 8: failed at "attempt 1 of 3"

    email = mission.email
    assert (email.state, email.attempt_number) == (TaskState.RETRYING.value, 2)
    assert email.input_context["previous_output"] == EMAIL
    assert email.input_context["verification_feedback"]["failures"] == [PLACEHOLDERS]


def test_when_its_attempts_are_spent_the_step_fails_saying_why(mission):
    _check_says_unfinished(mission, attempt=2)

    email = mission.email
    assert (email.state, email.attempt_number, email.failure_detail) == (TaskState.FAILED.value, 3, PLACEHOLDERS)
    assert email.failure_reason_code == "verification_fail"


def test_the_mission_fails_naming_the_step_once_nothing_can_move(mission, monkeypatch):
    _check_says_unfinished(mission, attempt=2)

    result, told = _reconcile(mission, monkeypatch)

    run = mission.run
    assert (run.state, run.stop_reason) == (RunState.FAILED.value, "dependency_failed")   # night 8: "running"
    assert STOPPED.match(run.stop_detail), run.stop_detail
    assert result.run_new_state == RunState.FAILED.value and told == [RunState.FAILED.value]
    assert (mission.summary.state, mission.summary.failure_reason_code) == (TaskState.SKIPPED.value,
                                                                            "dependency_failed")
    assert mission.margin.state == TaskState.VERIFIED.value
    assert (mission.card.status, mission.card.error_message) == ("failed", run.stop_detail)
    assert (mission.cards[1].status, mission.cards[1].error_message) == ("failed", PLACEHOLDERS)


def test_a_mission_with_a_step_still_working_waits_for_it(mission, monkeypatch):
    from core.models.orchestration import OrchestrationTask

    _check_says_unfinished(mission, attempt=2)
    other = OrchestrationTask(run_id=mission.run.id, title="Price the gift box", description="Do it.",
                              sequence_number=4, state=TaskState.RUNNING.value, state_type="active", max_retries=3)
    mission.db.add(other)
    mission.db.flush()

    _reconcile(mission, monkeypatch)
    assert mission.run.state == RunState.RUNNING.value          # its own work finishes first

    other.state, other.output = TaskState.VERIFIED.value, "£24.00"
    mission.db.flush()
    _reconcile(mission, monkeypatch)
    assert mission.run.state == RunState.FAILED.value and STOPPED.match(mission.run.stop_detail)


def test_the_last_step_failing_fails_its_mission_the_same_way(mission, monkeypatch):
    mission.summary.state, mission.summary.failure_reason_code = TaskState.SKIPPED.value, "replaced_by_replan"
    _check_says_unfinished(mission, attempt=2)        # every live step is done: the end rule decides

    _reconcile(mission, monkeypatch)
    assert mission.run.state == RunState.FAILED.value and STOPPED.match(mission.run.stop_detail)


def _failed(mission, monkeypatch):
    _check_says_unfinished(mission, attempt=2)
    _reconcile(mission, monkeypatch)
    assert mission.run.state == RunState.FAILED.value


def test_resume_runs_the_failed_step_and_the_steps_that_waited_for_it(mission, monkeypatch):
    from services.coordinator_service import CoordinatorService
    from services.orchestration_deps import DependencyResolver

    _failed(mission, monkeypatch)
    CoordinatorService().resume_mission(mission.db, mission.run.id, "user_test")

    assert mission.run.state == RunState.RUNNING.value
    assert (mission.email.state, mission.email.attempt_number) == (TaskState.PENDING.value, 0)
    assert mission.summary.state == TaskState.PENDING.value
    assert DependencyResolver.get_ready_tasks(mission.db, mission.run.id) == [mission.email]
    assert (mission.cards[1].status, mission.cards[1].error_message) == ("inbox", None)
    assert mission.card.status == "in_progress" and mission.card.error_message is None
    notes = (mission.card.runtime_ref or {}).get("session_notes") or []
    assert notes[-1]["note"] == "Resumed: the mission runs its failed steps again."


def test_pause_then_resume_runs_a_failed_step_again(mission):
    from services.coordinator_service import CoordinatorService

    _check_says_unfinished(mission, attempt=2)                # failed; the owner pauses before the next tick
    service = CoordinatorService()
    service.pause_mission(mission.db, mission.run.id, "user_test")
    service.resume_mission(mission.db, mission.run.id, "user_test")

    assert mission.run.state == RunState.RUNNING.value        # night 8: the step stayed failed
    assert (mission.email.state, mission.email.attempt_number) == (TaskState.PENDING.value, 0)


def test_replan_is_offered_and_replaces_what_the_failure_skipped(mission, monkeypatch):
    import services.coordinator_service as coordinator
    from modules.coordination.planner import DecompositionResult, PlannedTask

    _failed(mission, monkeypatch)
    redo = PlannedTask(temp_id="t1", title="Draft the email to all the cafes", description="Hi all, …",
                       agent_role="writer", sequence_number=1, task_type="llm_generation",
                       verification_criteria=[], required_tools=[], dependencies=[])

    async def replan(**kwargs):
        return DecompositionResult(tasks=[redo], dependencies=[], token_estimate=1000)

    monkeypatch.setattr(coordinator.MissionPlanner, "replan", staticmethod(replan))
    run = asyncio.run(coordinator.CoordinatorService().replan_mission(mission.db, mission.run.id, "user_test"))

    assert run.state == RunState.RUNNING.value                 # night 8: "expected 'failed'"
    assert (mission.email.failure_reason_code, mission.summary.failure_reason_code) == (
        "replaced_by_replan", "replaced_by_replan")
    assert mission.card.status == "in_progress"


def test_a_mission_that_failed_while_planning_is_not_resumed_into_nothing(db_session, seed_workspace):
    from core.models.orchestration import OrchestrationRun
    from modules.coordination.mission_retry import NO_STEPS_TO_RESUME
    from modules.tools.discovery.handlers_missions import resume_mission

    run = OrchestrationRun(workspace_id=UUID(seed_workspace()), goal="Find a replacement for Santa Barbara",
                           state=RunState.FAILED.value, created_by="user_test", config={},
                           stop_reason="coordinator_error", stop_detail="Plan validation failed")
    db_session.add(run)
    db_session.flush()

    reply = asyncio.run(resume_mission(db_session, run.workspace_id, {"mission_id": str(run.id)}))

    assert reply == {"success": False, "error": NO_STEPS_TO_RESUME}
    assert run.state == RunState.FAILED.value                  # night 8: "running" with 0 steps for 73 minutes
