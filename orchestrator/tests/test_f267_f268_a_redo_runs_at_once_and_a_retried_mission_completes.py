"""F267 and F268 (night 7b): a mission's redo, its retry and its end.

- F267: #0188 checked each step, but steps with no dependencies started side by side,
  so two waited in Review at once. #0188.2, sent back, did not run again for 2 min 10 s
  (a mission stays paused while any step waits for the owner), and meanwhile its card
  showed In progress with no start time and the rejected draft.
- F268: #0176, resumed after failing on night 7, ran and passed its four remaining steps
  and still ended "failed: 8 tasks skipped due to upstream failure". #0188 completed
  with nothing on its own card.
"""
from __future__ import annotations

from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from core.models.orchestration_enums import RunState, TaskState


@pytest.fixture
def mission(db_session, seed_workspace):
    """A mission that checks each step, its card, and three steps."""
    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task

    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal="Get the Christmas gift subscription ready",
                           state=RunState.RUNNING.value, created_by="user_test",
                           config={"approval_mode": "step_by_step"}, max_concurrent=3)
    db_session.add(run)
    db_session.flush()
    card = create_mission_board_task(db_session, run)
    steps, cards = [], []
    for n, title in enumerate(("Work out the coffee", "Draft the shop words", "Draft the club email"), start=1):
        step = OrchestrationTask(run_id=run.id, title=title, description="Do it.", sequence_number=n,
                                 state=TaskState.PENDING.value, state_type="initial", max_retries=3, attempt_number=0)
        db_session.add(step)
        db_session.flush()
        steps.append(step)
        cards.append(create_task_board_task(db_session, run, step))
    return NS(db=db_session, run=run, card=card, steps=steps, cards=cards)


def _dispatcher_saw(mission):
    from modules.coordination.one_step_at_a_time import one_step_at_a_time

    calls = []

    def dispatch(db, run, agents):
        calls.append(run.max_concurrent)
        return ["dispatched"]

    return calls, one_step_at_a_time(dispatch)(mission.db, mission.run, [])


def test_a_mission_that_checks_each_step_runs_one_step_at_a_time(mission):
    calls, out = _dispatcher_saw(mission)
    assert out == ["dispatched"] and calls == [1] and mission.run.max_concurrent == 1


def test_nothing_starts_while_a_step_is_busy(mission):
    from modules.coordination.one_step_at_a_time import ONE_AT_A_TIME

    for busy in (TaskState.RUNNING, TaskState.COMPLETED, TaskState.VERIFYING):
        mission.steps[0].state = busy.value
        mission.db.flush()
        calls, out = _dispatcher_saw(mission)
        assert calls == [] and out[0].skipped_reason == ONE_AT_A_TIME, busy


def test_any_other_mission_is_dispatched_as_before(mission):
    mission.run.config = {}
    mission.db.flush()
    calls, _ = _dispatcher_saw(mission)
    assert calls == [3]


def test_the_dispatcher_runs_through_it():
    from modules.coordination.dispatcher import MissionDispatcher

    assert MissionDispatcher.dispatch_ready.__wrapped__.__name__ == "dispatch_ready"


def test_a_step_sent_back_resumes_its_mission_and_its_card_starts_clean(mission):
    from core.services.ticket_reasons import WAITING_FOR_YOUR_CHECK
    from services.run_redo import _redo_mission_step

    step, card = mission.steps[1], mission.cards[1]
    step.state, step.output = TaskState.VERIFYING.value, "Exquisite journey! Order now!"
    step.input_context = {"waiting_for_owner": True}
    card.status, card.result, card.started_at = "review", step.output, None
    mission.run.state, mission.run.stop_detail = RunState.PAUSED.value, f"{WAITING_FOR_YOUR_CHECK}#0188.2"
    mission.db.flush()

    _redo_mission_step(mission.db, card, by="user:owner")

    assert step.state == TaskState.RETRYING.value and mission.run.state == RunState.RUNNING.value
    assert card.result is None and card.started_at is not None                # night 7b: rejected draft, no start


def _end(mission, steps_and_codes):
    from modules.coordination.reconciler import MissionReconciler

    for step, (state, code) in zip(mission.steps, steps_and_codes):
        step.state, step.failure_reason_code = state.value, code
    mission.db.flush()
    return MissionReconciler._advance_run_on_completion(
        db=mission.db, run=mission.run, all_tasks=mission.steps, failed_tasks=[], stalls_detected=0,
        stalls_recovered=0, tasks_failed=0)


def test_steps_a_replan_replaced_count_for_nothing_when_the_mission_ends(mission):
    out = _end(mission, [(TaskState.VERIFIED, None), (TaskState.VERIFIED, None),
                         (TaskState.SKIPPED, "replaced_by_replan")])
    assert out.run_new_state == RunState.VERIFYING.value                        # night 7b: "failed"


def test_a_step_skipped_for_a_failure_still_fails_the_mission(mission):
    out = _end(mission, [(TaskState.VERIFIED, None), (TaskState.VERIFIED, None),
                         (TaskState.SKIPPED, "dependency_failed")])
    assert out.run_new_state == RunState.FAILED.value


def test_a_retry_runs_the_steps_skipped_for_the_failure_too(mission):
    from core.models.orchestration_enums import ActorType
    from modules.coordination.mission_retry import retry_failed
    from services.orchestration_state import transition_run

    failed, skipped, replaced = mission.steps
    failed.state, failed.state_type = TaskState.FAILED.value, "terminal"
    skipped.state, skipped.failure_reason_code = TaskState.SKIPPED.value, "dependency_failed"
    replaced.state, replaced.failure_reason_code = TaskState.SKIPPED.value, "replaced_by_replan"
    mission.cards[1].status = "cancelled"
    transition_run(db=mission.db, run=mission.run, new_state=RunState.FAILED, actor_type=ActorType.COORDINATOR,
                   actor_id="reconciler", reason="Tasks failed")
    mission.db.flush()

    again = retry_failed(mission.db, mission.run, "user_test")

    assert again == [failed, skipped] and mission.run.state == RunState.PAUSED.value
    assert (failed.state, skipped.state, replaced.state) == (
        TaskState.PENDING.value, TaskState.PENDING.value, TaskState.SKIPPED.value)
    assert mission.cards[1].status == "inbox"                                 # its card runs again with it


def test_a_completed_missions_card_carries_its_result(mission):
    from services.orchestration_board_bridge import sync_mission_board_status

    for step, output in zip(mission.steps, ("105 kg roasted", "Shop words", "The summary, as approved")):
        step.state, step.output = TaskState.VERIFIED.value, output
    mission.run.state = RunState.COMPLETED.value
    mission.db.flush()

    sync_mission_board_status(mission.db, mission.run)
    assert mission.card.status == "done" and mission.card.result == "The summary, as approved"


def test_a_missions_card_that_says_something_keeps_it(mission):
    from services.orchestration_board_bridge import sync_mission_board_status

    mission.card.result = "The owner's own words"
    mission.steps[2].state, mission.steps[2].output = TaskState.VERIFIED.value, "The summary"
    mission.run.state = RunState.COMPLETED.value
    mission.db.flush()

    sync_mission_board_status(mission.db, mission.run)
    assert mission.card.result == "The owner's own words"
