"""F268 (night 8, again): Resume on the failed #0176 answered "completed / All tasks
verified successfully" with 8 steps still skipped, and its card turned up Cancelled
with no note.

#0176's skipped steps were steps two re-plans had replaced: they are not part of the
plan, so the mission rightly completed on the four steps the owner had approved. But
nothing said so, and the owner had dismissed the failed card on night 7b, so the card
never followed the mission again. Now Resume reopens the card, saying so; steps
skipped for a failure run again; and a completed mission says what ran and what a
re-plan replaced, on the mission and on its card.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from core.models.orchestration_enums import RunState, TaskState

COMPLETED = ("All 2 steps verified. 2 steps of the earlier plan were replaced when it was re-planned "
             "and did not run.")


@pytest.fixture
def mission(db_session, seed_workspace):
    """#0176's shape: two steps a re-plan replaced, an approved step, and a step skipped
    because another failed; the mission failed and the owner dismissed its card."""
    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task

    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal="A tasting card and a club email for the Guji",
                           state=RunState.FAILED.value, created_by="user_test", config={}, replan_count=2,
                           stop_reason="dependency_failed", stop_detail="8 tasks skipped due to upstream failure")
    db_session.add(run)
    db_session.flush()
    card = create_mission_board_task(db_session, run)
    card.status, card.error_message = "cancelled", run.stop_detail
    card.runtime_ref = {"cancelled": {"by": "user:1", "reason": "cancelled on the board"}}
    rows = (("Draft the tasting card", TaskState.SKIPPED, "replaced_by_replan", None),
            ("Draft the club email", TaskState.SKIPPED, "replaced_by_replan", None),
            ("Draft the tasting card again", TaskState.VERIFIED, None, "Blueberry, jasmine, dark chocolate."),
            ("Draft the club email again", TaskState.SKIPPED, "dependency_failed", None))
    steps = []
    for n, (title, state, code, output) in enumerate(rows, start=1):
        step = OrchestrationTask(run_id=run.id, title=title, description="Do it.", sequence_number=n,
                                 state=state.value, state_type="terminal", failure_reason_code=code, output=output,
                                 max_retries=3, attempt_number=0)
        db_session.add(step)
        db_session.flush()
        create_task_board_task(db_session, run, step)
        steps.append(step)
    db_session.flush()
    return NS(db=db_session, run=run, card=card, steps=steps)


def _notes(card):
    return [entry["note"] for entry in (card.runtime_ref or {}).get("session_notes") or []]


def test_resume_runs_the_step_skipped_for_a_failure_and_reopens_the_card(mission):
    from services.coordinator_service import CoordinatorService

    CoordinatorService().resume_mission(mission.db, mission.run.id, "user_test")

    first, second, approved, skipped = mission.steps
    assert mission.run.state == RunState.RUNNING.value
    assert skipped.state == TaskState.PENDING.value                             # night 8: still skipped
    assert (first.state, second.state, approved.state) == (
        TaskState.SKIPPED.value, TaskState.SKIPPED.value, TaskState.VERIFIED.value)
    assert mission.card.status == "in_progress"                                 # night 8: cancelled
    assert mission.card.error_message is None and "cancelled" not in mission.card.runtime_ref
    assert _notes(mission.card)[-1] == "Resumed: the mission runs its failed steps again."


@pytest.fixture
def quiet(monkeypatch):
    """The completion's side errands (consistency check, notifications, memory)."""
    import services.coordinator_service as coordinator

    async def nothing(*args, **kwargs):
        return None

    monkeypatch.setattr(coordinator.CoordinatorService, "_run_consistency_check", nothing)
    monkeypatch.setattr(coordinator, "_dispatch_mission_event", nothing)
    monkeypatch.setattr(coordinator, "_store_mission_memory_safe", nothing)
    return coordinator


def test_a_completed_mission_says_what_ran_and_what_a_replan_replaced(mission, quiet):
    first, second, approved, skipped = mission.steps
    skipped.state, skipped.failure_reason_code, skipped.output = TaskState.VERIFIED.value, None, "Dear club, …"
    mission.card.status = "in_progress"
    mission.run.state = RunState.VERIFYING.value
    mission.db.flush()

    asyncio.run(quiet.CoordinatorService()._complete_verified_run(mission.db, mission.run))

    assert mission.run.state == RunState.COMPLETED.value
    assert mission.run.stop_detail == COMPLETED                      # night 8: "All tasks verified successfully"
    assert mission.card.status == "done"
    assert _notes(mission.card)[-1] == f"The mission completed. {COMPLETED}"
