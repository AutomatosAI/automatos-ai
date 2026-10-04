"""F267 (night 8) — while a step of a mission that checks each step is redone, no other starts.

Night 8: the owner told #0383 "Show me each step when it's done and don't move on
until I've said it's right", and sent #0383.2 back. #0383.1, queued since the plan
was approved, started while #0383.2 waited for its redo: a step being redone
(RETRYING) did not hold the others, and the queued step was first in line.

Now a step being redone, or one waiting for the owner's check, holds every other
step; the redo itself goes first. A mark of waiting for the owner left on a step
that waits to run again holds nothing.
"""
from __future__ import annotations

from core.models import Agent
from core.models.orchestration_enums import TaskState
from modules.coordination.agent_matcher import AgentMatcher, MatchResult
from modules.coordination.dispatcher import MissionDispatcher
from modules.coordination.one_step_at_a_time import (
    ONE_AT_A_TIME,
    a_step_is_busy,
    another_step_goes_first,
    waits_its_turn,
)
from modules.coordination.owner_checks import WAITING_KEY
from tests.test_f267_f268_a_redo_runs_at_once_and_a_retried_mission_completes import mission as _f267_mission

mission = _f267_mission  # a mission that checks each step, its card and three steps, on the real schema


def _set(mission, **states_and_contexts):
    """Each named step (first, second, third) to a state, or (state, input_context)."""
    for name, value in states_and_contexts.items():
        step = mission.steps[("first", "second", "third").index(name)]
        state, context = value if isinstance(value, tuple) else (value, step.input_context)
        step.state, step.input_context = state.value, context
    mission.db.flush()


def test_a_step_sent_back_holds_every_other_step_but_its_own_redo(mission):
    first, second, third = mission.steps
    _set(mission, first=TaskState.QUEUED, second=TaskState.RETRYING)

    assert another_step_goes_first(mission.db, mission.run.id, first.id)       # night 8: #0383.1 started
    assert another_step_goes_first(mission.db, mission.run.id, third.id)
    assert not another_step_goes_first(mission.db, mission.run.id, second.id)  # the redo goes


def test_the_dispatcher_starts_the_redo_and_not_the_step_queued_before_it(mission, monkeypatch):
    first, second, _ = mission.steps
    writer = Agent(name="Content Creator", agent_type="chatbot", description="Writes the shop's words.",
                   status="active", configuration={}, model_config=None, workspace_id=mission.run.workspace_id,
                   created_by="test", owner_type="workspace", owner_id=str(mission.run.workspace_id))
    mission.db.add(writer)
    _set(mission, first=TaskState.QUEUED, second=TaskState.RETRYING)
    picked = MatchResult(agent_id=writer.id, agent_name=writer.name, total_score=0.9, tool_coverage=1.0,
                         skill_match=1.0, model_fit=1.0, availability=1.0, history=0.5, reason="the writer")
    monkeypatch.setattr(AgentMatcher, "compute_semantic_signals_sync", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(AgentMatcher, "match", staticmethod(lambda **kwargs: picked))
    monkeypatch.setattr("services.blueprint_validator.check_authority", lambda db, ws, agent_id: (True, []))

    results = MissionDispatcher.dispatch_ready(mission.db, mission.run, [writer])

    assert [(r.task_id, r.dispatched, r.skipped_reason) for r in results] == [
        (first.id, False, ONE_AT_A_TIME), (second.id, True, None)]
    mission.db.refresh(first)
    assert first.state == TaskState.QUEUED.value                                # it waits its turn
    assert (second.state, second.assigned_agent_id) == (TaskState.ASSIGNED.value, writer.id)


def test_a_step_waiting_for_the_owners_check_holds_every_other_step(mission):
    first, second, _ = mission.steps
    _set(mission, second=(TaskState.VERIFYING, {WAITING_KEY: True}))

    assert a_step_is_busy(mission.db, mission.run.id)
    assert another_step_goes_first(mission.db, mission.run.id, first.id)


def test_a_waiting_mark_left_on_a_step_that_waits_to_run_holds_nothing(mission):
    """A step sent back, or retried with its mission, may still carry the mark."""
    first, second, _ = mission.steps
    _set(mission, second=(TaskState.PENDING, {WAITING_KEY: True}))

    assert not a_step_is_busy(mission.db, mission.run.id)
    assert not another_step_goes_first(mission.db, mission.run.id, first.id)
    assert not another_step_goes_first(mission.db, mission.run.id, second.id)


def test_any_other_mission_dispatches_each_step_as_before(mission):
    _set(mission, second=TaskState.RETRYING)
    mission.run.config = {}
    mission.db.flush()
    calls = []

    waits_its_turn(lambda db, run, task, agents: calls.append(task.id))(
        mission.db, mission.run, mission.steps[0], [])

    assert calls == [mission.steps[0].id]


def test_the_dispatcher_runs_each_step_through_it():
    assert MissionDispatcher._dispatch_single.__code__ is waits_its_turn(len).__code__
