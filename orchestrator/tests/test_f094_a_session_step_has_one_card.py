"""F094 (night 5, persona) — a mission step run by a Claude Code agent has one board card.

Night 5's step 2fa5467f (mission 3805978e, NEWSROOM, "Synthesize Christmas Box Offer
Details") had four cards. Dispatch's card #964 said done while #980, the ticket the
step's Claude Code session worked, sat in review; each re-run filed another ticket
(#982, #985). The session lane now claims the step's card instead of filing one
beside it, and from then on it alone writes the card's status. On the real schema.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from core.models import Agent
from core.models.core import BoardTask
from core.models.orchestration import OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import RunState, TaskState
from services import cli_ticket_lane as lane
from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task, sync_board_status

TITLE = "Synthesize Christmas Box Offer Details"


@pytest.fixture
def quiet(monkeypatch):
    """A host is online; the board notices go nowhere."""
    monkeypatch.setattr(lane, "host_online", lambda db, ws: True)
    monkeypatch.setattr(lane, "no_cli_host_reason_for", lambda db, ws, cli: None)
    monkeypatch.setattr(lane, "_notify", lambda *args, **kwargs: None)


def _agent(db, ws, name, runtime):
    agent = Agent(name=name, agent_type="chatbot", description="", status="active", configuration={"runtime": runtime},
                  model_config=None, workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db.add(agent)
    db.flush()
    return agent


def _step(db, ws, *, runtime="cli", card=True):
    agent = _agent(db, ws, "NEWSROOM", runtime)
    run = OrchestrationRun(workspace_id=ws, goal="A Christmas box offer for the cafés", state=RunState.RUNNING.value,
                           created_by="user_test", config={})
    db.add(run)
    db.flush()
    task = OrchestrationTask(run_id=run.id, title=TITLE, description="Synthesize the offer.", sequence_number=2,
                             state=TaskState.RUNNING.value, state_type="active", assigned_agent_id=agent.id)
    db.add(task)
    db.flush()
    create_mission_board_task(db, run)
    if card:
        create_task_board_task(db, run, task)      # dispatch's card (#964)
    return run, task, agent


def _run_the_step(db, run, task, agent):
    """What the coordinator's _run_cli_ticket files, through run_cli_ticket_and_wait."""
    return lane.file_cli_ticket(db, workspace_id=run.workspace_id, agent_id=agent.id, title=TITLE,
                                prompt=f"Work on: {TITLE}", source_type="mission",
                                source_id=f"mission:{run.id}:{task.id}", tags=["mission"],
                                orchestration_run_id=run.id, orchestration_task_id=task.id)


def _session_ends(db, ticket, *, status="review", result="The offer, drafted."):
    ticket.status, ticket.result, ticket.completed_at = status, result, datetime.now(timezone.utc)
    db.flush()


def _cards(db, task):
    return db.query(BoardTask).filter(BoardTask.orchestration_task_id == task.id).order_by(BoardTask.id).all()


def test_the_session_works_the_steps_own_card_and_its_status_stands(db_session, seed_workspace, quiet):
    ws = UUID(seed_workspace())
    run, task, agent = _step(db_session, ws)
    (planned,) = _cards(db_session, task)

    ticket = _run_the_step(db_session, run, task, agent)
    _session_ends(db_session, ticket)                  # the close check held it for review
    task.state = TaskState.VERIFIED.value              # the mission verified the step
    sync_board_status(db_session, task)

    (card,) = _cards(db_session, task)                 # old: #964 done beside #980 in review
    assert (card.id, card.status, card.result) == (planned.id, "review", "The offer, drafted.")
    assert card.parent_task_id == planned.parent_task_id is not None   # still under the mission's card


def test_a_re_run_goes_on_the_same_card_and_keeps_what_the_last_run_ended_as(db_session, seed_workspace, quiet):
    ws = UUID(seed_workspace())
    run, task, agent = _step(db_session, ws)
    first = _run_the_step(db_session, run, task, agent)
    _session_ends(db_session, first, result="Attempt 1: the numbers are missing.")

    again = _run_the_step(db_session, run, task, agent)  # verification failed: the step runs again

    assert again.id == first.id and len(_cards(db_session, task)) == 1   # old: #982 beside #980 and #964
    assert (again.status, again.result, again.completed_at, again.runtime_ref) == ("assigned", None, None, None)
    (previous,) = again.planning_data["previous_runs"]
    assert (previous["status"], previous["note"]) == ("review", "Attempt 1: the numbers are missing.")


def test_a_card_whose_session_is_still_working_is_waited_on_as_it_is(db_session, seed_workspace, quiet):
    ws = UUID(seed_workspace())
    run, task, agent = _step(db_session, ws)
    first = _run_the_step(db_session, run, task, agent)
    first.status = "in_progress"                       # the lane stopped waiting (F164); the session works on
    db_session.flush()

    again = _run_the_step(db_session, run, task, agent)

    assert (again.id, again.status) == (first.id, "in_progress")
    assert len(_cards(db_session, task)) == 1


def test_a_step_moved_to_an_agent_the_lane_does_not_run_follows_the_mission_again(db_session, seed_workspace, quiet):
    ws = UUID(seed_workspace())
    run, task, agent = _step(db_session, ws)
    _session_ends(db_session, _run_the_step(db_session, run, task, agent))
    task.assigned_agent_id = _agent(db_session, ws, "QUILL", "api").id   # re-dispatched to an API agent
    task.state = TaskState.RUNNING.value
    sync_board_status(db_session, task)

    (card,) = _cards(db_session, task)
    assert (card.status, card.assigned_agent_id, card.source_id) == ("in_progress", task.assigned_agent_id, None)


def test_a_step_no_session_runs_keeps_the_missions_status(db_session, seed_workspace, quiet):
    ws = UUID(seed_workspace())
    run, task, agent = _step(db_session, ws, runtime="api")
    task.state = TaskState.VERIFIED.value
    sync_board_status(db_session, task)
    (card,) = _cards(db_session, task)
    assert card.status == "done"


def test_a_step_without_a_card_still_gets_a_ticket(db_session, seed_workspace, quiet):
    ws = UUID(seed_workspace())
    run, task, agent = _step(db_session, ws, card=False)
    ticket = _run_the_step(db_session, run, task, agent)
    (card,) = _cards(db_session, task)
    assert (card.id, card.source_type, card.status) == (ticket.id, "mission", "assigned")
