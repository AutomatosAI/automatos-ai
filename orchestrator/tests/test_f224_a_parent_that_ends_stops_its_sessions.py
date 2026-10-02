"""F224 (2 Oct, Gerard in manual testing) — when a task, playbook run, mission or
routine is cancelled, fails or dies, every Claude Code session it started stops.

Before, a cancelled mission only wrote a note on its step's card ("the session is
still working"), and a failed one touched nothing: the session kept spending the
subscription and could still act. A ticket dragged out of In Progress lost its
credential, but its host heard nothing until a heartbeat noticed, up to half a
minute later.

Now every one of those writers stops the run through the board's own cancel path
(services/board_cancel.py): the card is cancelled with who and why, its lease and
session credential are gone, and the CLI host's next event batch is told to stop
the session (``control: ["cancel"]``). On the real schema, through the host's own
claim, events, heartbeat and result.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from core.models.orchestration_enums import ActorType, RunState
from services import cli_host_service as svc
from services import coordinator_service as cs
from services.board_cancel import CANCEL_REQUESTED_KEY
from services.orchestration_state import transition_run
from tests.helpers_mission_lane import quiet_board, session_working as _session_working

TOOL_CALL = [{"event": "PreToolUse", "tool_name": "Bash"}]


@pytest.fixture
def quiet(monkeypatch):
    """The board's fan-out is its own suites' business (tests/helpers_mission_lane.py)."""
    return quiet_board(monkeypatch)


def _cancel(db, run):
    cs.CoordinatorService().cancel_mission(db, run.id, "user_test")


def _fail(db, run):
    transition_run(db, run, RunState.FAILED, ActorType.RECONCILER,
                   stop_reason="step_failed", stop_detail="a step failed")


def _host_told_to_stop(db, host, card, ticket) -> bool:
    """Both channels the host listens on: its next event batch, and its heartbeat."""
    events = asyncio.run(svc.record_events(db, host, card.id, TOOL_CALL))
    beat = svc.record_heartbeat(db, host, None, [{"task_id": card.id, "session_id": ticket["session_id"]}])
    return "cancel" in events["control"] and card.id in beat["stale"]


@pytest.mark.parametrize("end_it, reason", [
    (_cancel, "the mission was cancelled"), (_fail, "the mission failed"),
], ids=["cancelled", "failed"])
def test_a_mission_that_ends_without_finishing_stops_its_step_sessions(db_session, seed_workspace, quiet,
                                                                       end_it, reason):
    ws = UUID(seed_workspace())
    run, task, card, host, ticket = _session_working(db_session, ws)

    end_it(db_session, run)
    db_session.commit()
    db_session.refresh(card)

    assert card.status == "cancelled"                       # old: still in_progress, the session working on
    assert card.runtime_ref["cancelled"]["by"] == "the mission"
    assert card.runtime_ref["cancelled"]["reason"] == reason
    assert card.lease_until is None and card.runtime_ref.get(CANCEL_REQUESTED_KEY)
    assert svc.resolve_session_token(db_session, ticket["session_token"]) is None   # it can no longer act
    assert _host_told_to_stop(db_session, host, card, ticket)
    late = asyncio.run(svc.apply_result(db_session, host, card.id, {
        "attempt": ticket["attempt"], "status": "success", "result_text": "Drafted anyway.",
        "usage": {"input_tokens": 1, "output_tokens": 1}}))
    assert late["applied"] is False                          # what it did after the end is not the mission's


def test_a_queued_step_card_is_cancelled_so_the_lane_never_claims_it(db_session, seed_workspace, quiet):
    """A step whose session has not started yet must not start one for a mission
    that is over."""
    ws = UUID(seed_workspace())
    run, task, card, host, ticket = _session_working(db_session, ws)
    card.status = "assigned"                                 # queued, no session yet
    db_session.commit()

    _cancel(db_session, run)
    db_session.commit()
    db_session.refresh(card)

    assert card.status == "cancelled"
    assert svc.claim_for_host(db_session, host, 1)["tasks"] == []


def test_a_paused_mission_leaves_its_step_sessions_working(db_session, seed_workspace, quiet):
    """Only an end without finishing stops the sessions: a pause is not an end."""
    ws = UUID(seed_workspace())
    run, task, card, host, ticket = _session_working(db_session, ws)

    transition_run(db_session, run, RunState.PAUSED, ActorType.SYSTEM, stop_reason="budget", stop_detail="paused")
    db_session.commit()
    db_session.refresh(card)

    assert card.status == "in_progress" and not card.runtime_ref.get(CANCEL_REQUESTED_KEY)


def test_a_stop_the_database_refuses_never_costs_the_mission_its_cancel(db_session, seed_workspace, quiet,
                                                                       monkeypatch):
    """The stop runs inside the transaction that holds the mission's cancel. A
    statement that fails there must not abort it: each stop has its own savepoint."""
    from sqlalchemy import text

    import services.board_cancel as board_cancel

    ws = UUID(seed_workspace())
    run, task, card, host, ticket = _session_working(db_session, ws)

    def _refused(db, task, **kwargs):
        db.execute(text("SELECT 1 / 0"))                     # the stop's statement fails in Postgres

    monkeypatch.setattr(board_cancel, "stop_ticket_run", _refused)
    _cancel(db_session, run)
    db_session.flush()
    db_session.refresh(run)

    assert run.state == RunState.CANCELLED.value


def test_a_ticket_taken_out_of_in_progress_by_hand_tells_its_host_to_stop(db_session, seed_workspace, quiet):
    """The board's drag or status PATCH (end_session_claim): the host is told at
    its next event batch, not only when a heartbeat notices."""
    from api.board_tasks import end_session_claim

    ws = UUID(seed_workspace())
    run, task, card, host, ticket = _session_working(db_session, ws)

    end_session_claim(card, "in_progress", "review")
    card.status = "review"
    db_session.commit()

    events = asyncio.run(svc.record_events(db_session, host, card.id, TOOL_CALL))
    assert "cancel" in events["control"]                     # old: [] until the heartbeat noticed
    assert svc.resolve_session_token(db_session, ticket["session_token"]) is None


def test_a_ticket_no_host_claimed_has_nothing_to_stop():
    from types import SimpleNamespace

    from api.board_tasks import end_session_claim

    row = SimpleNamespace(lease_until=None, runtime_ref={"provider": "claude"})
    end_session_claim(row, "in_progress", "review")
    assert CANCEL_REQUESTED_KEY not in row.runtime_ref


# ── routines: a scheduled task or a heartbeat switched off ───────────────────

def _agent(db, ws, name="WATCHTOWER", **config):
    from core.models import Agent

    agent = Agent(name=name, agent_type="chatbot", description="", status="active",
                  configuration={"runtime": "cli", "provider": "claude", **config}, model_config=None,
                  workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db.add(agent)
    db.flush()
    return agent


def _ticket(db, ws, agent, source_type, source_id, status="in_progress"):
    from core.models.core import BoardTask

    live = status == "in_progress"
    ticket = BoardTask(workspace_id=ws, title=f"{source_type} · {source_id}", status=status, priority="low",
                       assigned_agent_id=agent.id, source_type=source_type, source_id=source_id,
                       runtime_ref={"host_id": "laptop", svc.SESSION_TOKEN_HASH_KEY: "hash"} if live else {})
    db.add(ticket)
    db.flush()
    return ticket


@pytest.mark.parametrize("status", ["cancelled", "paused"])
def test_a_routine_switched_off_stops_the_firing_in_flight(db_session, seed_workspace, quiet, status):
    from sqlalchemy import text

    from services.board_cancel import ROUTINE_OFF_REASONS
    from services.scheduled_task_service import ScheduledTaskService

    ws = UUID(seed_workspace())
    agent = _agent(db_session, ws)
    routine = db_session.execute(text(
        "INSERT INTO agent_scheduled_tasks (workspace_id, created_by_agent_id, target_agent_id, description, schedule) "
        "VALUES (CAST(:ws AS uuid), :agent, :agent, 'Morning numbers', '0 7 * * *') RETURNING id"),
        {"ws": str(ws), "agent": agent.id}).scalar()
    firing = _ticket(db_session, ws, agent, "scheduled_task", f"task:{routine}:20261002T0700")
    earlier = _ticket(db_session, ws, agent, "scheduled_task", f"task:{routine}:20261001T0700", status="done")
    another = _ticket(db_session, ws, agent, "scheduled_task", f"task:{routine}0:20261002T0700")
    db_session.commit()

    out = asyncio.run(ScheduledTaskService(db_session, ws).update_task_status(routine, status))
    assert out["success"] is True
    for row in (firing, earlier, another):
        db_session.refresh(row)

    assert firing.status == "cancelled"                       # old: in_progress, still spending
    assert firing.runtime_ref["cancelled"] == {"by": "the routine", "reason": ROUTINE_OFF_REASONS[status],
                                               "at": firing.runtime_ref["cancelled"]["at"]}
    assert firing.runtime_ref.get(CANCEL_REQUESTED_KEY) and svc.SESSION_TOKEN_HASH_KEY not in firing.runtime_ref
    assert (earlier.status, another.status) == ("done", "in_progress")   # its past runs, and another routine


def test_a_routine_switched_back_on_touches_nothing(db_session, seed_workspace, quiet):
    from sqlalchemy import text

    from services.scheduled_task_service import ScheduledTaskService

    ws = UUID(seed_workspace())
    agent = _agent(db_session, ws)
    routine = db_session.execute(text(
        "INSERT INTO agent_scheduled_tasks (workspace_id, created_by_agent_id, target_agent_id, description, schedule) "
        "VALUES (CAST(:ws AS uuid), :agent, :agent, 'Morning numbers', '0 7 * * *') RETURNING id"),
        {"ws": str(ws), "agent": agent.id}).scalar()
    firing = _ticket(db_session, ws, agent, "scheduled_task", f"task:{routine}:20261002T0700")
    db_session.commit()

    asyncio.run(ScheduledTaskService(db_session, ws).update_task_status(routine, "active"))
    db_session.refresh(firing)
    assert firing.status == "in_progress"


def test_switching_a_heartbeat_off_stops_its_session(db_session, seed_workspace, quiet):
    from types import SimpleNamespace

    from api.heartbeat import toggle_heartbeat
    from services.board_cancel import HEARTBEAT_OFF_REASON

    ws = UUID(seed_workspace())
    agent = _agent(db_session, ws, heartbeat={"enabled": True, "interval_minutes": 60})
    other = _agent(db_session, ws, name="LEDGER", heartbeat={"enabled": True, "interval_minutes": 60})
    beat = _ticket(db_session, ws, agent, "heartbeat", f"agent:{agent.id}")
    others_beat = _ticket(db_session, ws, other, "heartbeat", f"agent:{other.id}")
    db_session.commit()

    out = asyncio.run(toggle_heartbeat(agent.id, ctx=SimpleNamespace(workspace_id=ws), db=db_session))
    assert out["enabled"] is False
    db_session.refresh(beat)
    db_session.refresh(others_beat)

    assert beat.status == "cancelled"                         # old: in_progress, still spending
    assert beat.runtime_ref["cancelled"]["reason"] == HEARTBEAT_OFF_REASON
    assert beat.runtime_ref.get(CANCEL_REQUESTED_KEY) and svc.SESSION_TOKEN_HASH_KEY not in beat.runtime_ref
    assert others_beat.status == "in_progress"                # another agent's heartbeat runs on
