"""F094 follow-ups (TESTER, 09-26) — a step's card ends on its session's outcome.

Once the session lane claims a mission step's card, the lane alone writes its
status. The code review worried that a card could then stay claimed after its
session ended: the retry and the stall recovery clear the step's agent while the
session may still be working. Whatever the mission does meanwhile — retries the
step, recovers it from a stall, or stops waiting for it — the session's result
lands on the card and the card ends on that outcome. When the mission stopped
waiting, the card also says so, in a note, and its status stays the session's.
On the real schema, through the host's own claim and result.

A mission that ENDS without finishing (cancelled, failed) is F224's: it stops the
step's session and cancels the card (test_f224_a_parent_that_ends_stops_its_sessions).
"""
from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from core.models.orchestration_enums import TaskState
from modules.coordination.dispatcher import MissionDispatcher
from modules.coordination.reconciler import MissionReconciler
from services import cli_host_service as svc
from tests.helpers_mission_lane import quiet_board, session_working as _session_working

STOPPED_WAITING = ("The mission stopped waiting for this step after 60 minutes. The session is still working, "
                   "and its result will land here.")


@pytest.fixture
def quiet(monkeypatch):
    """The board's fan-out (approvals, notices, reports, the Canvas) is its own suites' business."""
    return quiet_board(monkeypatch)


def _retry(db, run, task):
    MissionDispatcher.record_task_completion(db, task, {"status": "error", "error": "the lane lost the database"})


def _stall_recovery(db, run, task):
    task.state = TaskState.STALLED.value
    db.flush()
    asyncio.run(MissionReconciler._recover_stalled_task(db, task))


def _stopped_waiting(db, run, task):
    MissionDispatcher.record_task_completion(db, task, {
        "status": "error", "timed_out": True, "still_running": True, "waited_s": 3600,
        "error": "ticket is still running after 3600 s — the Claude Code session carries on"})


@pytest.mark.parametrize("what_the_mission_did, note", [
    (_retry, None), (_stall_recovery, None), (_stopped_waiting, STOPPED_WAITING),
], ids=["retry", "stall-recovery", "stopped-waiting"])
def test_no_card_the_lane_runs_stays_claimed_after_its_session_ends(db_session, seed_workspace, quiet,
                                                                     what_the_mission_did, note):
    ws = UUID(seed_workspace())
    run, task, card, host, ticket = _session_working(db_session, ws)

    what_the_mission_did(db_session, run, task)            # the mission moves on while the session works
    db_session.refresh(card)
    assert card.status == "in_progress"                    # the card is still the session's

    out = asyncio.run(svc.apply_result(db_session, host, card.id, {
        "attempt": ticket["attempt"], "status": "success", "result_text": "The offer, drafted.",
        "usage": {"input_tokens": 10, "output_tokens": 5}}))
    db_session.refresh(card)

    assert out["applied"] is True
    assert (card.status, card.result) == ("done", "The offer, drafted.")   # it ends on the session's outcome
    notes = [(n["by"], n["note"]) for n in card.runtime_ref.get("session_notes") or []]
    assert notes == ([("the mission", note)] if note else [])
    assert [e["data"]["note"] for e in quiet if e["data"].get("note")] == ([note] if note else [])   # live, too


def test_the_notes_on_a_card_survive_the_claim_that_resumes_its_run(db_session, seed_workspace, quiet):
    """A session that parked on a question is claimed again to resume the same
    run; its card's notes, the mission's verdict among them, are kept."""
    ws = UUID(seed_workspace())
    run, task, card, host, ticket = _session_working(db_session, ws)
    _stopped_waiting(db_session, run, task)
    db_session.refresh(card)
    card.status = "assigned"                               # the answer re-queued it
    db_session.flush()

    (again,) = svc.claim_for_host(db_session, host, 1)["tasks"]
    db_session.refresh(card)

    assert again["task_id"] == card.id
    assert [n["note"] for n in card.runtime_ref["session_notes"]] == [STOPPED_WAITING]
