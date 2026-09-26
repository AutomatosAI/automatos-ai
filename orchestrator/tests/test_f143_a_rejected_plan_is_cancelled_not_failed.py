"""F143 (night 4) — a plan the owner rejects is cancelled, not failed.

Card #967 (run 5cfff76a): the owner rejected the plan, and the run was recorded
as failed. The card went ``failed``, and the watch scored a run that never ran
(0.5/10) and posted "needs a look … terminal state failed" into the owner's chat.
A rejected plan now ends ``cancelled`` with the owner's reason:

- the watch closes a cancelled target without scoring it;
- the card maps cancelled to ``done`` (PRD-204 S4), and its error_message keeps
  "Plan rejected: <reason>", so ``done`` never reads as success (JEV keys watch
  verdicts on that prefix);
- the tool reply says "rejected by the owner", never completed or failed.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

from core.models.core import BoardTask
from core.models.orchestration import OrchestrationRun
from core.models.orchestration_enums import RunState
from core.models.watch_enums import WatchStatus
from modules.tools.discovery.handlers_missions import reject_mission
from services.watch_service import WatchService

REASON = "Use WRITER, COUNTINGHOUSE and OPS as I said, and keep the visits to 20 minutes"


def _mission(db, ws):
    run = OrchestrationRun(workspace_id=ws, goal="Plan the café visits", state=RunState.AWAITING_APPROVAL.value,
                           created_by="user_test")
    db.add(run)
    db.flush()
    card = BoardTask(workspace_id=ws, title="Mission: Plan the café visits", status="review",
                     source_type="orchestration", orchestration_run_id=run.id)
    db.add(card)
    watch = WatchService.create_watch(db, workspace_id=ws, watch_type="mission", target_type="mission",
                                      target_id=str(run.id), title="Watch: Plan the café visits")
    db.flush()
    return run, card, watch


def test_a_rejected_plan_is_cancelled_with_the_owners_reason(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    run, card, watch = _mission(db_session, ws)
    reply = asyncio.run(reject_mission(db_session, ws, {"mission_id": str(run.id), "reason": REASON,
                                                        "_created_by": "user_clerk_owner"}))
    assert reply["success"] is True and reply["state"] == "cancelled"
    assert reply["message"] == (f"Mission {run.id} rejected by the owner: {REASON}. "
                                "It never ran, so it is closed as cancelled.")
    db_session.refresh(run)
    assert (run.state, run.stop_reason, run.stop_detail) == ("cancelled", "human_cancelled", f"Plan rejected: {REASON}")


def test_the_card_is_done_but_says_the_plan_was_rejected(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    run, card, _watch = _mission(db_session, ws)
    asyncio.run(reject_mission(db_session, ws, {"mission_id": str(run.id), "reason": REASON}))
    db_session.refresh(card)
    assert card.status == "done"
    assert card.error_message == f"Plan rejected: {REASON}"
    assert card.completed_at is not None


def test_the_watch_closes_the_rejected_plan_without_scoring_it(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    run, _card, watch = _mission(db_session, ws)
    asyncio.run(reject_mission(db_session, ws, {"mission_id": str(run.id), "reason": REASON}))
    db_session.refresh(watch)
    assert watch.status == WatchStatus.CANCELLED.value and watch.closed_at is not None
    assert watch.final_score is None
    assert "was cancelled" in watch.final_verdict and f"Plan rejected: {REASON}" in watch.final_verdict
