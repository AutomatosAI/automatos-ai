"""F284 (night 8): a step of a finished mission can be corrected; its mission opens again.

Night 8: the owner sent back steps of missions that had finished (#0214.1, #0236.2,
#0304.2, #0305.2, #0356.2, #0408.1, #0454.2), wrong margins among them: #0408 says
82.35% where it is 62.35%, #0305 53.5% where it is 31.6%. Each was refused with
"…which has finished: a mission only runs its steps while it runs. Re-run the mission
from its page"; the page's Re-run starts a new mission from the goal, Resume takes a
paused mission and Replan a failed one, so the wrong numbers stayed on the mission
cards. #0250.1, sent back 2 s after its mission completed, got through and stayed
"in progress" for good: the redo read the mission unlocked while the coordinator was
completing it, and nothing ran a step of a completed mission.

Now a Reject or a re-brief of a step of a mission that completed or failed opens the
mission again. The step is redone on its card with the owner's words, and what was
built from it runs again (F286, modules/coordination/redo_dependents). The mission runs,
resumed as Resume resumes it: a failed mission's failed steps run again (F247), and a
spent budget is raised (F153). Its card goes back to In progress without the result it
had (its history keeps it), and when the mission completes again the card takes the new
result (F268). Its old completion, result and knowledge ingest go, so the corrected
result is saved again.

The redo decides on the mission's row, locked until the redo commits (``locked_run``).
The mission's end takes the same lock and skips a held mission (the next tick decides),
so a send-back either lands before the mission ends, or waits for the end and opens it
again. The wait is short (``LOCK_WAIT``): the board's Reject runs on the event loop
(F105). A mission still being written after that refuses the redo with nothing changed
("send it back again in a moment"). The redo also writes the mission's row
(``touched``), so a tick that read the mission before the redo can't end it on what it
read: the row's version no longer matches. A cancelled mission's steps never run again,
and a step of a plan waiting for approval is changed in the plan: both are refused in
words that name what exists on the mission's page.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

# A mission whose step is redone as it stands: running, or paused (Resume runs the redo).
RUNS_ITS_STEPS = ("running", "paused")
# A mission that ended, or stopped while completing (its completion failed), opens again.
OPENS_AGAIN = ("completed", "failed", "verifying")
# A mission that finished (or was finishing): its finish time and result go when it opens again.
FINISHED = ("completed", "verifying")
# How long a send-back waits for the mission's row while the coordinator writes it.
LOCK_WAIT = "2s"
# Postgres: a lock not had within lock_timeout, and a deadlock broken by aborting this side.
LOCK_NOT_AVAILABLE = "55P03"
DEADLOCK_DETECTED = "40P01"
# What a finished mission's ingest left on its config (coordinator_service, PRD-179 S2).
INGEST_MARKERS = ("output_document_id", "output_ingest", "output_ingest_failed")
CARD_WHY = "the owner sent back {label}, so the mission runs again"

CANCELLED = ("{label} is a step of the mission \"{goal}\", which was cancelled, so its steps don't run again. "
             "Re-run on the mission's page ({where}) starts it again as a new mission.")
NOT_APPROVED = ("{label} is a step of the mission \"{goal}\", whose plan waits for your approval, so it hasn't "
                "run yet. Change the step in the plan on the mission's page ({where}), then approve the plan.")
NOT_RUNNING = ("{label} is a step of the mission \"{goal}\", which is {doing} right now. Send it back once the "
               "mission is running.")
BUSY = ("{label}'s mission \"{goal}\" is recording its work or finishing right now, so nothing was changed. "
        "Send it back again in a moment.")
DOING = {"pending": "still being planned", "planning": "still being planned", "replanning": "being re-planned",
         "awaiting_human": "waiting for your answer"}


def refusal_for(run: Any, label: str) -> Optional[str]:
    """Why a step of ``run`` can't be redone now, naming what the owner can do instead;
    None when it can: the mission runs its steps, or it ended and opens again."""
    if run.state in RUNS_ITS_STEPS or run.state in OPENS_AGAIN:
        return None
    words = _words(run, label)
    if run.state == "cancelled":
        return CANCELLED.format(**words)
    if run.state == "awaiting_approval":
        return NOT_APPROVED.format(**words)
    return NOT_RUNNING.format(doing=DOING.get(run.state, run.state.replace("_", " ")), **words)


def busy_refusal(run: Any, label: str) -> str:
    """The refusal while another transaction holds ``run`` past ``LOCK_WAIT``."""
    return BUSY.format(**_words(run, label))


def _words(run: Any, label: str) -> Dict[str, str]:
    from services.run_cancel import GOAL_SHOWN_CHARS

    return {"label": label, "goal": (run.goal or "")[:GOAL_SHOWN_CHARS], "where": f"/missions/{run.id}"}


def waits_briefly(db: Any) -> None:
    """Every lock this transaction waits for from here on is given up after ``LOCK_WAIT``."""
    db.execute(text("SELECT set_config('lock_timeout', :wait, true)"), {"wait": LOCK_WAIT})


def lost_the_lock(exc: OperationalError) -> bool:
    """Whether ``exc`` is a lock wait given up (``LOCK_WAIT``) or a deadlock broken on this side."""
    return getattr(getattr(exc, "orig", None), "pgcode", None) in (LOCK_NOT_AVAILABLE, DEADLOCK_DETECTED)


def locked_run(db: Any, run: Any) -> Any:
    """``run`` read again with its row locked until the caller's transaction ends,
    waiting for a tick that is writing it (at most ``LOCK_WAIT``, ``waits_briefly``).
    Raises OperationalError when the wait is given up (``lost_the_lock``)."""
    from core.models.orchestration import OrchestrationRun

    return (db.query(OrchestrationRun)
            .filter(OrchestrationRun.id == run.id, OrchestrationRun.workspace_id == run.workspace_id)
            .with_for_update()
            .populate_existing()
            .first())


def touched(run: Any) -> None:
    """The redo writes the mission's row: a tick that read it before can't end it."""
    run.updated_at = datetime.now(timezone.utc)


def reopen(db: Any, run: Any, *, by: str, label: str) -> bool:
    """Open ``run`` (locked by the caller) again for the redo of its step ``label`` when it
    ended: its card back to In progress without its result, and the mission running as
    Resume runs it. False, changing nothing, when it had not ended. The caller commits."""
    from services.coordinator_service import CoordinatorService

    if run.state not in OPENS_AGAIN:
        return False
    _card_runs_again(db, run, why=CARD_WHY.format(label=label), by=by)
    if run.state in FINISHED:
        _without_its_end(run)
    CoordinatorService().resume_mission(db, run.id, by)  # a failed one retries (F247); a spent budget is raised (F153)
    logger.info("[F284] mission %s runs again: %s sent back %s", run.id, by, label)
    return True


def _without_its_end(run: Any) -> None:
    """A finished mission's finish time, result, progress ledger and ingest markers go: it
    runs again, its old progress is never read as a stall (a retry drops the ledger too,
    F247), and its corrected result is saved when it completes again."""
    from modules.coordination.mission_retry import LEDGER_KEY

    dropped = {LEDGER_KEY, *INGEST_MARKERS}
    run.completed_at = None
    run.output_summary = None
    run.config = {key: value for key, value in (run.config or {}).items() if key not in dropped}


def _card_runs_again(db: Any, run: Any, *, why: str, by: str) -> None:
    from core.models.core import BoardTask
    from services.redo_cards import card_runs_again
    from services.run_cancel import MISSION_CARD

    card = db.query(BoardTask).filter(BoardTask.source_type == MISSION_CARD, BoardTask.orchestration_run_id == run.id,
                                      BoardTask.workspace_id == run.workspace_id).first()
    if card is not None:
        card_runs_again(card, why=why, by=by)


__all__ = ["BUSY", "CANCELLED", "LOCK_WAIT", "NOT_APPROVED", "busy_refusal", "locked_run", "lost_the_lock",
           "refusal_for", "reopen", "touched", "waits_briefly"]
