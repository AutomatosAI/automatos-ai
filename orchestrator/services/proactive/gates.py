"""PRD-268 §6.9, FR-13, K14 — what a check-in checks before anything else, in order.

1. **The switches** (``services.proactive.switches``): the platform mode and the workspace
   switch (off is ``skipped_off``), then the addressee's pause (``skipped_paused``). A
   workspace with no owner to address has nobody to suggest to: ``skipped_off``.
2. **``background_allowed``** (``services.credits.routing``): something other than the
   trial pays for background work, else ``skipped_credits``. A check that cannot complete
   refuses too: a check-in is never worth paying for from the trial.
3. **``refuse_new_work``** (``services.daily_spend_guard``): the day's spend ceiling, else
   ``skipped_cap``. The ceiling is off (0) by default and fails open, as for every other
   start of autonomous work.
4. **The caps** (``modules.proactive.run_refusal``): the workspace's check-ins that reached
   the model in the last day, against ``max_model_runs_per_day`` (``skipped_cap``). The
   §6.9 defaults; a workspace's own caps are US-C07a's.

The first check that refuses names the run's outcome and what refused it. A refused
check-in writes its run row and makes no model call (the runner).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.proactive import RUN_SKIPPED_CAP, RUN_SKIPPED_CREDITS, RUN_SKIPPED_OFF, RUN_SKIPPED_PAUSED
from modules.proactive import Caps, run_refusal
from services.proactive.runs import model_runs_today
from services.proactive.switches import MODE_OFF, effective_mode, person_paused

logger = logging.getLogger(__name__)

__all__ = [
    "CREDITS_UNREADABLE",
    "NO_ADDRESSEE",
    "PAUSED",
    "SPEND_CEILING",
    "SWITCHED_OFF",
    "TRIAL_ONLY",
    "Refusal",
    "start_refusal",
]

# What refused a check-in's start, recorded as its run's error_class.
SWITCHED_OFF = "switched_off"
NO_ADDRESSEE = "no_addressee"
PAUSED = "paused"
TRIAL_ONLY = "trial_only"
CREDITS_UNREADABLE = "credits_unreadable"
SPEND_CEILING = "spend_ceiling"


@dataclass(frozen=True)
class Refusal:
    """A check-in refused at its start: the run outcome, and what refused it."""

    outcome: str
    reason: str


def _switch_refusal(db: Session, workspace_id: UUID, owner_id: Optional[int]) -> Optional[Refusal]:
    if effective_mode(db, workspace_id) == MODE_OFF:
        return Refusal(RUN_SKIPPED_OFF, SWITCHED_OFF)
    if owner_id is None:
        return Refusal(RUN_SKIPPED_OFF, NO_ADDRESSEE)
    if person_paused(db, workspace_id, owner_id):
        return Refusal(RUN_SKIPPED_PAUSED, PAUSED)
    return None


def _credits_refusal(db: Session, workspace_id: UUID) -> Optional[Refusal]:
    from services.credits.routing import background_allowed

    try:
        with db.begin_nested():
            allowed = background_allowed(db, workspace_id)
    except Exception:  # noqa: BLE001 — a check that did not complete refuses the check-in
        logger.exception("[proactive] background_allowed could not be read for %s: check-in refused", workspace_id)
        return Refusal(RUN_SKIPPED_CREDITS, CREDITS_UNREADABLE)
    return None if allowed else Refusal(RUN_SKIPPED_CREDITS, TRIAL_ONLY)


def _spend_refusal(db: Session, workspace_id: UUID, what: str) -> Optional[Refusal]:
    from services.daily_spend_guard import refuse_new_work

    return Refusal(RUN_SKIPPED_CAP, SPEND_CEILING) if refuse_new_work(db, workspace_id, what) else None


def _cap_refusal(db: Session, workspace_id: UUID, now: datetime) -> Optional[Refusal]:
    refusal = run_refusal(Caps(), model_runs_today(db, workspace_id, now))
    return Refusal(refusal.outcome, refusal.cap) if refusal is not None else None


def start_refusal(db: Session, workspace_id: UUID, owner_id: Optional[int], now: datetime,
                  what: str) -> Optional[Refusal]:
    """The first check that refuses this check-in's start, or None to go ahead. ``owner_id``
    is the addressee (K11); ``what`` names the check-in in the spend guard's log."""
    checks: Tuple[Callable[[], Optional[Refusal]], ...] = (
        lambda: _switch_refusal(db, workspace_id, owner_id),
        lambda: _credits_refusal(db, workspace_id),
        lambda: _spend_refusal(db, workspace_id, what),
        lambda: _cap_refusal(db, workspace_id, now),
    )
    for check in checks:
        refusal = check()
        if refusal is not None:
            logger.info("[proactive] %s refused: %s (%s)", what, refusal.outcome, refusal.reason)
            return refusal
    return None
