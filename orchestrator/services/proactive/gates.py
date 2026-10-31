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
4. **The caps** (``services.proactive.limits``): the workspace's own over the §6.9 defaults
   in config. Caps that cannot be read, or that are wrong, refuse (``skipped_cap``,
   ``caps_unreadable``). Then each refusal is ``skipped_cap``, named by its cap: the
   workspace's check-ins that reached the model in the last day (``max_model_runs_per_day``);
   a check-in sooner than ``min_interval_minutes`` after its last run that ran, less the
   scheduler's grace; and a suggestion cap at 0, which leaves nothing to record. The caps
   that count suggestions refuse in ``platform_suggest_next_step``, so the read-only run
   still goes ahead and a failing streak can still end.

The first check that refuses names the run's outcome and what refused it. A refused
check-in writes its run row and makes no model call (the runner).

PRD-268 W3 (K11): a follow-up's firing starts through the same checks (``follow_up=True``,
the person as the addressee), except two that are a check-in's own. It is due when Auto said,
not on an interval, so ``min_interval_minutes`` (read against Auto's workspace check-in when
there is no agent) never refuses it; and it may say one thing where it cannot suggest, so a
suggestion cap at 0 does not refuse it either (``platform_suggest_next_step`` still does).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.proactive import RUN_SKIPPED_CAP, RUN_SKIPPED_CREDITS, RUN_SKIPPED_OFF, RUN_SKIPPED_PAUSED
from modules.proactive import CapRefusal, Caps, interval_refusal, run_refusal, suggestion_refusal
from services.proactive.limits import workspace_caps
from services.proactive.runs import (
    KIND_AGENT,
    KIND_WORKSPACE,
    SCHEDULER_GRACE,
    finished_runs,
    model_runs_today,
    ran_at,
)
from services.proactive.switches import MODE_OFF, effective_mode, person_paused

logger = logging.getLogger(__name__)

__all__ = [
    "CAPS_UNREADABLE",
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
CAPS_UNREADABLE = "caps_unreadable"


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


def _read_caps(db: Session, workspace_id: UUID) -> Optional[Caps]:
    """The workspace's caps, or None when they cannot be read or are wrong (logged)."""
    try:
        with db.begin_nested():
            return workspace_caps(db, workspace_id)
    except Exception:  # noqa: BLE001 — caps that cannot be read refuse the check-in
        logger.exception("[proactive] the caps of %s could not be read: check-in refused", workspace_id)
        return None


def _too_soon(db: Session, workspace_id: UUID, agent_id: Optional[int], caps: Caps,
              now: datetime) -> Optional[CapRefusal]:
    """``min_interval_minutes`` while this check-in's last run that ran is too recent."""
    kind = KIND_WORKSPACE if agent_id is None else KIND_AGENT
    last = finished_runs(db, workspace_id, kind, agent_id, 1)
    return interval_refusal(caps, ran_at(last[0]) if last else None, now, SCHEDULER_GRACE)


def _allows_nothing(caps: Caps, agent_id: Optional[int]) -> Optional[CapRefusal]:
    """A suggestion cap at 0, which leaves the check-in nothing it could record. Caps that
    count (the day's new ones, an agent's open ones) are the suggestion tool's: the read-only
    run still goes ahead, so a failing streak can still end."""
    return suggestion_refusal(caps, made_this_run=0, new_today_for_person=0,
                              open_for_agent=None if agent_id is None else 0)


def _cap_refusal(db: Session, workspace_id: UUID, agent_id: Optional[int], now: datetime,
                 follow_up: bool) -> Optional[Refusal]:
    caps = _read_caps(db, workspace_id)
    if caps is None:
        return Refusal(RUN_SKIPPED_CAP, CAPS_UNREADABLE)
    model_runs: Tuple[Callable[[], Optional[CapRefusal]], ...] = (
        lambda: run_refusal(caps, model_runs_today(db, workspace_id, now)),
    )
    check_in_only: Tuple[Callable[[], Optional[CapRefusal]], ...] = (
        lambda: _too_soon(db, workspace_id, agent_id, caps, now),
        lambda: _allows_nothing(caps, agent_id),
    )
    checks = model_runs if follow_up else model_runs + check_in_only
    for check in checks:
        refusal = check()
        if refusal is not None:
            return Refusal(refusal.outcome, refusal.cap)
    return None


def start_refusal(db: Session, workspace_id: UUID, owner_id: Optional[int], now: datetime,
                  what: str, agent_id: Optional[int] = None, *, follow_up: bool = False) -> Optional[Refusal]:
    """The first check that refuses this check-in's start, or None to go ahead. ``owner_id``
    is the addressee (K11); ``what`` names the check-in in the spend guard's log;
    ``agent_id`` is the agent whose check-in it is, None for Auto's workspace check-in;
    ``follow_up`` a follow-up's firing, which no interval and no suggestion cap at 0 refuses."""
    checks: Tuple[Callable[[], Optional[Refusal]], ...] = (
        lambda: _switch_refusal(db, workspace_id, owner_id),
        lambda: _credits_refusal(db, workspace_id),
        lambda: _spend_refusal(db, workspace_id, what),
        lambda: _cap_refusal(db, workspace_id, agent_id, now, follow_up),
    )
    for check in checks:
        refusal = check()
        if refusal is not None:
            logger.info("[proactive] %s refused: %s (%s)", what, refusal.outcome, refusal.reason)
            return refusal
    return None
