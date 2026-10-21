"""The one-time trial, as credits (PRD-265 US-005; PRD-222 W1·S9 before it).

A new account's first workspace gets ``TRIAL_CREDITS`` (500) in the ``trial``
bucket, through the ledger. Once per person: the grant is keyed on its owner
(``source_type = 'trial_grant'``, ``source_ref = <owner id>``), so the ledger's
unique key refuses a second one whichever workspace asks. Grants pause — and the
person stays eligible — when the kill switch (``TRIAL_ENABLED``) is off or today's
platform-wide trial spend has reached ``TRIAL_GLOBAL_DAILY_CREDITS``; trials
already granted keep working.

Spending the trial is ordinary ask accounting (``services/credits/asks.py``): the
trial bucket is spent first. Background work is never paid from it.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text

from services.credits.ledger import (
    BUCKET_TRIAL,
    KIND_ADJUST,
    KIND_GRANT,
    Entry,
    is_recorded,
    read_balance,
    record,
)

logger = logging.getLogger(__name__)

SOURCE_TRIAL_GRANT = "trial_grant"
SOURCE_TRIAL_RESET = "trial_reset"

GRANTED = "granted"
DISABLED = "disabled"
ALREADY_HELD = "already_held"
DAILY_CAP_REACHED = "daily_cap_reached"


def owner_ref(workspace_id: Any, owner_id: Any) -> str:
    """The once-per-person key: the owner, or the workspace when it has none."""
    return str(owner_id) if owner_id is not None else f"workspace:{workspace_id}"


def trial_spent_today(db: Any) -> int:
    """Trial credits spent today (UTC), platform-wide, net of refunds."""
    row = db.execute(text(
        "SELECT COALESCE(-SUM(credits), 0) FROM credit_ledger "
        "WHERE bucket = 'trial' AND kind IN ('debit', 'refund') "
        "AND created_at >= date_trunc('day', now() AT TIME ZONE 'utc') AT TIME ZONE 'utc'"
    )).fetchone()
    return max(0, int(row[0] or 0)) if row else 0


def decide_trial_grant(*, enabled: bool, already_held: bool, spent_today: int, daily_cap: int) -> str:
    """Pure: ``granted``, or why not (``disabled`` | ``already_held`` | ``daily_cap_reached``)."""
    if not enabled:
        return DISABLED
    if already_held:
        return ALREADY_HELD
    if spent_today >= daily_cap:
        return DAILY_CAP_REACHED
    return GRANTED


def grant_trial(db: Any, workspace_id: Any, *, owner_id: Any) -> Optional[int]:
    """Grant the one-time trial to a freshly provisioned workspace.

    Returns the credits granted (the caller commits), or None when skipped,
    always logging why. A pause is a normal, visible outcome, never an error.
    """
    from config import config

    ref = owner_ref(workspace_id, owner_id)
    spent_today = trial_spent_today(db)
    decision = decide_trial_grant(
        enabled=config.TRIAL_ENABLED, already_held=is_recorded(db, SOURCE_TRIAL_GRANT, ref),
        spent_today=spent_today, daily_cap=config.TRIAL_GLOBAL_DAILY_CREDITS,
    )
    if decision == GRANTED:
        entry = Entry(kind=KIND_GRANT, source_type=SOURCE_TRIAL_GRANT, source_ref=ref,
                      credits=int(config.TRIAL_CREDITS), bucket=BUCKET_TRIAL, actor="system",
                      reason="one-time trial")
        decision = GRANTED if record(db, workspace_id, entry) else ALREADY_HELD
    if decision != GRANTED:
        logger.info("Trial grant skipped for workspace %s (owner=%s): %s (spent today %d/%d)",
                    workspace_id, owner_id, decision, spent_today, config.TRIAL_GLOBAL_DAILY_CREDITS)
        return None
    logger.info("Trial granted for workspace %s (owner=%s): %d credits", workspace_id, owner_id, config.TRIAL_CREDITS)
    return int(config.TRIAL_CREDITS)


def reset_trial(db: Any, workspace_id: Any, *, owner_id: Any) -> Optional[int]:
    """The dev onboarding reset (PRD-222 W1·S10): bring the trial back to full.

    The once-per-person grant never repeats, so a workspace that already held its
    trial gets an adjustment back up to ``TRIAL_CREDITS``; one that never held it
    gets the ordinary grant (and its pauses). Returns the trial balance after, or
    None when paused. Does not commit.
    """
    from config import config

    if not is_recorded(db, SOURCE_TRIAL_GRANT, owner_ref(workspace_id, owner_id)):
        return grant_trial(db, workspace_id, owner_id=owner_id)
    if not config.TRIAL_ENABLED:
        return None
    top_up = int(config.TRIAL_CREDITS) - read_balance(db, workspace_id).trial
    if top_up:
        record(db, workspace_id, Entry(
            kind=KIND_ADJUST, source_type=SOURCE_TRIAL_RESET, source_ref=uuid.uuid4().hex,
            credits=top_up, bucket=BUCKET_TRIAL, actor="onboarding-reset", reason="dev onboarding reset",
        ))
    return int(config.TRIAL_CREDITS)
