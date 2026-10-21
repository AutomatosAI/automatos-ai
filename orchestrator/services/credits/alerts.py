"""Balance alerts: running low, and run out (PRD-265 — "when a user burns their $20 on
a $29 plan, we tell them").

A workspace's **allowance** is what it was given to spend: the plan's monthly
credits once billing is open on a plan with platform AI, otherwise the trial.
Two alerts, each sent once per crossing, to the workspace's notification
destinations (the bell, and wherever the owner routes ``credits_low`` /
``credits_exhausted``):

- **running low** — the spendable balance is at or under ``CREDITS_ALERT_LOW_SHARE``
  of the allowance (20%: Pro at 400 of 2,000; a trial at 100 of 500);
- **run out** — nothing spendable is left: work on the platform's AI keys pauses,
  and calls on the workspace's own keys keep working.

``credit_balances.alert_level`` remembers the last alert sent; when credits are
added (a top-up, the new month, an admin grant) the level drops back, so the next
crossing alerts again. The summary endpoint uses the same levels for the amber and
paused states.
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Optional

from sqlalchemy import text

from services.credits.ledger import read_balance
from services.credits.routing import FUNDED_BY_CREDITS, funding_for

logger = logging.getLogger(__name__)

LEVEL_OK = 0
LEVEL_LOW = 1
LEVEL_OUT = 2

EVENT_LOW = "credits_low"
EVENT_OUT = "credits_exhausted"


def allowance(plan: Optional[str]) -> int:
    """What the workspace was given to spend: the plan's month, or the trial."""
    from config import config
    from services.plan_tiers import monthly_credits, platform_ai_allowed

    if config.BILLING_OPEN and platform_ai_allowed(plan) and monthly_credits(plan) > 0:
        return monthly_credits(plan)
    return int(config.TRIAL_CREDITS)


def level_for(spendable: int, plan: Optional[str]) -> int:
    """The alert level a spendable balance is at."""
    from config import config

    if spendable <= 0:
        return LEVEL_OUT
    if spendable <= math.ceil(allowance(plan) * float(config.CREDITS_ALERT_LOW_SHARE)):
        return LEVEL_LOW
    return LEVEL_OK


def _copy(level: int, spendable: int) -> tuple[str, str]:
    from config import config

    way_out = ("Top up or upgrade on the Billing tab, or add your own AI key in Settings."
               if config.BILLING_OPEN else "Add your own AI key in Settings to keep going.")
    if level == LEVEL_OUT:
        return ("Your AI credits have run out",
                f"Work on Automatos's AI keys is paused; calls on your own keys keep working. {way_out}")
    return ("Your AI credits are running low", f"{spendable} credits left. {way_out}")


def _notify(db: Any, workspace_id: str, level: int, spendable: int) -> None:
    from core.services.notification_dispatcher import NotificationDispatcher

    title, message = _copy(level, spendable)
    event = EVENT_OUT if level == LEVEL_OUT else EVENT_LOW
    try:
        asyncio.run(NotificationDispatcher(db, workspace_id).dispatch(
            event_type=event, title=title, message=message, status="warning",
        ))
    except Exception:  # noqa: BLE001 — an alert that can't be sent never stops the meter
        logger.exception("[credits] %s alert for workspace %s not sent", event, workspace_id)


def check_alerts(db: Any, workspace_id: str) -> Optional[int]:
    """Send the low / run-out alert once per crossing, and reset the level when
    credits were added. Returns the level alerted, or None. Commits."""
    from config import config

    row = db.execute(text(
        "SELECT w.plan, b.alert_level FROM credit_balances b JOIN workspaces w ON w.id = b.workspace_id "
        "WHERE b.workspace_id = CAST(:ws AS uuid)"
    ), {"ws": str(workspace_id)}).fetchone()
    if row is None:
        return None
    plan, stored = row[0], int(row[1] or 0)
    funding = funding_for(plan, read_balance(db, workspace_id), edition=config.AUTH_EDITION)
    if funding.funded_by != FUNDED_BY_CREDITS:
        return None
    level = level_for(funding.spendable, plan)
    if level == stored:
        return None
    db.execute(text("UPDATE credit_balances SET alert_level = :level WHERE workspace_id = CAST(:ws AS uuid)"),
               {"level": level, "ws": str(workspace_id)})
    db.commit()
    if level < stored:
        return None
    _notify(db, str(workspace_id), level, funding.spendable)
    return level
