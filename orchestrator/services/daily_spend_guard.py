"""The day's spend ceiling, and the stop it is supposed to apply (F034).

``llm_cost_audit.daily_budget_alert_usd`` has been seeded at $20 for a long
time and NOTHING read it: it did not alert and it did not stop. Night 1
(2026-09-18) spent $31.56 with the ceiling set to 20, and the persona's own
report asked for exactly this — "a budget that stops something".

What a ceiling may and may not do matters here. It does NOT kill work that is
already running: cancelling a Claude Code session mid-write loses the work and
the money already spent on it. It refuses to START new autonomous work — a
board ticket picked up for execution, a mission dispatched — which is where the
next dollar would go. A human asking Auto a question is never blocked: the
person is present, they can see the number, and it is their call.

Subscription-billed CLI sessions cost no API money and are deliberately still
counted in the read-out but never the reason for a stop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

SPEND_CATEGORY = "llm_cost_audit"
SPEND_CEILING_KEY = "daily_budget_alert_usd"
# 0 or unset means no ceiling — the guard is off, which is the historical
# behaviour and stays the default for anyone who never set the dial.
NO_CEILING = 0.0


@dataclass(frozen=True)
class SpendState:
    """Where the day stands against its ceiling."""

    spent_usd: float
    ceiling_usd: float
    over: bool
    since: datetime

    @property
    def message(self) -> str:
        return (
            f"Spend this window is ${self.spent_usd:.2f}, over the "
            f"${self.ceiling_usd:.2f} ceiling (llm_cost_audit.spend_ceiling_usd), "
            f"counted since {self.since:%Y-%m-%d %H:%M} UTC. No new autonomous work "
            "will start until the ceiling is raised or the window rolls over. Work "
            "already running is not affected, and you can still ask Auto anything — "
            "the ceiling stops the platform spending on its own, not you."
        )


def ceiling_usd(db: Session) -> float:
    """The configured ceiling, or ``NO_CEILING`` when unset/unreadable.

    Its OWN dial first (``spend_ceiling_usd``, or the SPEND_CEILING_USD env),
    then the legacy ``daily_budget_alert_usd`` so an operator who set the old
    key still gets a ceiling. The old key's name says "alert", which is exactly
    why borrowing it was wrong.
    """
    from config import config as app_config

    for category, key in ((SPEND_CATEGORY, "spend_ceiling_usd"), (SPEND_CATEGORY, SPEND_CEILING_KEY)):
        try:
            from core.llm.manager import get_system_setting

            raw = get_system_setting(category, key, "")
            if raw not in (None, ""):
                return max(0.0, float(raw))
        except Exception:  # noqa: BLE001 — an unreadable dial must not block work
            logger.debug("[spend-guard] setting %s.%s unreadable", category, key, exc_info=True)
    try:
        return max(0.0, float(getattr(app_config, "SPEND_CEILING_USD", 0) or 0))
    except (TypeError, ValueError):
        return NO_CEILING


def window_start() -> datetime:
    """When the current spend window opened.

    The window is a "spend day" beginning at ``SPEND_DAY_START_HOUR`` in
    ``SPEND_TIMEZONE`` — midday by default. An overnight run from the evening
    into the small hours therefore sits wholly inside ONE window, and the
    previous night sits wholly inside the previous one. A UTC calendar day did
    neither: it split every run down the middle at 01:00 local and charged last
    night's spend to tonight.
    """
    from config import config as app_config

    hour = int(getattr(app_config, "SPEND_DAY_START_HOUR", 12) or 0)
    hour = min(max(hour, 0), 23)
    try:
        tz = ZoneInfo(str(getattr(app_config, "SPEND_TIMEZONE", "UTC") or "UTC"))
    except Exception:  # noqa: BLE001 — an unknown zone must not stop the guard
        logger.debug("[spend-guard] unknown SPEND_TIMEZONE — using UTC", exc_info=True)
        tz = timezone.utc

    local_now = datetime.now(tz)
    start = local_now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if local_now < start:
        start -= timedelta(days=1)      # before today's boundary → still yesterday's window
    return start.astimezone(timezone.utc)


def spent_today_usd(db: Session, workspace_id: Any) -> float:
    """This window's API spend for this workspace, from the ledger."""
    try:
        total = db.execute(
            sa_text(
                "SELECT COALESCE(SUM(total_cost), 0) FROM llm_usage "
                "WHERE workspace_id = CAST(:ws AS uuid) AND created_at >= :since"
            ),
            {"ws": str(workspace_id), "since": window_start()},
        ).scalar()
        return float(total or 0.0)
    except Exception:  # noqa: BLE001 — fail OPEN: a ledger fault must not halt the platform
        logger.warning("[spend-guard] could not read this window's spend — allowing", exc_info=True)
        return 0.0


def spend_state(db: Session, workspace_id: Any) -> SpendState:
    """Today's spend against the ceiling. Fails open in every direction."""
    limit = ceiling_usd(db)
    since = window_start()
    if limit <= NO_CEILING:
        return SpendState(spent_usd=0.0, ceiling_usd=0.0, over=False, since=since)
    spent = spent_today_usd(db, workspace_id)
    return SpendState(spent_usd=spent, ceiling_usd=limit, over=spent >= limit, since=since)


def refuse_new_work(db: Session, workspace_id: Any, what: str) -> Optional[str]:
    """The reason to refuse starting ``what``, or ``None`` to go ahead.

    Call this at the point new autonomous work would START, never on a path a
    human is waiting on.
    """
    state = spend_state(db, workspace_id)
    if not state.over:
        return None
    logger.warning("[spend-guard] refusing to start %s — %s", what, state.message)
    return state.message
