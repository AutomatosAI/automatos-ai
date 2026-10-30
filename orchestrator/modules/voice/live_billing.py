"""
PRD-265 D-6: an Auto Live call is paid from credits, like chat
===============================================================

Gerard, 30 Oct 2026: "voice should just work like chat or anything they do, it's billed
against their credits so their choice to use voice or not". So:

- **What a call costs** is what Retell charges us for it: the ``call_ended`` event's
  ``call.call_cost.combined_cost``, in cents. It is charged at the rule every model call
  follows, ``ceil(cost x 100 x CREDITS_MARKUP)`` credits (``routing.credits_for``). An event
  that carries no cost is charged by the call's minutes at
  ``VOICE_LIVE_ESTIMATE_USD_PER_MINUTE``, with a warning naming the call, as the meter
  charges a model it has no price for.
- **Once per call:** the debit is keyed on the call id, so a redelivered event changes nothing.
- **Who pays:** a workspace paying in credits. The local edition, an allowance plan and a free
  owner are never charged; their calls are still kept in ``voice_calls``.
- **No credits, no call:** :func:`has_credits` is checked before Retell is asked
  (``mint.require_voice_live_plan``).
- **What this does not charge:** the spoken answers' model calls are metered as chat turns
  already (``consumers/chatbot/service.py``). Our own server voice (``api/voice_speech.py``)
  is free: it costs us no more per use than the device's own voice.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from sqlalchemy.orm import Session

from config import config
from services.credits.ledger import KIND_DEBIT, Entry, record
from services.credits.meter import BAND_METERED
from services.credits.routing import FUNDED_BY_CREDITS, can_pay, credits_for, workspace_funding

logger = logging.getLogger(__name__)

# The ledger's source for a call's charge; the Billing tab shows it as "Auto Live call".
SOURCE_VOICE_LIVE = "voice_live_call"
CREDITS_REFUSAL_MESSAGE = (
    "Auto Live is paid from your credits, and this workspace has none left. "
    "Top up on the Billing tab, then call again."
)
_SECONDS_PER_MINUTE = 60
_CENTS_PER_DOLLAR = 100.0


def call_cost_usd(call: Any) -> Optional[float]:
    """Pure: what Retell says a call cost us, in dollars, or None when the event does not say."""
    cost = call.get("call_cost") if isinstance(call, dict) else None
    cents = cost.get("combined_cost") if isinstance(cost, dict) else None
    if isinstance(cents, bool) or not isinstance(cents, (int, float)):
        return None
    if not math.isfinite(cents) or cents < 0:
        return None
    return cents / _CENTS_PER_DOLLAR


def estimate_usd(duration_seconds: Optional[int]) -> float:
    """Pure: a call's minutes at the estimate rate, for an event with no cost."""
    minutes = max(0, duration_seconds or 0) / _SECONDS_PER_MINUTE
    return minutes * float(config.VOICE_LIVE_ESTIMATE_USD_PER_MINUTE)


def has_credits(db: Session, workspace_id: Any) -> bool:
    """Whether a call may start: a workspace paying in credits needs some left; any other pays nothing."""
    return can_pay(workspace_funding(db, workspace_id), 1)


def _cost_of(row: Any, call: Any) -> Optional[float]:
    cost = call_cost_usd(call)
    if cost is not None:
        return cost
    if row.status != "ended":  # never connected and no cost given: nothing to charge
        return None
    estimate = estimate_usd(row.duration_seconds)
    logger.warning("[credits] Auto Live call %s carried no cost: charged at the estimate $%.4f",
                   row.call_id, estimate)
    return estimate


def charge_call(db: Session, row: Any, call: Any) -> int:
    """Debit one finished call's cost from its workspace's credits. Returns the credits charged
    (0 when nothing was: not paying in credits, nothing to charge, or charged already).

    It does not commit: the caller folds the charge into the same transaction as the call's row.
    """
    if row.workspace_id is None:
        return 0
    funding = workspace_funding(db, row.workspace_id)
    if funding.funded_by != FUNDED_BY_CREDITS:
        return 0
    cost = _cost_of(row, call)
    credits = credits_for(cost or 0.0)
    if not credits:
        return 0
    minutes = math.ceil(max(0, row.duration_seconds or 0) / _SECONDS_PER_MINUTE)
    recorded = record(db, row.workspace_id, Entry(
        kind=KIND_DEBIT, source_type=SOURCE_VOICE_LIVE, source_ref=str(row.call_id), credits=-credits,
        actor="system", reason=f"Auto Live call, {minutes} min", band=BAND_METERED,
        charged_credits=credits, actual_cost_usd=round(cost, 6),
    ), spend_buckets=funding.spend_buckets)
    return credits if recorded else 0  # a redelivered event: charged already, nothing changes


def after_charge(db: Session, workspace_id: Any) -> None:
    """After the charge is committed: the running-low and run-out notices, as after the meter's."""
    from services.credits.alerts import check_alerts

    check_alerts(db, str(workspace_id))
