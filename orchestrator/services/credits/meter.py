"""The credit meter: every platform-funded model call costs what it cost (PRD-265).

Credits are metered (Gerard, 20 Oct 2026 — like OpenRouter: tokens × the model's
price, not a fixed charge per ask). Every ``CREDITS_METER_INTERVAL_SECONDS`` the
meter takes the model calls waiting for it — ``llm_usage`` rows with
``charge_class = 'customer'`` (customer work; overhead is absorbed, decision D-3)
— and for each one:

- on the customer's own key, the local edition, or an allowance plan: nothing;
- otherwise it debits ``credits_for(cost)``: the row's ``total_cost`` (the
  provider's own figure when it reported one, else the model's registered price —
  one price source, ``UsageTracker.track``). A free route (NVIDIA, price
  multiplier 0) and a subscription session stay at $0. A paid route whose price
  is unknown books $0 — that is metered at the conservative estimate instead, with
  a warning naming the model so its price gets added;
- background work never spends the trial;
- the row becomes ``charged`` in the same transaction as its debit, which takes it
  out of the meter's index.

Each debit is keyed on its ``llm_usage`` row, so a tick that runs twice, or two
ticks that overlap, change nothing the second time. After a batch, each touched
workspace gets its balance alerts (``alerts.py``). Once a day the absorbed overhead
is recorded per workspace (:func:`record_overhead`) for margin reporting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional

from sqlalchemy import text

from services.credits.asks import is_background_lane
from services.credits.ledger import BUCKET_OVERHEAD, KIND_DEBIT, Balance, Entry, record
from services.credits.routing import FUNDED_BY_CREDITS, credits_for, free_workspace_ids, funding_for

logger = logging.getLogger(__name__)

SOURCE_USAGE = "usage"
SOURCE_OVERHEAD_DAY = "overhead_day"
BAND_METERED = "metered"
CUSTOMER = "customer"
CHARGED = "charged"
OVERHEAD = "overhead"
SUBSCRIPTION_TIER = "subscription"
BATCH = 500
MAX_BATCHES = 20


@dataclass(frozen=True)
class PendingCall:
    """One customer-work model call waiting for the meter."""

    usage_id: int
    workspace_id: str
    model_id: str
    provider: str
    tier: Optional[str]
    request_type: Optional[str]
    input_tokens: int
    output_tokens: int
    total_cost: float
    is_byok: bool


def billable_cost(call: PendingCall) -> float:
    """What the call cost the platform, in USD (0 when the platform didn't pay)."""
    if call.is_byok:
        return 0.0
    if call.total_cost > 0:
        return call.total_cost
    if not (call.input_tokens or call.output_tokens) or call.tier == SUBSCRIPTION_TIER:
        return 0.0
    from core.llm.providers import price_multiplier_for

    multiplier = price_multiplier_for(call.provider)
    if multiplier <= 0:
        return 0.0  # a free route (NVIDIA) — $0 is its real price
    from core.llm.manager import estimate_cost_usd

    estimate = estimate_cost_usd(call.model_id, call.input_tokens, call.output_tokens) * multiplier
    logger.warning("[credits] %s via %s has no known price: metered at the estimate $%.6f (usage %s)",
                   call.model_id, call.provider, estimate, call.usage_id)
    return estimate


def _pending(db: Any, after_id: int) -> list[PendingCall]:
    rows = db.execute(text(
        "SELECT id, workspace_id::text, model_id, provider, tier, request_type, input_tokens, output_tokens, "
        "total_cost, COALESCE(is_byok, false) FROM llm_usage "
        "WHERE charge_class = :customer AND id > :after ORDER BY id LIMIT :batch"
    ), {"customer": CUSTOMER, "after": after_id, "batch": BATCH}).fetchall()
    return [PendingCall(int(r[0]), r[1], r[2] or "unknown", r[3] or "unknown", r[4], r[5],
                        int(r[6] or 0), int(r[7] or 0), float(r[8] or 0), bool(r[9])) for r in rows]


def _plans(db: Any, workspace_ids: set) -> dict:
    rows = db.execute(text("SELECT id::text, plan FROM workspaces WHERE id::text = ANY(:ids)"),
                      {"ids": sorted(workspace_ids)}).fetchall()
    return {r[0]: r[1] for r in rows}


def meter_call(db: Any, call: PendingCall, plan: Optional[str], *, free: bool = False) -> bool:
    """Debit one call (if the platform paid for it) and mark it charged. Commits.
    Returns True when credits were debited. A free owner's call is recorded, never debited."""
    from config import config

    funding = funding_for(plan, Balance(), edition=config.AUTH_EDITION,
                          background=is_background_lane(call.request_type), free=free)
    cost = billable_cost(call) if funding.funded_by == FUNDED_BY_CREDITS else 0.0
    credits = credits_for(cost)
    if credits:
        record(db, call.workspace_id, Entry(
            kind=KIND_DEBIT, source_type=SOURCE_USAGE, source_ref=str(call.usage_id), credits=-credits,
            actor="system", reason=call.model_id[:255], band=BAND_METERED, charged_credits=credits,
            actual_cost_usd=round(cost, 6),
        ), spend_buckets=funding.spend_buckets)
    db.execute(text("UPDATE llm_usage SET charge_class = :charged WHERE id = :id"),
               {"charged": CHARGED, "id": call.usage_id})
    db.commit()
    return bool(credits)


def _meter_batch(db: Any, calls: list[PendingCall]) -> tuple[int, set]:
    plans = _plans(db, {c.workspace_id for c in calls})
    free = free_workspace_ids(db, plans)
    debited, touched = 0, set()
    for call in calls:
        try:
            if meter_call(db, call, plans.get(call.workspace_id), free=call.workspace_id in free):
                debited += 1
                touched.add(call.workspace_id)
        except Exception:  # noqa: BLE001 — one bad row never blocks the rest; it stays pending
            db.rollback()
            logger.exception("[credits] metering usage %s failed", call.usage_id)
    return debited, touched


def meter_pending(db: Any) -> int:
    """Meter every call waiting (in batches), then send the touched workspaces'
    balance alerts. Returns how many calls were debited."""
    from services.credits.alerts import check_alerts

    debited, touched, after_id = 0, set(), 0
    for _ in range(MAX_BATCHES):
        calls = _pending(db, after_id)
        if not calls:
            break
        count, workspaces = _meter_batch(db, calls)
        debited, touched = debited + count, touched | workspaces
        after_id = calls[-1].usage_id
    for workspace_id in sorted(touched):
        check_alerts(db, workspace_id)
    return debited


def record_overhead(db: Any, day: date) -> int:
    """One zero-credit ``overhead`` row per workspace for ``day`` (UTC): the cost of
    its absorbed overhead calls (D-3), for margin reporting. Commits."""
    start = datetime(day.year, day.month, day.day)
    rows = db.execute(text(
        "SELECT workspace_id::text, SUM(total_cost) FROM llm_usage "
        "WHERE charge_class = :overhead AND NOT COALESCE(is_byok, false) AND total_cost > 0 "
        "AND created_at >= :start AND created_at < :end GROUP BY workspace_id"
    ), {"overhead": OVERHEAD, "start": start, "end": start + timedelta(days=1)}).fetchall()
    written = 0
    for workspace_id, cost in rows:
        written += int(record(db, workspace_id, Entry(
            kind=KIND_DEBIT, source_type=SOURCE_OVERHEAD_DAY, source_ref=f"{workspace_id}:{day.isoformat()}",
            credits=0, bucket=BUCKET_OVERHEAD, actor="system", reason="absorbed overhead (D-3)",
            actual_cost_usd=round(float(cost or 0), 6),
        )))
    db.commit()
    return written
