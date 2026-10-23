"""What a payment does to credits (PRD-265 US-010, FR-5, FR-7).

- **A paid period** (a subscription invoice): a new period first expires what is
  left of the last month, then grants the plan's monthly credits, expiring at the
  period end. A second invoice in the same period (an upgrade) grants the
  difference up to the new plan's month, prorated by the share of the period left,
  as Stripe prorates the price: an upgrade on the period's last day pays a day's
  difference and gets a day's credits. Toggling plans never mints credits.
  ``billing_accounts.credited_period_start``/``_end`` + ``period_credits`` remember
  what the period was given, whatever order the events arrive in.
- **A top-up** grants ``TOPUP_CREDITS`` per pack, valid ``TOPUP_VALID_DAYS``.
- **A refund or a dispute** takes back the paid-for share of the credits that
  payment granted, to a negative bucket if they were spent (work then pauses);
  a dispute won gives them back. Every movement is keyed on the payment, so a
  replay, or a later partial refund, moves only what is still owed.

Nothing here commits: the webhook's transaction lands the plan, the account and
the ledger together.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import text

from services.credits.expiry import expire_monthly
from services.credits.ledger import (
    BUCKET_MONTHLY,
    BUCKET_TOPUP,
    KIND_GRANT,
    KIND_REFUND,
    KIND_TOPUP,
    Entry,
    lock_balance,
    record,
)

logger = logging.getLogger(__name__)

SOURCE_INVOICE = "stripe_invoice"
SOURCE_TOPUP = "stripe_topup"
SOURCE_REFUND = "stripe_refund"
SOURCE_DISPUTE = "stripe_dispute"
SOURCE_DISPUTE_WON = "stripe_dispute_won"
CLAWBACK_SOURCES = (SOURCE_REFUND, SOURCE_DISPUTE, SOURCE_DISPUTE_WON)


def grant_source(source_type: str, source_ref: str) -> str:
    return f"{source_type}:{source_ref}"


def _same_period(a: Optional[datetime], b: datetime) -> bool:
    return a is not None and int(a.timestamp()) == int(b.timestamp())


def share_left(period_start: Optional[datetime], changed_at: Optional[datetime], period_end: datetime) -> float:
    """The share of a paid period still to run at ``changed_at`` (0–1). Pure.
    Unknown bounds count as the whole period."""
    if period_start is None or changed_at is None:
        return 1.0
    length = (period_end - period_start).total_seconds()
    if length <= 0:
        return 1.0
    return min(1.0, max(0.0, (period_end - changed_at).total_seconds() / length))


def grant_period(db: Any, account: Any, *, plan: str, invoice_id: str, period_end: datetime,
                 period_start: Optional[datetime] = None) -> int:
    """Credit one paid subscription invoice. Returns the credits granted."""
    from services.plan_tiers import monthly_credits

    workspace_id = account.workspace_id
    target = monthly_credits(plan)
    balance = lock_balance(db, workspace_id)
    if _same_period(account.credited_period_end, period_end):
        extra = max(0, target - int(account.period_credits or 0))
        grant = math.ceil(extra * share_left(account.credited_period_start, period_start, period_end))
        period_credits = max(target, int(account.period_credits or 0))
    else:
        expire_monthly(db, workspace_id, balance.period_end or account.credited_period_end)
        grant, period_credits = target, target
        account.credited_period_start = period_start
    if grant > 0:
        record(db, workspace_id, Entry(
            kind=KIND_GRANT, source_type=SOURCE_INVOICE, source_ref=invoice_id, credits=grant,
            bucket=BUCKET_MONTHLY, actor="stripe", reason=f"{plan} monthly credits", expires_at=period_end,
        ))
    db.execute(text("UPDATE credit_balances SET period_end = :end WHERE workspace_id = CAST(:ws AS uuid)"),
               {"end": period_end, "ws": str(workspace_id)})
    account.credited_period_end = period_end
    account.period_credits = period_credits
    return grant


def grant_topup(db: Any, workspace_id: Any, *, session_id: str, quantity: int) -> int:
    """Credit one paid top-up. Returns the credits granted (0 on a replay)."""
    from config import config

    credits = int(config.TOPUP_CREDITS) * max(1, int(quantity))
    expires_at = datetime.now(timezone.utc) + timedelta(days=int(config.TOPUP_VALID_DAYS))
    written = record(db, workspace_id, Entry(
        kind=KIND_TOPUP, source_type=SOURCE_TOPUP, source_ref=session_id, credits=credits,
        bucket=BUCKET_TOPUP, actor="stripe", reason="top-up", expires_at=expires_at,
    ))
    return credits if written else 0


@dataclass(frozen=True)
class Granted:
    """The credits one payment granted: where they went and how many."""

    workspace_id: Any
    bucket: str
    credits: int
    amount_cents: Optional[int]


def granted_by_payment(db: Any, payment_ref: str) -> Optional[Granted]:
    """What the payment ``payment_ref`` granted, found through the event that
    recorded it (``billing_events.payment_ref`` → ``grant_source`` → the ledger)."""
    row = db.execute(text(
        "SELECT grant_source, amount_cents FROM billing_events WHERE payment_ref = :ref "
        "AND grant_source IS NOT NULL ORDER BY id LIMIT 1"
    ), {"ref": payment_ref}).fetchone()
    if row is None:
        return None
    source_type, _, source_ref = str(row[0]).partition(":")
    rows = db.execute(text(
        "SELECT workspace_id, bucket, credits FROM credit_ledger WHERE source_type = :t AND source_ref = :r "
        "AND credits > 0 ORDER BY id"
    ), {"t": source_type, "r": source_ref}).fetchall()
    if not rows:
        return None
    return Granted(workspace_id=rows[0][0], bucket=rows[0][1], credits=sum(int(r[2]) for r in rows),
                   amount_cents=row[1])


def clawed_back(db: Any, payment_ref: str) -> int:
    """Credits taken back from ``payment_ref``'s grant so far (net of disputes won)."""
    prefix = f"{payment_ref}:"
    total = db.execute(text(
        "SELECT COALESCE(SUM(credits), 0) FROM credit_ledger WHERE source_type = ANY(:sources) "
        "AND left(source_ref, :n) = :prefix"
    ), {"sources": list(CLAWBACK_SOURCES), "n": len(prefix), "prefix": prefix}).scalar()
    return -int(total or 0)


def _share(credits: int, part_cents: int, whole_cents: Optional[int]) -> int:
    """The credits ``part`` of a payment of ``whole`` paid for, rounded down (a
    refund never takes back more than it paid for). Pure."""
    if not whole_cents or whole_cents <= 0 or part_cents >= whole_cents:
        return credits
    return (credits * max(0, part_cents)) // whole_cents


def _take(db: Any, granted: Granted, *, credits: int, source_type: str, source_ref: str, reason: str) -> int:
    if credits <= 0:
        return 0
    written = record(db, granted.workspace_id, Entry(
        kind=KIND_REFUND, source_type=source_type, source_ref=source_ref, credits=-credits,
        bucket=granted.bucket, actor="stripe", reason=reason,
    ))
    return credits if written else 0


def claw_back_refund(db: Any, granted: Granted, *, payment_ref: str, charge_cents: int, refunded_cents: int) -> int:
    """Take back the refunded share of a payment's credits (the running total)."""
    owed = _share(granted.credits, refunded_cents, charge_cents or granted.amount_cents)
    already = clawed_back(db, payment_ref)
    take = min(owed, granted.credits) - already
    return _take(db, granted, credits=take, source_type=SOURCE_REFUND,
                 source_ref=f"{payment_ref}:refund:{refunded_cents}", reason="payment refunded")


def claw_back_dispute(db: Any, granted: Granted, *, payment_ref: str, dispute_id: str, disputed_cents: int) -> int:
    """Take back the disputed share of a payment's credits while it is disputed."""
    owed = _share(granted.credits, disputed_cents, granted.amount_cents)
    take = min(owed, granted.credits - clawed_back(db, payment_ref))
    return _take(db, granted, credits=take, source_type=SOURCE_DISPUTE,
                 source_ref=f"{payment_ref}:dispute:{dispute_id}", reason="payment disputed")


def restore_dispute(db: Any, granted: Granted, *, payment_ref: str, dispute_id: str) -> int:
    """A dispute won: give back the credits it took."""
    taken = -int(db.execute(text(
        "SELECT COALESCE(SUM(credits), 0) FROM credit_ledger WHERE source_type = :t AND source_ref = :r"
    ), {"t": SOURCE_DISPUTE, "r": f"{payment_ref}:dispute:{dispute_id}"}).scalar() or 0)
    if taken <= 0:
        return 0
    written = record(db, granted.workspace_id, Entry(
        kind=KIND_GRANT, source_type=SOURCE_DISPUTE_WON, source_ref=f"{payment_ref}:dispute_won:{dispute_id}",
        credits=taken, bucket=granted.bucket, actor="stripe", reason="dispute won",
    ))
    return taken if written else 0
