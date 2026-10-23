"""Credits that expire (PRD-265 §4.3).

- **A month's credits** expire at the end of their paid period. A paid renewal
  expires what is left before it grants the new month (``services/billing/grants.py``);
  when no renewal is paid (the card failed, the plan was cancelled) the hourly
  sweep expires them ``BILLING_MONTHLY_EXPIRY_GRACE_HOURS`` after the period end.
  Both paths key the expiry on ``(workspace, period end)``, so it happens once.
- **A top-up** lasts ``TOPUP_VALID_DAYS``. Top-ups are spent oldest first, so the
  credits still left of an expiring pack are the top-up balance less what the
  younger packs still hold; the sweep expires exactly that, then records a marker
  for the pack (even at zero) so it is never looked at again.

Only positive credits expire: a negative bucket (a clawback) stays to be repaid.
Each workspace's expiry is its own transaction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import text

from services.credits.ledger import BUCKET_MONTHLY, BUCKET_TOPUP, KIND_EXPIRE, Entry, lock_balance, record

logger = logging.getLogger(__name__)

SOURCE_MONTHLY_EXPIRE = "monthly_expire"
SOURCE_TOPUP_EXPIRE = "topup_expire"


def monthly_expiry_ref(workspace_id: Any, period_end: datetime) -> str:
    return f"{workspace_id}:{int(period_end.timestamp())}"


def expire_monthly(db: Any, workspace_id: Any, period_end: Optional[datetime]) -> int:
    """Expire what is left of the month that ended at ``period_end``. Returns the
    credits expired (0 when none were left or it was done already). Does not commit."""
    if period_end is None:
        return 0
    left = lock_balance(db, workspace_id).monthly
    if left <= 0:
        return 0
    written = record(db, workspace_id, Entry(
        kind=KIND_EXPIRE, source_type=SOURCE_MONTHLY_EXPIRE, source_ref=monthly_expiry_ref(workspace_id, period_end),
        credits=-left, bucket=BUCKET_MONTHLY, actor="system", reason="monthly credits expired at the period end",
    ))
    return left if written else 0


@dataclass(frozen=True)
class _Pack:
    source_type: str
    source_ref: str
    credits: int
    expires_at: datetime


_LIVE_PACKS = text(
    "SELECT l.source_type, l.source_ref, l.credits, l.expires_at FROM credit_ledger l "
    "WHERE l.workspace_id = CAST(:ws AS uuid) AND l.kind = 'topup' AND l.bucket = 'topup' "
    "AND l.credits > 0 AND l.expires_at IS NOT NULL AND NOT EXISTS ("
    "  SELECT 1 FROM credit_ledger e WHERE e.source_type = 'topup_expire' "
    "  AND e.source_ref = l.source_type || ':' || l.source_ref) "
    "ORDER BY l.expires_at, l.id"
)


def _left_per_pack(packs: list[_Pack], balance: int) -> list[int]:
    """What each pack (oldest first) still holds when spending took the oldest first. Pure."""
    left = [0] * len(packs)
    remaining = max(0, balance)
    for index in range(len(packs) - 1, -1, -1):
        left[index] = min(packs[index].credits, remaining)
        remaining -= left[index]
    return left


def expire_topups(db: Any, workspace_id: Any, now: datetime) -> int:
    """Expire the workspace's top-up packs past their date. Does not commit."""
    balance = lock_balance(db, workspace_id).topup
    packs = [_Pack(*row) for row in db.execute(_LIVE_PACKS, {"ws": str(workspace_id)}).fetchall()]
    expired = 0
    for pack, left in zip(packs, _left_per_pack(packs, balance)):
        if pack.expires_at > now:
            continue
        record(db, workspace_id, Entry(
            kind=KIND_EXPIRE, source_type=SOURCE_TOPUP_EXPIRE, source_ref=f"{pack.source_type}:{pack.source_ref}",
            credits=-left, bucket=BUCKET_TOPUP, actor="system", reason="top-up expired",
        ))
        expired += left
    return expired


_DUE_MONTHS = text(
    "SELECT workspace_id, period_end FROM credit_balances "
    "WHERE period_end IS NOT NULL AND period_end < :cutoff AND monthly > 0"
)
_DUE_TOPUPS = text(
    "SELECT DISTINCT l.workspace_id FROM credit_ledger l WHERE l.kind = 'topup' AND l.bucket = 'topup' "
    "AND l.credits > 0 AND l.expires_at <= :now AND NOT EXISTS ("
    "  SELECT 1 FROM credit_ledger e WHERE e.source_type = 'topup_expire' "
    "  AND e.source_ref = l.source_type || ':' || l.source_ref)"
)


def _each(db: Any, label: str, workspace_id: Any, work) -> int:
    try:
        done = work()
        db.commit()
        return done
    except Exception:
        db.rollback()
        logger.exception("[credits] %s expiry for workspace %s failed; the next sweep retries", label, workspace_id)
        return 0


def expire_due(db: Any, now: Optional[datetime] = None) -> dict:
    """The hourly sweep: months past their grace, then top-ups past their date.
    Commits per workspace; one failure never stops the others."""
    from config import config

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=int(config.BILLING_MONTHLY_EXPIRY_GRACE_HOURS))
    monthly = sum(_each(db, "monthly", ws, lambda ws=ws, end=end: expire_monthly(db, ws, end))
                  for ws, end in db.execute(_DUE_MONTHS, {"cutoff": cutoff}).fetchall())
    topups = sum(_each(db, "top-up", ws, lambda ws=ws: expire_topups(db, ws, now))
                 for (ws,) in db.execute(_DUE_TOPUPS, {"now": now}).fetchall())
    return {"monthly": monthly, "topup": topups}
