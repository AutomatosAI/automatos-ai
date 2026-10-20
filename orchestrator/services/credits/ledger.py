"""The credit ledger's one writer (PRD-265 US-002).

Every credit movement goes through :func:`record`. In one transaction it:

1. locks the workspace's ``credit_balances`` row (creating it at zero first), so
   two writers for the same workspace queue instead of racing;
2. checks whether the source event (``source_type`` + ``source_ref``) is already
   in the ledger; if it is, nothing changes (idempotent per source event);
3. splits the movement into per-bucket rows (:func:`allocate`) and inserts them;
4. moves the balance columns by the same amounts.

It never commits: the caller owns the transaction, so a movement lands together
with whatever caused it. The unique key ``(source_type, source_ref, bucket)``
backs step 2 for a source event that two workspaces could claim at once (a trial
grant keyed on its owner): the losing insert rolls back to a savepoint and
:func:`record` reports it as already recorded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Iterable, Optional

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)

KIND_GRANT = "grant"
KIND_TOPUP = "topup"
KIND_DEBIT = "debit"
KIND_REFUND = "refund"
KIND_ADJUST = "adjust"
KIND_EXPIRE = "expire"

BUCKET_TRIAL = "trial"
BUCKET_MONTHLY = "monthly"
BUCKET_TOPUP = "topup"
BUCKET_OVERHEAD = "overhead"  # absorbed platform cost: carries no credits

# PRD-265 §4.3: trial is spent first, then the month's credits, then top-ups.
SPEND_ORDER = (BUCKET_TRIAL, BUCKET_MONTHLY, BUCKET_TOPUP)
BALANCE_BUCKETS = SPEND_ORDER


@dataclass(frozen=True)
class Balance:
    """A workspace's credits by bucket. Any bucket may be negative."""

    trial: int = 0
    monthly: int = 0
    topup: int = 0
    period_end: Optional[datetime] = None

    def of(self, bucket: str) -> int:
        return int(getattr(self, bucket, 0) or 0)

    def spendable(self, buckets: Iterable[str] = SPEND_ORDER) -> int:
        """Credits the given buckets can still pay (a negative bucket pays nothing)."""
        return sum(max(0, self.of(b)) for b in buckets)

    def as_dict(self) -> dict:
        return {"trial": self.trial, "monthly": self.monthly, "topup": self.topup}


@dataclass(frozen=True)
class Entry:
    """One credit movement to record.

    ``credits`` is signed: positive adds, negative takes. A movement with no
    ``bucket`` must be a take (a debit): :func:`allocate` spreads it across the
    spendable buckets in spend order. Every other movement names its bucket, or
    gives its exact ``split`` (``(bucket, credits)`` pairs) — a refund returns
    credits to the buckets its debit took them from.
    """

    kind: str
    source_type: str
    source_ref: str
    credits: int
    bucket: Optional[str] = None
    actor: Optional[str] = None
    reason: Optional[str] = None
    band: Optional[str] = None
    charged_credits: Optional[int] = None
    actual_cost_usd: Optional[float] = None
    expires_at: Optional[datetime] = None
    split: tuple = ()


def allocate(balance: Balance, entry: Entry, spend_buckets: Iterable[str] = SPEND_ORDER) -> list[tuple[str, int]]:
    """The ``(bucket, credits)`` rows ``entry`` writes against ``balance``. Pure.

    A take with no named bucket draws from ``spend_buckets`` in order, each only
    down to zero; whatever is left over (an overdraft) lands on the last of them,
    which then goes negative and pauses spending until it is paid back.
    """
    if entry.split:
        return [(bucket, int(credits)) for bucket, credits in entry.split]
    if entry.bucket is not None:
        return [(entry.bucket, int(entry.credits))]
    if entry.credits >= 0:
        raise ValueError(f"a credit movement of {entry.credits} must name its bucket")
    buckets = tuple(spend_buckets) or (BUCKET_TRIAL,)
    remaining = -int(entry.credits)
    rows: list[tuple[str, int]] = []
    for bucket in buckets:
        take = min(max(0, balance.of(bucket)), remaining)
        if take:
            rows.append((bucket, -take))
            remaining -= take
    if remaining:
        rows = _add_overdraft(rows, buckets[-1], remaining)
    return rows


def _add_overdraft(rows: list[tuple[str, int]], bucket: str, amount: int) -> list[tuple[str, int]]:
    merged = [(b, c - amount) if b == bucket else (b, c) for b, c in rows]
    if any(b == bucket for b, _ in rows):
        return merged
    return merged + [(bucket, -amount)]


def _lock_balance(db: Any, workspace_id: Any) -> Balance:
    db.execute(
        text("INSERT INTO credit_balances (workspace_id) VALUES (CAST(:ws AS uuid)) "
             "ON CONFLICT (workspace_id) DO NOTHING"),
        {"ws": str(workspace_id)},
    )
    row = db.execute(
        text("SELECT trial, monthly, topup, period_end FROM credit_balances "
             "WHERE workspace_id = CAST(:ws AS uuid) FOR UPDATE"),
        {"ws": str(workspace_id)},
    ).fetchone()
    return Balance(trial=row[0], monthly=row[1], topup=row[2], period_end=row[3])


def read_balance(db: Any, workspace_id: Any) -> Balance:
    """The workspace's balance (all zero when it has never had a movement)."""
    row = db.execute(
        text("SELECT trial, monthly, topup, period_end FROM credit_balances "
             "WHERE workspace_id = CAST(:ws AS uuid)"),
        {"ws": str(workspace_id)},
    ).fetchone()
    if row is None:
        return Balance()
    return Balance(trial=row[0], monthly=row[1], topup=row[2], period_end=row[3])


def is_recorded(db: Any, source_type: str, source_ref: str) -> bool:
    """True when any row of this source event is already in the ledger."""
    row = db.execute(
        text("SELECT 1 FROM credit_ledger WHERE source_type = :t AND source_ref = :r LIMIT 1"),
        {"t": source_type, "r": source_ref},
    ).fetchone()
    return row is not None


def entry_rows(db: Any, source_type: str, source_ref: str) -> list[tuple[str, int]]:
    """The ``(bucket, credits)`` rows a recorded source event wrote, oldest first."""
    rows = db.execute(
        text("SELECT bucket, credits FROM credit_ledger WHERE source_type = :t AND source_ref = :r "
             "ORDER BY id"),
        {"t": source_type, "r": source_ref},
    ).fetchall()
    return [(r[0], int(r[1])) for r in rows]


_INSERT = text(
    "INSERT INTO credit_ledger (workspace_id, kind, bucket, credits, expires_at, source_type, "
    "source_ref, actor, reason, band, charged_credits, actual_cost_usd) VALUES "
    "(CAST(:ws AS uuid), :kind, :bucket, :credits, :expires_at, :source_type, :source_ref, "
    ":actor, :reason, :band, :charged_credits, :actual_cost_usd)"
)


def _insert_rows(db: Any, workspace_id: Any, entry: Entry, rows: list[tuple[str, int]]) -> None:
    """One row per bucket. The event's totals (``charged_credits``,
    ``actual_cost_usd``) ride the first row only, so summing the ledger never
    counts them twice."""
    cost = Decimal(str(entry.actual_cost_usd)) if entry.actual_cost_usd is not None else None
    for index, (bucket, credits) in enumerate(rows):
        first = index == 0
        db.execute(_INSERT, {
            "ws": str(workspace_id), "kind": entry.kind, "bucket": bucket, "credits": credits,
            "expires_at": entry.expires_at, "source_type": entry.source_type,
            "source_ref": entry.source_ref, "actor": entry.actor, "reason": entry.reason,
            "band": entry.band, "charged_credits": entry.charged_credits if first else None,
            "actual_cost_usd": cost if first else None,
        })


def _move_balance(db: Any, workspace_id: Any, rows: list[tuple[str, int]]) -> None:
    delta = {b: 0 for b in BALANCE_BUCKETS}
    for bucket, credits in rows:
        if bucket in delta:
            delta[bucket] += credits
    if not any(delta.values()):
        return
    db.execute(
        text("UPDATE credit_balances SET trial = trial + :trial, monthly = monthly + :monthly, "
             "topup = topup + :topup, updated_at = now() WHERE workspace_id = CAST(:ws AS uuid)"),
        {"ws": str(workspace_id), **delta},
    )


def record(
    db: Any, workspace_id: Any, entry: Entry, *, spend_buckets: Iterable[str] = SPEND_ORDER,
) -> bool:
    """Record ``entry`` for ``workspace_id`` once. Returns False when its source
    event was already recorded (nothing changed). Does not commit."""
    balance = _lock_balance(db, workspace_id)
    if is_recorded(db, entry.source_type, entry.source_ref):
        return False
    rows = allocate(balance, entry, spend_buckets)
    savepoint = db.begin_nested()
    try:
        _insert_rows(db, workspace_id, entry, rows)
        _move_balance(db, workspace_id, rows)
    except IntegrityError:
        savepoint.rollback()
        logger.info("[credits] %s:%s was recorded concurrently; nothing changed",
                    entry.source_type, entry.source_ref)
        return False
    savepoint.commit()
    return True
