"""What a workspace's members see of its credits (PRD-265 US-008).

``billing_summary`` is the header balance and the paused banner's source: plan,
status, balance by bucket, and whether platform-funded work is paused.
``ledger_page`` is the owner's statement: the ledger, newest first (one line per
metered call), plus the last 30 days' credits by model.

The platform's own cost (``actual_cost_usd``) and margin never leave the server:
customers see credits, not our prices.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import text

from services.credits.ledger import read_balance
from services.credits.routing import (
    FUNDED_BY_CREDITS,
    funding_for,
    has_own_keys,
)

USAGE_WINDOW_DAYS = 30
# What the ledger shows a customer for each movement's source.
_SOURCE_LABELS = {
    "usage": "AI usage",
    "trial_grant": "Trial credits",
    "trial_reset": "Trial reset",
    "admin_adjust": "Adjustment by Automatos",
}
_HIDDEN_SOURCES = ("overhead_day",)


def billing_summary(db: Any, workspace: Any) -> dict:
    """Plan, status, renewal date and balance by bucket for one workspace."""
    from config import config
    from services.credits.alerts import LEVEL_LOW, LEVEL_OUT, level_for
    from services.plan_tiers import exposure_for_plan, platform_ai_allowed

    plan = getattr(workspace, "plan", None)
    balance = read_balance(db, workspace.id)
    funding = funding_for(plan, balance, edition=config.AUTH_EDITION)
    metered = funding.funded_by == FUNDED_BY_CREDITS
    level = level_for(funding.spendable, plan) if metered else None
    paused = level == LEVEL_OUT
    exposure = exposure_for_plan(plan or "basic")
    return {
        "plan": plan,
        "plan_name": exposure.get("display_name"),
        "price_usd": exposure.get("display_price_usd"),
        "price_label": exposure.get("price_label"),
        "billing_open": bool(config.BILLING_OPEN),
        "status": "paused" if paused else "active",
        "renewal_date": balance.period_end.isoformat() if balance.period_end else None,
        "balance": {**balance.as_dict(), "spendable": funding.spendable if metered else None},
        "trial_credits": int(config.TRIAL_CREDITS),
        "platform_ai": platform_ai_allowed(plan),
        "metered": metered,
        "paused": paused,
        "low": level == LEVEL_LOW,
        "own_keys": has_own_keys(workspace),
    }


def _ledger_rows(db: Any, workspace_id: Any, page: int, page_size: int) -> tuple[list, int]:
    params = {"ws": str(workspace_id), "hidden": list(_HIDDEN_SOURCES)}
    total = db.execute(text(
        "SELECT count(*) FROM credit_ledger WHERE workspace_id = CAST(:ws AS uuid) "
        "AND NOT (source_type = ANY(:hidden)) AND credits <> 0"
    ), params).scalar() or 0
    rows = db.execute(text(
        "SELECT id, created_at, kind, bucket, credits, source_type, band, reason FROM credit_ledger "
        "WHERE workspace_id = CAST(:ws AS uuid) AND NOT (source_type = ANY(:hidden)) AND credits <> 0 "
        "ORDER BY id DESC LIMIT :limit OFFSET :offset"
    ), {**params, "limit": page_size, "offset": (page - 1) * page_size}).fetchall()
    items = [{
        "id": r[0], "at": r[1].isoformat() if r[1] else None, "kind": r[2], "bucket": r[3],
        "credits": int(r[4]), "what": _SOURCE_LABELS.get(r[5], r[5]),
        "model": r[7] if r[5] == "usage" else None,  # a metered call's reason is its model
    } for r in rows]
    return items, int(total)


def _usage(db: Any, workspace_id: Any, since: datetime) -> dict:
    by_model = db.execute(text(
        "SELECT reason, count(DISTINCT source_ref), COALESCE(-SUM(credits), 0) FROM credit_ledger "
        "WHERE workspace_id = CAST(:ws AS uuid) AND source_type = 'usage' AND created_at >= :since "
        "GROUP BY reason ORDER BY 3 DESC LIMIT 20"
    ), {"ws": str(workspace_id), "since": since}).fetchall()
    return {
        "since": since.isoformat(),
        "by_model": [{"model": r[0], "calls": int(r[1]), "credits": int(r[2])} for r in by_model],
    }


def ledger_page(db: Any, workspace_id: Any, *, page: int, page_size: int, now: Optional[datetime] = None) -> dict:
    """One page of the ledger, newest first, and the last 30 days' usage."""
    since = (now or datetime.now(timezone.utc)) - timedelta(days=USAGE_WINDOW_DAYS)
    items, total = _ledger_rows(db, workspace_id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size,
            "usage": _usage(db, workspace_id, since)}
