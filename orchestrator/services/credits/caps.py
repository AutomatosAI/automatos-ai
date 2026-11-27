"""A spend cap per product workspace (PRD-274, the proposal in #1520).

Since Wave 5d a product's workspaces spend their owner's credits. A cap limits what one of them may spend in a
calendar month (UTC), so one customer can't spend the owner's whole balance and pause every other one:

- **The cap** (``monthly_cap``): the workspace's own ``settings.product_credit_cap`` (the product's key sets it,
  ``api/products.py``), else its product's ``workspace_monthly_credit_cap`` (an admin sets it), else none: then
  only the owner's balance limits, as before.
- **The count** (``count_spend``): each debit for the workspace adds its credits to the month's row of
  ``product_workspace_spend`` in the debit's own transaction (``payer.debit``), so ``cap_left`` reads one row.
- **The stop** (``cap_left``): the credit gate makes an ask's budget the smaller of the owner's spendable credits
  and the cap left, and refuses with ``WorkspaceCapReachedError`` when none is (``gate.py``); background work and
  an Auto Live call check it before they start. As with the balance, one run overshoots by at most one call, but
  each ask takes the cap left at its first call as its budget, so N asks running at once can each spend it: the
  cap is a monthly limit, not a hard real-time one.
- **The notices** (``check_cap_notices``): the owner is told once a month when a workspace reaches 80% of its cap,
  and once when it reaches it, on its ``credits_low`` / ``credits_exhausted`` destinations, naming the workspace
  by its external id. A changed cap starts them again (``cap_changed``).

Calls on a workspace's own keys are never metered, so never capped.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

SETTING = "product_credit_cap"  # a workspace's own cap, in its settings
NEAR_SHARE = 0.8
LEVEL_NONE, LEVEL_NEAR, LEVEL_REACHED = 0, 1, 2
_MONTH = "CAST(date_trunc('month', now() AT TIME ZONE 'utc') AS date)"
_PRODUCT_CAP = text("SELECT workspace_monthly_credit_cap FROM products WHERE id = CAST(:product AS uuid)")
_SPENT = text(f"SELECT credits, notified_level FROM product_workspace_spend "  # noqa: S608 — a constant, no input
              f"WHERE workspace_id = CAST(:ws AS uuid) AND month = {_MONTH}")
_COUNT = text(f"INSERT INTO product_workspace_spend (workspace_id, month, credits) "  # noqa: S608 — a constant
              f"VALUES (CAST(:ws AS uuid), {_MONTH}, :credits) ON CONFLICT (workspace_id, month) "
              f"DO UPDATE SET credits = product_workspace_spend.credits + EXCLUDED.credits")
# Conditional, so of two meter runs that both see a new level only the one that moves the row sends the notice.
_NOTIFIED = text(f"UPDATE product_workspace_spend SET notified_level = :level "  # noqa: S608 — a constant
                 f"WHERE workspace_id = CAST(:ws AS uuid) AND month = {_MONTH} AND notified_level < :level")
# A changed cap starts the month's notices again, so the owner hears when the workspace reaches the new one.
_RENOTIFY = text(f"UPDATE product_workspace_spend SET notified_level = 0 "  # noqa: S608 — a constant
                 f"WHERE workspace_id = CAST(:ws AS uuid) AND month = {_MONTH}")


def _cap_value(value: Any) -> Optional[int]:
    try:
        cap = int(value)
    except (TypeError, ValueError):
        return None
    return cap if cap >= 1 else None


def monthly_cap(db: Any, workspace: Any) -> Optional[int]:
    """The credits ``workspace`` (a loaded row) may spend this month; None: no cap (or no product's workspace)."""
    from services.credits.payer import _uuid

    settings = getattr(workspace, "settings", None) or {}
    product_id = _uuid(settings.get("product_id"))
    if product_id is None:
        return None
    own = _cap_value(settings.get(SETTING))
    if own is not None:
        return own
    return _cap_value(db.execute(_PRODUCT_CAP, {"product": product_id}).scalar())


def spent_this_month(db: Any, workspace_id: Any) -> int:
    row = db.execute(_SPENT, {"ws": str(workspace_id)}).first()
    return int(row[0]) if row else 0


def cap_left(db: Any, workspace_id: Any) -> Optional[int]:
    """The credits left under the workspace's cap this month (zero or less: reached); None when it has no cap."""
    from core.models.workspaces import Workspace

    workspace = db.query(Workspace).get(workspace_id)
    cap = monthly_cap(db, workspace) if workspace is not None else None
    return None if cap is None else cap - spent_this_month(db, workspace_id)


def under_cap(db: Any, workspace_id: Any) -> bool:
    """Whether work may start: the workspace has no cap, or credits left under it."""
    left = cap_left(db, workspace_id)
    return left is None or left > 0


def cap_changed(db: Any, workspace_id: Any) -> None:
    """A workspace's cap changed: this month's notices start again (no commit)."""
    db.execute(_RENOTIFY, {"ws": str(workspace_id)})


def count_spend(db: Any, workspace_id: Any, credits: int) -> None:
    """Adds a debit's credits to the workspace's month (no commit: the debit's transaction)."""
    if credits > 0:
        db.execute(_COUNT, {"ws": str(workspace_id), "credits": int(credits)})


def _level(spent: int, cap: int) -> int:
    if spent >= cap:
        return LEVEL_REACHED
    return LEVEL_NEAR if spent >= cap * NEAR_SHARE else LEVEL_NONE


def check_cap_notices(db: Any, workspace_id: Any) -> Optional[int]:
    """Tells the owner, once a month each, when the workspace reaches 80% of its cap and its cap. Returns the
    level told, or None. Commits."""
    from core.models.workspaces import Workspace

    workspace = db.query(Workspace).get(workspace_id)
    cap = monthly_cap(db, workspace) if workspace is not None else None
    if cap is None:
        return None
    row = db.execute(_SPENT, {"ws": str(workspace_id)}).first()
    if row is None:
        return None
    level = _level(int(row[0]), cap)
    if level <= int(row[1]):
        return None
    if db.execute(_NOTIFIED, {"ws": str(workspace_id), "level": level}).rowcount == 0:
        db.rollback()  # another run recorded this level first, and tells the owner
        return None
    db.commit()
    _notify_owner(db, workspace, level, int(row[0]), cap)
    return level


def _notify_owner(db: Any, workspace: Any, level: int, spent: int, cap: int) -> None:
    from core.services.notification_dispatcher import NotificationDispatcher
    from services.credits.alerts import EVENT_LOW, EVENT_OUT
    from services.credits.payer import payer_of

    settings = workspace.settings or {}
    who = settings.get("source_external_id") or str(workspace.id)
    if level == LEVEL_REACHED:
        event, title = EVENT_OUT, f"A product workspace reached its monthly cap: {who}"
        message = f"{who} has spent {spent} of its {cap}-credit cap this month; its work on Automatos's AI keys is paused."
    else:
        event, title = EVENT_LOW, f"A product workspace is near its monthly cap: {who}"
        message = f"{who} has spent {spent} of its {cap}-credit cap this month."
    try:
        asyncio.run(NotificationDispatcher(db, payer_of(db, workspace)).dispatch(
            event_type=event, title=title, message=message, status="warning"))
    except Exception:  # noqa: BLE001 — a notice that can't be sent never stops the meter
        logger.exception("[credits] cap notice for workspace %s not sent", workspace.id)
