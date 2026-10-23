"""A workspace's billing account: its customer and subscription (PRD-265 US-010)."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text

from core.models.billing import BillingAccount

logger = logging.getLogger(__name__)

# A subscription in these states still pays for its plan (past_due: Stripe is
# retrying the card). Anything else — canceled, unpaid, incomplete_expired — has
# ended, and the workspace goes back to the entry plan.
ACTIVE_STATUSES = frozenset({"active", "trialing", "past_due"})
ENDED_STATUSES = frozenset({"canceled", "unpaid", "incomplete_expired"})


def account_for_workspace(db: Any, workspace_id: Any) -> Optional[BillingAccount]:
    return db.query(BillingAccount).filter(BillingAccount.workspace_id == workspace_id).first()


def account_for_customer(db: Any, customer_id: Optional[str]) -> Optional[BillingAccount]:
    if not customer_id:
        return None
    return db.query(BillingAccount).filter(BillingAccount.customer_id == customer_id).first()


def has_live_subscription(account: Optional[BillingAccount]) -> bool:
    return bool(account and account.subscription_id and account.subscription_status in ACTIVE_STATUSES)


def link_account(db: Any, workspace_id: Any, customer_id: str, *, provider: str) -> BillingAccount:
    """The workspace's account, created for ``customer_id`` if it has none (a
    second writer for the same workspace changes nothing). Does not commit."""
    from config import config

    db.execute(text(
        "INSERT INTO billing_accounts (workspace_id, provider, provider_account, customer_id) "
        "VALUES (CAST(:ws AS uuid), :provider, :account, :customer) ON CONFLICT DO NOTHING"
    ), {"ws": str(workspace_id), "provider": provider, "account": config.STRIPE_ACCOUNT_ID or None,
        "customer": customer_id})
    db.flush()
    return account_for_workspace(db, workspace_id)


def _workspace_id(ref: Optional[str]) -> Optional[uuid.UUID]:
    try:
        return uuid.UUID(str(ref)) if ref else None
    except ValueError:
        return None


def resolve_account(db: Any, customer_id: Optional[str], workspace_ref: Optional[str], *,
                    provider: str) -> Optional[BillingAccount]:
    """The account an event belongs to: by its customer, else by the workspace our
    checkout stamped on it (then linked to that customer). None when neither holds."""
    account = account_for_customer(db, customer_id)
    if account is not None:
        return account
    workspace_id = _workspace_id(workspace_ref)
    if workspace_id is None or not customer_id:
        return None
    existing = account_for_workspace(db, workspace_id)
    if existing is not None:
        logger.warning("[billing] workspace %s already has customer %s; event names %s",
                       workspace_id, existing.customer_id, customer_id)
        return existing
    return link_account(db, workspace_id, customer_id, provider=provider)
