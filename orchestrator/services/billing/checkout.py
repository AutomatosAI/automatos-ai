"""Buying: checkout for a plan or a top-up, the portal, the Business invite (PRD-265 US-009, US-007).

- **Who may buy:** the workspace's owner or admins. While ``BILLING_OPEN`` is false
  (the beta) only platform admins can check out, so Stripe's test mode and the
  live card tests run before customers can pay.
- **A plan** (Own keys or Pro) opens Stripe Checkout; a workspace that already has
  a live subscription changes plan through the portal's confirm page instead, so
  it never ends up with two subscriptions. Business is never self-serve: a
  platform admin issues a one-workspace checkout link (:func:`business_invite`).
- **A top-up** needs a live subscription on a plan whose credits pay for platform
  AI (Pro, Business): on Own keys a top-up could never be spent.

The workspace's customer is created once, before its first checkout, so every
webhook can find the workspace by customer.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from services.billing.accounts import account_for_workspace, has_live_subscription, link_account
from services.billing.catalog import price_for_plan, topup_price
from services.billing.provider import KIND_PLAN, KIND_TOPUP, CheckoutRequest, PlanChange
from services.billing.stripe_events import KIND_TOPUP as META_TOPUP
from services.billing.stripe_events import META_KIND, META_QUANTITY, META_WORKSPACE

BILLING_TAB_PATH = "/settings?tab=billing"
# Stripe keeps a Checkout Session open for at most 24 hours.
INVITE_VALID_HOURS = 24
INVITE_SAFETY_MINUTES = 5


class BillingRefused(Exception):
    """A purchase the rules don't allow. ``status`` is the HTTP status to answer
    with; the message is a plain sentence for the customer."""

    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


def checkout_open_for(user: Any) -> bool:
    """Paid plans are on sale, or the caller is a platform admin testing them."""
    from config import config
    from core.auth.roles import caller_is_admin

    return bool(config.BILLING_OPEN) or bool(user and caller_is_admin(user))


def is_configured() -> bool:
    from config import config

    return bool(config.STRIPE_SECRET_KEY)


def _urls() -> dict:
    from config import config

    base = config.FRONTEND_URL.rstrip("/") + BILLING_TAB_PATH
    return {"success": f"{base}&checkout=success", "cancel": f"{base}&checkout=cancelled", "return": base}


def _customer(db: Any, provider: Any, workspace: Any, email: Optional[str]) -> str:
    account = account_for_workspace(db, workspace.id)
    if account is not None:
        return account.customer_id
    customer_id = provider.ensure_customer(workspace_id=str(workspace.id), name=workspace.name, email=email)
    link_account(db, workspace.id, customer_id, provider=provider.name)
    db.commit()
    return customer_id


def _request(workspace: Any, customer_id: str, *, kind: str, price_id: str, plan: Optional[str],
             expires_at: Optional[datetime] = None) -> CheckoutRequest:
    from config import config

    urls = _urls()
    meta = {META_WORKSPACE: str(workspace.id), META_KIND: META_TOPUP if kind == KIND_TOPUP else KIND_PLAN}
    if plan:
        meta["plan"] = plan
    if kind == KIND_TOPUP:
        meta[META_QUANTITY] = "1"
    return CheckoutRequest(workspace_id=str(workspace.id), customer_id=customer_id, kind=kind, price_id=price_id,
                           success_url=urls["success"], cancel_url=urls["cancel"], metadata=meta,
                           consent_text=config.BILLING_WITHDRAWAL_CONSENT_TEXT, expires_at=expires_at)


def start_plan_checkout(db: Any, provider: Any, workspace: Any, *, plan: str, email: Optional[str]) -> dict:
    """Where the owner goes to buy ``plan``: Stripe Checkout, or the portal's
    plan-change page for a workspace that already subscribes."""
    from services.plan_tiers import is_self_serve

    if not is_self_serve(plan):
        raise BillingRefused(400, "That plan isn't sold here. Business is by invite: contact Automatos.")
    account = account_for_workspace(db, workspace.id)
    if has_live_subscription(account):
        if account.plan == plan:
            raise BillingRefused(409, "You're already on this plan.")
        change = PlanChange(subscription_id=account.subscription_id, price_id=price_for_plan(plan))
        url = provider.portal_link(customer_id=account.customer_id, return_url=_urls()["return"], change=change)
        return {"url": url, "flow": "portal"}
    price_id = price_for_plan(plan)
    customer_id = _customer(db, provider, workspace, email)
    link = provider.create_checkout(_request(workspace, customer_id, kind=KIND_PLAN, price_id=price_id, plan=plan))
    return {"url": link.url, "flow": "checkout"}


def topup_available(account: Any, plan: Optional[str]) -> bool:
    from services.plan_tiers import platform_ai_allowed

    return has_live_subscription(account) and platform_ai_allowed(plan)


def start_topup_checkout(db: Any, provider: Any, workspace: Any, *, email: Optional[str]) -> dict:
    account = account_for_workspace(db, workspace.id)
    if not topup_available(account, workspace.plan):
        raise BillingRefused(409, "Top-ups are for Pro and Business subscriptions. Choose a plan first.")
    link = provider.create_checkout(_request(workspace, account.customer_id, kind=KIND_TOPUP,
                                             price_id=topup_price(), plan=None))
    return {"url": link.url, "flow": "checkout"}


def open_portal(db: Any, provider: Any, workspace: Any) -> str:
    account = account_for_workspace(db, workspace.id)
    if account is None:
        raise BillingRefused(404, "There is nothing to manage yet: no plan has been bought here.")
    return provider.portal_link(customer_id=account.customer_id, return_url=_urls()["return"])


def business_invite(db: Any, provider: Any, workspace: Any) -> dict:
    """A Business checkout link for one workspace, valid just under 24 hours."""
    if has_live_subscription(account_for_workspace(db, workspace.id)):
        raise BillingRefused(409, "This workspace already has a subscription; change its plan in Stripe.")
    price_id = price_for_plan("business")
    customer_id = _customer(db, provider, workspace, None)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=INVITE_VALID_HOURS, minutes=-INVITE_SAFETY_MINUTES)
    link = provider.create_checkout(_request(workspace, customer_id, kind=KIND_PLAN, price_id=price_id,
                                             plan="business", expires_at=expires_at))
    return {"url": link.url, "expires_at": (link.expires_at or expires_at).isoformat()}
