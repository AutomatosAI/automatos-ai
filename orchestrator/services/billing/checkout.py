"""Buying: an in-page checkout for a plan or a top-up, and the Business offer (PRD-265 US-009, US-007).

- **Who may buy:** the workspace's owner or admins. While ``BILLING_OPEN`` is false
  (the beta) only platform admins can check out, so Stripe's test mode and the
  live card tests run before customers can pay.
- **A plan** (Own keys or Pro) opens an in-page checkout session: the Billing tab
  mounts Stripe's card fields in our own form, so the customer never leaves Studio.
  A workspace that already subscribes changes plan instead (``manage.py``), so it
  never ends up with two subscriptions.
- **Business** is never self-serve: a platform admin *offers* it to one workspace
  (:func:`offer_business`) for ``BUSINESS_OFFER_DAYS``; while the offer stands, that
  workspace's owner buys it on the same in-page checkout. No link leaves Studio.
- **A top-up** needs a live subscription on a plan whose credits pay for platform
  AI (Pro, Business): on Own keys a top-up could never be spent.
- **Terms:** when counsel's consent text is set (D-4), the buyer must tick our box;
  the acceptance time is stamped on the session (``metadata.terms_accepted_at``).

The workspace's customer is created once, before its first checkout, so every
webhook can find the workspace by customer.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from services.billing.accounts import account_for_workspace, has_live_subscription, link_account
from services.billing.catalog import price_for_plan, topup_price
from services.billing.provider import KIND_PLAN, KIND_TOPUP, CheckoutRequest
from services.billing.stripe_events import KIND_TOPUP as META_TOPUP
from services.billing.stripe_events import META_KIND, META_QUANTITY, META_WORKSPACE

BILLING_TAB_PATH = "/settings?tab=billing"
BUSINESS_PLAN = "business"
TERMS_ACCEPTED_META = "terms_accepted_at"


class BillingRefused(Exception):
    """A purchase or change the rules don't allow. ``status`` is the HTTP status to
    answer with; the message is a plain sentence for the customer."""

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


def publishable_key() -> Optional[str]:
    from config import config

    return config.STRIPE_PUBLISHABLE_KEY or None


def return_url() -> str:
    """Where a redirect-based payment method comes back to (cards never leave)."""
    from config import config

    return config.FRONTEND_URL.rstrip("/") + BILLING_TAB_PATH + "&checkout=return"


def business_offered(account: Any, now: Optional[datetime] = None) -> bool:
    until = getattr(account, "business_offer_until", None) if account is not None else None
    return until is not None and until > (now or datetime.now(timezone.utc))


def ensure_customer(db: Any, provider: Any, workspace: Any, email: Optional[str]) -> Any:
    """The workspace's billing account, its provider customer created the first time."""
    account = account_for_workspace(db, workspace.id)
    if account is not None:
        return account
    customer_id = provider.ensure_customer(workspace_id=str(workspace.id), name=workspace.name, email=email)
    account = link_account(db, workspace.id, customer_id, provider=provider.name)
    db.commit()
    return account


def _require_terms(accepted: bool) -> dict:
    from config import config

    if not config.BILLING_WITHDRAWAL_CONSENT_TEXT:
        return {}
    if not accepted:
        raise BillingRefused(422, "Tick the box to accept the terms.")
    return {TERMS_ACCEPTED_META: datetime.now(timezone.utc).isoformat()}


def _session(provider: Any, workspace: Any, customer_id: str, *, kind: str, price_id: str,
             plan: Optional[str], terms: dict) -> dict:
    meta = {META_WORKSPACE: str(workspace.id), META_KIND: META_TOPUP if kind == KIND_TOPUP else KIND_PLAN, **terms}
    if plan:
        meta["plan"] = plan
    if kind == KIND_TOPUP:
        meta[META_QUANTITY] = "1"
    session = provider.create_checkout(CheckoutRequest(
        workspace_id=str(workspace.id), customer_id=customer_id, kind=kind, price_id=price_id,
        return_url=return_url(), metadata=meta))
    return {"client_secret": session.client_secret, "publishable_key": publishable_key()}


def start_plan_checkout(db: Any, provider: Any, workspace: Any, *, plan: str, email: Optional[str],
                        accept_terms: bool = False) -> dict:
    """An in-page checkout session for ``plan``."""
    from services.plan_tiers import is_self_serve

    account = account_for_workspace(db, workspace.id)
    if not (is_self_serve(plan) or (plan == BUSINESS_PLAN and business_offered(account))):
        raise BillingRefused(400, "That plan isn't sold here. Business is by invite: contact Automatos.")
    if has_live_subscription(account):
        raise BillingRefused(409, "You already have a plan — change it below.")
    terms = _require_terms(accept_terms)
    price_id = price_for_plan(plan)
    account = ensure_customer(db, provider, workspace, email)
    return _session(provider, workspace, account.customer_id, kind=KIND_PLAN, price_id=price_id,
                    plan=plan, terms=terms)


def topup_available(account: Any, plan: Optional[str]) -> bool:
    from services.plan_tiers import platform_ai_allowed

    return has_live_subscription(account) and platform_ai_allowed(plan)


def start_topup_checkout(db: Any, provider: Any, workspace: Any, *, accept_terms: bool = False) -> dict:
    account = account_for_workspace(db, workspace.id)
    if not topup_available(account, workspace.plan):
        raise BillingRefused(409, "Top-ups are for Pro and Business subscriptions. Choose a plan first.")
    terms = _require_terms(accept_terms)
    return _session(provider, workspace, account.customer_id, kind=KIND_TOPUP, price_id=topup_price(),
                    plan=None, terms=terms)


def offer_business(db: Any, provider: Any, workspace: Any) -> datetime:
    """Let this workspace buy Business on its Billing tab for ``BUSINESS_OFFER_DAYS``."""
    from config import config

    if has_live_subscription(account_for_workspace(db, workspace.id)):
        raise BillingRefused(409, "This workspace already has a subscription; change its plan instead.")
    price_for_plan(BUSINESS_PLAN)  # refuse an offer nobody could take up
    account = ensure_customer(db, provider, workspace, None)
    until = datetime.now(timezone.utc) + timedelta(days=int(config.BUSINESS_OFFER_DAYS))
    account.business_offer_until = until
    db.commit()
    return until
