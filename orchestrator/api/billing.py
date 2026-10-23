"""Billing for a workspace's members (PRD-265 US-008).

- ``GET /api/v1/billing/summary`` — any member: plan, status, renewal date,
  balance by bucket, and whether platform-funded work is paused. The header
  balance and the paused banner read it.
- ``GET /api/v1/billing/ledger`` — the workspace's owner or admins: the credit
  ledger, newest first (a line per metered call), and the last 30 days' credits by model.
- ``GET /api/v1/billing/plans`` — any member: the plans, the top-up and this
  workspace's subscription, for the Billing tab (PRD-265 US-011).
- ``POST /api/v1/billing/checkout`` — the owner or admins: an in-page checkout
  session (its client secret) the Billing tab mounts Stripe's card fields with; the
  customer never leaves Studio (US-009). While billing is closed only platform
  admins may check out. The card, invoices, cancelling and plan changes are in
  ``api/billing_manage.py``.

Hosted edition only: the local edition never meters, so both answer 404 there.
Plain ``def`` routes: they read a synchronous session (FastAPI runs them in its
threadpool), never ``async def`` over a blocking query.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db
from core.models.workspaces import Workspace

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/billing", tags=["Billing"])

LEDGER_PAGE_SIZE = 50
LEDGER_PAGE_SIZE_MAX = 200


def hosted_workspace(db: Session, ctx: RequestContext) -> Workspace:
    from config import config

    if config.AUTH_EDITION == "local":
        raise HTTPException(status_code=404, detail="Not found")
    if not ctx.workspace_id:
        raise HTTPException(status_code=400, detail="Workspace context required")
    workspace = db.query(Workspace).filter(Workspace.id == ctx.workspace_id).first()
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


@router.get("/summary")
def get_billing_summary(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Plan, status, renewal date and credits by bucket for the caller's workspace."""
    from services.credits.summary import billing_summary

    return billing_summary(db, hosted_workspace(db, ctx))


@router.get("/ledger")
def get_billing_ledger(
    page: int = Query(1, ge=1),
    page_size: int = Query(LEDGER_PAGE_SIZE, ge=1, le=LEDGER_PAGE_SIZE_MAX),
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The workspace's credit ledger, newest first, with the last 30 days' credits by model."""
    from services.credits.summary import ledger_page

    workspace = hosted_workspace(db, ctx)
    return ledger_page(db, workspace.id, page=page, page_size=page_size)


class CheckoutBody(BaseModel):
    kind: Literal["plan", "topup"]
    plan: Optional[str] = Field(None, min_length=1, max_length=32)
    # Required true when counsel's consent text is set (the box on our form).
    accept_terms: bool = False


def billing_call(work: Callable[[Any], Any]) -> Any:
    """Run ``work(provider)``, answering in plain words when billing can't."""
    import httpx

    from services.billing.checkout import BillingRefused
    from services.billing.stripe_api import StripeError, StripeNotConfigured
    from services.billing.stripe_provider import stripe_provider

    try:
        return work(stripe_provider())
    except BillingRefused as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    except StripeNotConfigured as exc:
        logger.error("[billing] not configured: %s", exc)
        raise HTTPException(status_code=503, detail="Billing isn't set up yet.") from exc
    except (StripeError, httpx.TransportError) as exc:
        logger.exception("[billing] Stripe refused or was unreachable")
        raise HTTPException(status_code=502, detail="Stripe couldn't be reached. Try again in a minute.") from exc


def _require_open(ctx: RequestContext) -> None:
    from services.billing.checkout import checkout_open_for

    if not checkout_open_for(ctx.user):
        raise HTTPException(status_code=409, detail="Paid plans aren't on sale yet.")


@router.get("/plans")
def get_billing_plans(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The plans, the top-up and this workspace's subscription, for the Billing tab."""
    from core.auth.workspace_admin import may_see_own_workspace_health
    from services.billing.plans import plans_view

    workspace = hosted_workspace(db, ctx)
    return plans_view(db, workspace, user=ctx.user, can_manage=may_see_own_workspace_health(db, ctx))


@router.post("/checkout")
def start_billing_checkout(
    body: CheckoutBody,
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """An in-page checkout session for buying a plan or a top-up."""
    from services.billing.checkout import start_plan_checkout, start_topup_checkout
    from services.plan_tiers import tier_key

    workspace = hosted_workspace(db, ctx)
    _require_open(ctx)
    email, terms = getattr(ctx.user, "email", None), body.accept_terms
    if body.kind == "topup":
        return billing_call(lambda provider: start_topup_checkout(db, provider, workspace, accept_terms=terms))
    if not body.plan:
        raise HTTPException(status_code=422, detail="Name the plan to buy.")
    plan = tier_key(body.plan) or ""
    return billing_call(lambda provider: start_plan_checkout(db, provider, workspace, plan=plan, email=email,
                                                             accept_terms=terms))
