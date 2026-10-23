"""Managing a subscription inside Studio (PRD-265 W2): card, invoices, cancel, plan change.

The owner or an admin does all of it on Settings → Billing; nothing redirects to a
Stripe page (Gerard, 23 Oct 2026). Every route is ``require_workspace_admin`` and
hosted-only (404 in the local edition):

- ``GET  /api/v1/billing/payment-method`` — the card on file.
- ``POST /api/v1/billing/payment-method/setup`` — a setup session for a new card,
  entered in our tab with Stripe's card fields; ``POST /api/v1/billing/payment-method``
  makes the finished setup's card the default.
- ``GET  /api/v1/billing/invoices`` and ``GET /api/v1/billing/invoices/{id}/pdf`` —
  the invoices, and each PDF served by us.
- ``POST /api/v1/billing/subscription/cancel`` / ``…/resume`` — end at the period end, or not.
- ``POST /api/v1/billing/plan-change/preview``, ``POST /api/v1/billing/plan-change`` and
  ``POST /api/v1/billing/plan-change/cancel`` — upgrade now (prorated) or downgrade at
  the period end, and undo a scheduled downgrade.

Plain ``def`` routes over the synchronous session (FastAPI's threadpool).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.billing import billing_call, hosted_workspace
from core.auth.dependencies import RequestContext
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db

router = APIRouter(prefix="/api/v1/billing", tags=["Billing"])


class SetupDoneBody(BaseModel):
    setup_intent_id: str = Field(..., min_length=6, max_length=255, pattern=r"^seti_[A-Za-z0-9]+$")


class PlanBody(BaseModel):
    plan: str = Field(..., min_length=1, max_length=32)


def _plan(body: PlanBody) -> str:
    from services.plan_tiers import tier_key

    plan = tier_key(body.plan)
    if not plan:
        raise HTTPException(status_code=422, detail="Name the plan.")
    return plan


def _email(ctx: RequestContext) -> Optional[str]:
    return getattr(ctx.user, "email", None)


@router.get("/payment-method")
def get_payment_method(ctx: RequestContext = Depends(require_workspace_admin),
                       db: Session = Depends(get_db)) -> Dict[str, Any]:
    """The card the plan is charged on, or none."""
    from services.billing.manage import card_view

    workspace = hosted_workspace(db, ctx)
    return billing_call(lambda provider: card_view(db, provider, workspace))


@router.post("/payment-method/setup")
def start_payment_method_setup(ctx: RequestContext = Depends(require_workspace_admin),
                               db: Session = Depends(get_db)) -> Dict[str, Any]:
    """A setup session for entering a new card in our tab."""
    from services.billing.manage import start_card_change

    workspace = hosted_workspace(db, ctx)
    return billing_call(lambda provider: start_card_change(db, provider, workspace, email=_email(ctx)))


@router.post("/payment-method")
def save_payment_method(body: SetupDoneBody, ctx: RequestContext = Depends(require_workspace_admin),
                        db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Make the card a finished setup saved the default for the plan."""
    from services.billing.manage import finish_card_change

    workspace = hosted_workspace(db, ctx)
    return billing_call(lambda provider: finish_card_change(db, provider, workspace, setup_id=body.setup_intent_id))


@router.get("/invoices")
def list_invoices(ctx: RequestContext = Depends(require_workspace_admin),
                  db: Session = Depends(get_db)) -> Dict[str, Any]:
    """The workspace's invoices, newest first."""
    from services.billing.manage import invoices_view

    workspace = hosted_workspace(db, ctx)
    return billing_call(lambda provider: invoices_view(db, provider, workspace))


@router.get("/invoices/{invoice_id}/pdf")
def download_invoice(invoice_id: str, ctx: RequestContext = Depends(require_workspace_admin),
                     db: Session = Depends(get_db)) -> Response:
    """One invoice as a PDF, served by us (the browser never visits Stripe)."""
    from services.billing.manage import invoice_pdf

    workspace = hosted_workspace(db, ctx)
    content, name = billing_call(lambda provider: invoice_pdf(db, provider, workspace, invoice_id=invoice_id))
    return Response(content=content, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.post("/subscription/cancel")
def cancel_subscription(ctx: RequestContext = Depends(require_workspace_admin),
                        db: Session = Depends(get_db)) -> Dict[str, Any]:
    """End the plan at the period end."""
    from services.billing.manage import cancel_plan

    workspace = hosted_workspace(db, ctx)
    return billing_call(lambda provider: cancel_plan(db, provider, workspace))


@router.post("/subscription/resume")
def resume_subscription(ctx: RequestContext = Depends(require_workspace_admin),
                        db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Keep the plan after all."""
    from services.billing.manage import resume_plan

    workspace = hosted_workspace(db, ctx)
    return billing_call(lambda provider: resume_plan(db, provider, workspace))


@router.post("/plan-change/preview")
def preview_plan_change(body: PlanBody, ctx: RequestContext = Depends(require_workspace_admin),
                        db: Session = Depends(get_db)) -> Dict[str, Any]:
    """What a plan change costs now and when it takes effect."""
    from services.billing.manage import preview_change

    workspace, plan = hosted_workspace(db, ctx), _plan(body)
    return billing_call(lambda provider: preview_change(db, provider, workspace, plan=plan))


@router.post("/plan-change")
def change_plan(body: PlanBody, ctx: RequestContext = Depends(require_workspace_admin),
                db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Upgrade now (prorated) or downgrade at the period end."""
    from services.billing.manage import change_plan as change

    workspace, plan = hosted_workspace(db, ctx), _plan(body)
    return billing_call(lambda provider: change(db, provider, workspace, plan=plan))


@router.post("/plan-change/cancel")
def cancel_plan_change(ctx: RequestContext = Depends(require_workspace_admin),
                       db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Keep the current plan: undo a downgrade waiting for the period end."""
    from services.billing.manage import cancel_change

    workspace = hosted_workspace(db, ctx)
    return billing_call(lambda provider: cancel_change(db, provider, workspace))
