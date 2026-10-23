"""Platform admin: grant or take back a workspace's credits (PRD-265 US-007).

``POST /api/admin/workspaces/{workspace_id}/credits`` writes one ``adjust`` row to
the credit ledger, with the admin who made it and their reason, and records an
audit event. A positive amount grants, a negative one takes back; the trial bucket
is the default, because trial credits pay for platform AI on every plan (the beta
runs on trial credits and admin grants until billing opens, PRD-265 §10).

``POST /api/admin/workspaces/{workspace_id}/business-invite`` offers Business to that
one workspace for ``BUSINESS_OFFER_DAYS``: its owner then buys it on their own
Billing tab, inside Studio, and is told so by a notification. Business is by invite,
never self-serve, so this offer is the only way to buy it (PRD-265 US-007).

It sits beside the operator console's other workspace routes
(``api/admin_workspaces.py``, same prefix) and has the same gate: platform admins
only, 403 for everyone else.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.workspaces import Workspace

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/workspaces", tags=["Admin Credits"])

SOURCE_ADMIN_ADJUST = "admin_adjust"
MAX_ADJUST_CREDITS = 1_000_000


class CreditAdjustBody(BaseModel):
    credits: int = Field(..., ge=-MAX_ADJUST_CREDITS, le=MAX_ADJUST_CREDITS)
    bucket: Literal["trial", "monthly", "topup"] = "trial"
    reason: str = Field(..., min_length=3, max_length=500)
    # The caller's id for this adjustment: a retried or double-submitted request with
    # the same id changes nothing the second time (the ledger keys on it). Without
    # one, every request is a new adjustment.
    request_id: Optional[str] = Field(None, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")

    @field_validator("credits")
    @classmethod
    def _not_zero(cls, value: int) -> int:
        if value == 0:
            raise ValueError("credits must not be zero")
        return value


def _require_platform_admin(ctx: RequestContext) -> None:
    from core.auth.roles import caller_is_admin

    if not ctx.user or not caller_is_admin(ctx.user):
        raise HTTPException(status_code=403, detail="Admin access required")


@router.post("/{workspace_id}/credits")
def adjust_workspace_credits(
    workspace_id: UUID,
    body: CreditAdjustBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Grant (positive) or take back (negative) credits, with a reason. Platform admin only."""
    _require_platform_admin(ctx)
    from core.auth.actor import resolve_internal_user_id
    from core.workspaces.audit import AuditService
    from services.credits.ledger import KIND_ADJUST, Entry, read_balance, record

    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if workspace is None or getattr(workspace, "deleted_at", None):
        raise HTTPException(status_code=404, detail="Workspace not found")
    actor = str(getattr(ctx.user, "email", None) or getattr(ctx.user, "id", "admin"))
    source_ref = f"{workspace_id}:{body.request_id}" if body.request_id else uuid.uuid4().hex
    written = record(db, workspace_id, Entry(
        kind=KIND_ADJUST, source_type=SOURCE_ADMIN_ADJUST, source_ref=source_ref,
        credits=body.credits, bucket=body.bucket, actor=actor, reason=body.reason,
    ))
    db.commit()  # the ledger first: the audit write commits (or rolls back) on its own
    if not written:
        return {"workspace_id": str(workspace_id), "applied": False,
                "balance": read_balance(db, workspace_id).as_dict()}
    AuditService(db).log(
        workspace_id=str(workspace_id),
        user_id=resolve_internal_user_id(db, ctx),
        action="workspace.credits_adjusted",
        resource_type="workspace",
        resource_id=str(workspace_id),
        resource_name=workspace.name,
        details={"credits": body.credits, "bucket": body.bucket, "reason": body.reason, "actor_email": actor},
    )
    logger.info("[credits] admin %s adjusted workspace %s by %+d %s", actor, workspace_id, body.credits, body.bucket)
    return {"workspace_id": str(workspace_id), "applied": True, "balance": read_balance(db, workspace_id).as_dict()}


@router.post("/{workspace_id}/business-invite")
def create_business_invite(
    workspace_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Offer Business to one workspace; its owner buys it on their Billing tab. Platform admin only."""
    _require_platform_admin(ctx)
    from api.billing import billing_call
    from services.billing.checkout import offer_business
    from services.billing.notices import NOTICE_BUSINESS_OFFERED, Notice, send_notice

    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if workspace is None or getattr(workspace, "deleted_at", None):
        raise HTTPException(status_code=404, detail="Workspace not found")
    until = billing_call(lambda provider: offer_business(db, provider, workspace))
    send_notice(db, workspace_id, Notice(
        NOTICE_BUSINESS_OFFERED, "Business is available to you",
        f"Automatos has offered this workspace the Business plan until {until:%d %b %Y}. "
        "Subscribe on the Billing tab in Settings.", status="info"))
    logger.info("[billing] admin %s offered Business to workspace %s until %s",
                getattr(ctx.user, "email", "?"), workspace_id, until.isoformat())
    return {"workspace_id": str(workspace_id), "business_offer_until": until.isoformat()}
