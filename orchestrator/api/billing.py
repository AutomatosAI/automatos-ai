"""Billing for a workspace's members (PRD-265 US-008).

- ``GET /api/v1/billing/summary`` — any member: plan, status, renewal date,
  balance by bucket, and whether platform-funded work is paused. The header
  balance and the paused banner read it.
- ``GET /api/v1/billing/ledger`` — the workspace's owner or admins: the credit
  ledger, newest first (a line per metered call), and the last 30 days' credits by model.

Hosted edition only: the local edition never meters, so both answer 404 there.
Plain ``def`` routes: they read a synchronous session (FastAPI runs them in its
threadpool), never ``async def`` over a blocking query.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db
from core.models.workspaces import Workspace

router = APIRouter(prefix="/api/v1/billing", tags=["Billing"])

LEDGER_PAGE_SIZE = 50
LEDGER_PAGE_SIZE_MAX = 200


def _hosted_workspace(db: Session, ctx: RequestContext) -> Workspace:
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

    return billing_summary(db, _hosted_workspace(db, ctx))


@router.get("/ledger")
def get_billing_ledger(
    page: int = Query(1, ge=1),
    page_size: int = Query(LEDGER_PAGE_SIZE, ge=1, le=LEDGER_PAGE_SIZE_MAX),
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The workspace's credit ledger, newest first, with the last 30 days' credits by model."""
    from services.credits.summary import ledger_page

    workspace = _hosted_workspace(db, ctx)
    return ledger_page(db, workspace.id, page=page, page_size=page_size)
