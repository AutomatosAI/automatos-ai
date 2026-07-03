"""PRD-181 S3/S4 — GDPR API: data export + erasure-with-cascade.

Tenant-scoped: every endpoint operates on the caller's own ``ctx.workspace_id``,
so a workspace admin can export or erase **their own** workspace's data (which is
how a GDPR subject request reaches a tenant). Whole-workspace erasure is
irreversible, so it requires an explicit confirmation echo of the workspace id.

The heavy lifting is in ``services.gdpr_service`` (the real cascade across SQL,
Qdrant field memory, and mem0). Every action is audited there.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.database.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/gdpr", tags=["gdpr"])


def _actor_ref(ctx: RequestContext) -> str:
    uid = getattr(ctx, "user_id", None) or getattr(ctx, "internal_user_id", None)
    return f"user:{uid}" if uid is not None else "user:unknown"


def _require_workspace_admin(ctx: RequestContext) -> None:
    """GDPR export/erasure is an admin/owner action on the tenant."""
    role = getattr(ctx, "workspace_role", None) or getattr(getattr(ctx, "user", None), "system_role", None)
    if role not in ("owner", "admin", "super_admin"):
        raise HTTPException(status_code=403, detail="GDPR actions require workspace admin or owner")


@router.get("/export")
async def export_my_workspace(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Export this workspace's data (SQL + field memory + mem0) as a JSON bundle."""
    _require_workspace_admin(ctx)
    from services.gdpr_service import export_workspace

    bundle = export_workspace(db, ctx.workspace_id, requested_by=_actor_ref(ctx))
    return JSONResponse(content=bundle)


@router.post("/erase")
async def erase_my_workspace(
    body: Dict[str, Any] = Body(default_factory=dict),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Erase this whole workspace across every store (irreversible).

    Requires ``{"confirm_workspace_id": "<this workspace id>"}`` to guard against
    an accidental fire.
    """
    _require_workspace_admin(ctx)
    confirm = str(body.get("confirm_workspace_id") or "")
    if confirm != str(ctx.workspace_id):
        raise HTTPException(
            status_code=422,
            detail="confirm_workspace_id must equal the current workspace id to erase",
        )
    from services.gdpr_service import erase_workspace

    result = erase_workspace(db, ctx.workspace_id, requested_by=_actor_ref(ctx))
    return result


@router.post("/erase-subject")
async def erase_subject(
    body: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Erase a single data subject's data within this workspace where a subject
    tag exists. Stores lacking a subject tag are reported in ``gaps``.

    Body: ``{"subject_id": "<id>"}``. This is the platform entrypoint the future
    Shopify ``customers/redact`` webhook will call (the Remix handler itself is
    out of scope for this repo — flagged for the Shopify pod).
    """
    _require_workspace_admin(ctx)
    subject_id = str(body.get("subject_id") or "").strip()
    if not subject_id:
        raise HTTPException(status_code=422, detail="subject_id is required")
    from services.gdpr_service import erase_data_subject

    result = erase_data_subject(
        db, workspace_id=ctx.workspace_id, subject_id=subject_id, requested_by=_actor_ref(ctx)
    )
    return result
