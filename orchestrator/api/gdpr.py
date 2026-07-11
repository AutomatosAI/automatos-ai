"""PRD-181 S3/S4 — GDPR API: data export + erasure-with-cascade.

Tenant-scoped: every endpoint operates on the caller's own ``ctx.workspace_id``,
so a workspace admin can export or erase **their own** workspace's data (which is
how a GDPR subject request reaches a tenant). Whole-workspace erasure is
irreversible, so it requires an explicit confirmation echo of the workspace id.

The heavy lifting is in ``services.gdpr_service`` (the real cascade across SQL,
Qdrant field memory, and durable memory). Every action is audited there.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/gdpr", tags=["gdpr"])


def _actor_ref(ctx: RequestContext) -> str:
    uid = getattr(ctx, "user_id", None) or getattr(ctx, "internal_user_id", None)
    return f"user:{uid}" if uid is not None else "user:unknown"


# PRD-196 S7: consolidated onto the canonical ``require_workspace_admin``
# (PRD-185 S12) — the hand-rolled ``_require_workspace_admin`` (a role-string
# check, different semantics from the membership-row check) is deleted so there
# is ONE admin semantic across the governance + GDPR surfaces.


@router.get("/export")
async def export_my_workspace(
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Export this workspace's data (SQL + field memory + durable memory) as a JSON bundle."""
    from services.gdpr_service import export_workspace

    bundle = export_workspace(db, ctx.workspace_id, requested_by=_actor_ref(ctx))
    return JSONResponse(content=bundle)


@router.post("/erase")
async def erase_my_workspace(
    body: Dict[str, Any] = Body(default_factory=dict),
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Erase this whole workspace across every store (irreversible).

    Requires ``{"confirm_workspace_id": "<this workspace id>"}`` to guard against
    an accidental fire.
    """
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
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Erase a single data subject's data within this workspace where a subject
    tag exists (PRD-196 S6 made field + durable real deletes). Stores lacking a
    subject tag (SQL) are reported in ``gaps``; pre-tag rows in ``untagged_history``.

    Body: ``{"subject_id": "<id>"}``. This is the platform entrypoint the Shopify
    ``customers/redact`` webhook calls (the Remix handler itself is out of scope
    for this repo — flagged for the Shopify pod).
    """
    subject_id = str(body.get("subject_id") or "").strip()
    if not subject_id:
        raise HTTPException(status_code=422, detail="subject_id is required")
    from services.gdpr_service import erase_data_subject

    result = erase_data_subject(
        db, workspace_id=ctx.workspace_id, subject_id=subject_id, requested_by=_actor_ref(ctx)
    )
    return result
