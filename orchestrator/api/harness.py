"""HARNESS self-management HTTP API — PRD-142 Wave 4 (W4-S1).

Authenticated approve/reject endpoints for queued high-risk HARNESS
prescriptions. The US-024 escalation points a workspace admin **here** (the
Command Center), never to a channel reply: the inbound webhook
(``api/webhooks.py``) performs **no sender authorization** — possession of the
URL-as-secret is its only credential — so a channel command can't be trusted to
mutate state. Approval happens on this authenticated, tenant-scoped surface,
where the calling principal is known and re-checked as a workspace admin.

The approve/reject logic itself lives in ``api.harness_commands`` (the handler
the design always intended, with its own fail-closed admin gate). This router
only (1) resolves the authenticated principal to the integer ``users.id`` that
gate expects, and (2) maps the handler's result to HTTP status codes. The flag
``HARNESS_SELF_MANAGEMENT_ENABLED`` still governs whether anything happens.

Approval is **human-admin-only by design**: a principal that doesn't resolve to
an active owner/admin ``users`` row — e.g. an API-key / service principal, which
carries no clerk id or email — fails closed (403). The platform never grants a
service account self-management authority over itself.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.harness_commands import handle_harness_command
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.core import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/harness", tags=["HARNESS"])


def _resolve_internal_user_id(db: Session, ctx: RequestContext) -> Optional[int]:
    """Resolve the integer ``users.id`` for the authenticated principal.

    ``ctx.user.id`` is a Clerk string id or email, but the HARNESS authz keys on
    the integer ``workspace_members.user_id`` — so the row is looked up first.
    Fail-closed: returns None when nothing matches, and the handler's admin gate
    then refuses (an unresolved principal can never mutate state). Mirrors
    ``api.team._resolve_internal_user_id`` — there is no shared util yet.
    """
    if not ctx.user:
        return None
    user = None
    if ctx.user.clerk_user_id:
        user = db.query(User).filter(User.clerk_user_id == ctx.user.clerk_user_id).first()
    if not user and ctx.user.email:
        user = db.query(User).filter(User.email == ctx.user.email).first()
    return user.id if user else None


async def _run_command(
    db: Session, ctx: RequestContext, command: str, rx_id: str
) -> Dict[str, Any]:
    """Resolve the caller, run the HARNESS command, map the result to HTTP."""
    user_id = _resolve_internal_user_id(db, ctx)
    # caller_identity carries the integer users.id; a None id fails the handler's
    # admin gate closed (→ 403), so an unresolved principal changes nothing.
    result = await handle_harness_command(
        db, ctx.workspace_id, command, rx_id, {"user_id": user_id}
    )
    if result.get("unauthorized"):
        raise HTTPException(status_code=403, detail=result.get("message"))
    if not result.get("success"):
        # Flag disabled, unknown/already-rejected rx, unresolved target, etc.
        # The message carries the reason; 409 keeps it distinct from an authz 403.
        raise HTTPException(status_code=409, detail=result.get("message"))
    return result


@router.post("/prescriptions/{rx_id}/approve", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def approve_prescription(
    rx_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Approve a queued HARNESS prescription — applies it now, audited + reversible."""
    return await _run_command(db, ctx, "/approve", rx_id)


@router.post("/prescriptions/{rx_id}/reject", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def reject_prescription(
    rx_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Reject a queued HARNESS prescription — suppresses future re-proposal."""
    return await _run_command(db, ctx, "/reject", rx_id)


@router.get("/self-learning")
async def self_learning_health(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Self-learning health for the Command Center tile (PRD-142 Wave 4, W4-S16 backend).

    Workspace-scoped read combining the three self-learning signals: HARNESS loop
    status, the tool-routing recorder's stats, and a prescription summary from the
    structured store. Each section is best-effort — a failure in one (e.g. the
    harness_prescriptions table not yet migrated in this env) degrades to an
    ``error`` marker rather than 500-ing the tile. The frontend component that
    renders this is human-built (W4-S16 FE half).
    """
    ws = ctx.workspace_id
    out: Dict[str, Any] = {"workspace_id": str(ws)}

    try:
        from services.harness_service import get_harness_service
        out["harness"] = get_harness_service().get_status(ws, db=db)
    except Exception as exc:  # noqa: BLE001 — best-effort tile section
        out["harness"] = {"error": str(exc)}

    try:
        from modules.tools.discovery.signal_recorder import get_tool_signal_recorder
        out["tool_routing"] = get_tool_signal_recorder().stats()
    except Exception as exc:  # noqa: BLE001
        out["tool_routing"] = {"error": str(exc)}

    return out
