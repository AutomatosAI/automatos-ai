"""PRD-181 S2 (F060) — approval-grant API.

The human-in-the-loop surface for the durable approval grants that gate board
tasks, playbook runs, and (future) scheduled/webhook agents. A pending grant
holds its subject blocked; granting it here authorises the subject and re-queues
a blocked board task; revoking a live grant retracts the authorisation.

Every state change is audited (governance action) via the S1 nullable-user audit
path — the actor here is the human approver.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.database.database import get_db
from core.models.approval_grants import ApprovalGrant, GrantStatus, SUBJECT_BOARD_TASK
from core.models.core import BoardTask

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/approval-grants", tags=["approval-grants"])


def _actor_ref(ctx: RequestContext) -> str:
    uid = getattr(ctx, "user_id", None) or getattr(ctx, "internal_user_id", None)
    return f"user:{uid}" if uid is not None else "user:unknown"


def _audit(db: Session, ctx: RequestContext, action: str, grant: ApprovalGrant) -> None:
    try:
        from core.workspaces.audit import AuditService

        AuditService(db).log(
            workspace_id=str(grant.workspace_id),
            user_id=getattr(ctx, "internal_user_id", None) or getattr(ctx, "user_id", None)
            if isinstance(getattr(ctx, "user_id", None), int) else None,
            actor_type="user",
            action=action,
            resource_type="approval_grant",
            resource_id=str(grant.id),
            resource_name=f"{grant.subject_type}:{grant.subject_id}",
            details={
                "subject_type": grant.subject_type,
                "subject_id": grant.subject_id,
                "status": grant.status,
                "actor": _actor_ref(ctx),
            },
        )
    except Exception:
        logger.warning("[approval_grants.api] audit failed for %s", action, exc_info=True)


@router.get("")
async def list_grants(
    status: Optional[str] = None,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """List this workspace's approval grants, newest first. Filter by ``status``."""
    q = db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == ctx.workspace_id)
    if status:
        q = q.filter(ApprovalGrant.status == status)
    rows: List[ApprovalGrant] = q.order_by(ApprovalGrant.requested_at.desc()).limit(200).all()
    return {"grants": [g.to_dict() for g in rows]}


def _load_grant(db: Session, ctx: RequestContext, grant_id: int) -> ApprovalGrant:
    grant = (
        db.query(ApprovalGrant)
        .filter(ApprovalGrant.id == grant_id, ApprovalGrant.workspace_id == ctx.workspace_id)
        .first()
    )
    if grant is None:
        raise HTTPException(status_code=404, detail="Approval grant not found")
    return grant


@router.post("/{grant_id}/grant")
async def grant_approval(
    grant_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Approve a pending grant (a human says yes) and re-queue any blocked subject."""
    from core.services.approval_grants import grant_grant

    grant = _load_grant(db, ctx, grant_id)
    if grant.status != GrantStatus.PENDING.value:
        raise HTTPException(status_code=422, detail=f"Grant is not pending (status: {grant.status})")

    grant_grant(grant, granted_by=_actor_ref(ctx))
    _requeue_subject(db, grant)
    db.commit()
    _audit(db, ctx, "approval_grant:granted", grant)
    return {"grant": grant.to_dict()}


@router.post("/{grant_id}/deny")
async def deny_approval(
    grant_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Refuse a pending grant — the subject fails rather than retrying."""
    from core.services.approval_grants import deny_grant

    grant = _load_grant(db, ctx, grant_id)
    if grant.status != GrantStatus.PENDING.value:
        raise HTTPException(status_code=422, detail=f"Grant is not pending (status: {grant.status})")

    deny_grant(grant, revoked_by=_actor_ref(ctx))
    _fail_subject(db, grant)
    db.commit()
    _audit(db, ctx, "approval_grant:denied", grant)
    return {"grant": grant.to_dict()}


@router.post("/{grant_id}/revoke")
async def revoke_approval(
    grant_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Retract a granted authorisation before it expires."""
    from core.services.approval_grants import revoke_grant

    grant = _load_grant(db, ctx, grant_id)
    if grant.status not in (GrantStatus.GRANTED.value, GrantStatus.PENDING.value):
        raise HTTPException(status_code=422, detail=f"Grant cannot be revoked (status: {grant.status})")

    revoke_grant(grant, revoked_by=_actor_ref(ctx))
    db.commit()
    _audit(db, ctx, "approval_grant:revoked", grant)
    return {"grant": grant.to_dict()}


def _requeue_subject(db: Session, grant: ApprovalGrant) -> None:
    """On grant, return a blocked board task to the dispatch queue."""
    if grant.subject_type != SUBJECT_BOARD_TASK:
        return
    try:
        task = db.query(BoardTask).get(int(grant.subject_id))
    except (TypeError, ValueError):
        return
    if task is None or task.status != "blocked":
        return
    task.status = "assigned"
    task.blocked_at = None
    task.blocked_reason = None
    try:
        from services.board_dispatcher import notify_task_available

        notify_task_available(db, workspace_id=grant.workspace_id, task_id=task.id)
    except Exception:
        logger.warning("[approval_grants.api] notify_task_available failed", exc_info=True)


def _fail_subject(db: Session, grant: ApprovalGrant) -> None:
    """On deny, fail a blocked board task (the human refused the action)."""
    if grant.subject_type != SUBJECT_BOARD_TASK:
        return
    try:
        task = db.query(BoardTask).get(int(grant.subject_id))
    except (TypeError, ValueError):
        return
    if task is None or task.status != "blocked":
        return
    from datetime import datetime, timezone

    task.status = "failed"
    task.error_message = "Approval denied by a human reviewer"
    task.completed_at = datetime.now(timezone.utc)
