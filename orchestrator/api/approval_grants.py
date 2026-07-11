"""PRD-181 S2 (F060) — approval-grant API.

The human-in-the-loop surface for the durable approval grants that gate board
tasks, playbook runs, and (future) scheduled/webhook agents. A pending grant
holds its subject blocked; granting it here authorises the subject and re-queues
a blocked board task; revoking a live grant retracts the authorisation.

PRD-196 S2 (P2-15, governance C.8): the whole surface is workspace-admin gated
via the canonical ``require_workspace_admin`` dependency (PRD-185 S12) — a
plain member must not be able to authorise an agent's destructive action, and
the approvals surface is ws-admin-only by Gerard's posture call (196 Q2).

Every state change is audited (governance action) via the S1 nullable-user audit
path — the actor here is the human approver.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.workspace_admin import require_workspace_admin
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
    ctx: RequestContext = Depends(require_workspace_admin),
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
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Approve a pending grant (a human says yes) and re-queue any blocked subject."""
    from core.services.approval_grants import grant_grant

    grant = _load_grant(db, ctx, grant_id)
    if grant.status != GrantStatus.PENDING.value:
        raise HTTPException(status_code=422, detail=f"Grant is not pending (status: {grant.status})")

    grant_grant(grant, granted_by=_actor_ref(ctx))
    # PRD-193 S4: for tool_call subjects this re-dispatches the stored call
    # (consume-then-execute inside this one transaction boundary).
    await _requeue_subject(db, grant)
    db.commit()
    _audit(db, ctx, "approval_grant:granted", grant)
    return {"grant": grant.to_dict()}


@router.post("/{grant_id}/deny")
async def deny_approval(
    grant_id: int,
    ctx: RequestContext = Depends(require_workspace_admin),
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
    ctx: RequestContext = Depends(require_workspace_admin),
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


async def _requeue_subject(db: Session, grant: ApprovalGrant) -> None:
    """On grant, resume the blocked subject.

    - ``board_task``: return the blocked task to the dispatch queue (unchanged).
    - ``tool_call`` (PRD-193 S4, P2-12): re-dispatch the stored call through
      the spine — or, for board-originated asks, ride the existing board
      re-queue so the re-run completes into the now-active grant (S2).
    """
    from core.models.approval_grants import SUBJECT_TOOL_CALL

    if grant.subject_type == SUBJECT_TOOL_CALL:
        await _resume_tool_call(db, grant)
        return
    if grant.subject_type != SUBJECT_BOARD_TASK:
        return
    _requeue_blocked_task(db, grant.workspace_id, grant.subject_id)


def _requeue_blocked_task(db: Session, workspace_id: Any, task_id: Any) -> bool:
    """Return a BLOCKED board task to the dispatch queue. True iff re-queued.

    Extracted unchanged from the board branch of ``_requeue_subject`` so the
    PRD-193 S4 board linkage (a ``tool_call`` grant carrying
    ``details.board_task_id``) resumes through the SAME code path.
    """
    try:
        task = db.query(BoardTask).get(int(task_id))
    except (TypeError, ValueError):
        return False
    if task is None or task.status != "blocked":
        return False
    task.status = "assigned"
    task.blocked_at = None
    task.blocked_reason = None
    try:
        from services.board_dispatcher import notify_task_available

        notify_task_available(db, workspace_id=workspace_id, task_id=task.id)
    except Exception:
        logger.warning("[approval_grants.api] notify_task_available failed", exc_info=True)
    return True


async def _resume_tool_call(db: Session, grant: ApprovalGrant) -> None:
    """PRD-193 S4 (P2-12): approving must complete the work, not just flip a row.

    Board-originated asks (``details.board_task_id`` present + task still
    blocked) resume via the EXISTING board re-queue — the task re-run retries
    the call and meets the now-active grant (S2), so nothing double-executes
    (locked decision 4: lean board linkage). Everything else re-dispatches the
    stored call directly through ``UnifiedToolExecutor.execute_tool`` — the
    same spine the original call would have taken, so telemetry, the policy
    seam, and outcome capture all fire; nothing is exempted by being approved.

    The outcome summary lands on ``details.executed_result`` (returned in the
    grant response so the S3 card swaps to its executed state). Fail LOUD but
    contained: a failing re-dispatch surfaces as an honest failure on the
    grant — never a fake success (tool-runtime dossier C.3), and never an
    exception out of the grant endpoint.
    """
    from datetime import datetime, timezone

    details = dict(grant.details) if isinstance(grant.details, dict) else {}

    board_task_id = details.get("board_task_id")
    if board_task_id is not None and _requeue_blocked_task(
        db, grant.workspace_id, board_task_id
    ):
        grant.details = {
            **details,
            "executed_result": {
                "resumed_via": "board_task_requeue",
                "board_task_id": board_task_id,
                "executed_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        return

    action = details.get("action") or grant.tool_name
    params = details.get("params")
    params = dict(params) if isinstance(params, dict) else {}
    caller_context = details.get("caller_context")
    caller_context = dict(caller_context) if isinstance(caller_context, dict) else None

    if not action:
        grant.details = {
            **details,
            "executed_result": {
                "success": False,
                "error": "grant carries no stored action to resume",
                "executed_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        return

    try:
        from modules.tools.execution.unified_executor import UnifiedToolExecutor

        executor = UnifiedToolExecutor(db)
        raw = await executor.execute_tool(
            tool_name=str(action),
            parameters=params,
            agent_id=int(grant.agent_id or 0),
            workspace_id=grant.workspace_id,
            trace_id=f"grant-resume-{grant.id}",
            caller_context=caller_context,
        )
        raw = raw if isinstance(raw, dict) else {}
        ok = bool(raw.get("success"))
        summary: Dict[str, Any] = {
            "success": ok,
            "error": (str(raw.get("error"))[:500] if (not ok and raw.get("error")) else None),
            "requires_confirmation": bool(raw.get("requires_confirmation")),
            "executed_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.error(
            "[approval_grants.api] tool_call resume failed for grant %s",
            grant.id, exc_info=True,
        )
        summary = {
            "success": False,
            "error": str(exc)[:500],
            "requires_confirmation": False,
            "executed_at": datetime.now(timezone.utc).isoformat(),
        }
    grant.details = {**details, "executed_result": summary}


def _fail_subject(db: Session, grant: ApprovalGrant) -> None:
    """On deny, fail the blocked subject (the human refused the action)."""
    from core.models.approval_grants import SUBJECT_TOOL_CALL

    if grant.subject_type == SUBJECT_TOOL_CALL:
        # PRD-193 S4: a denied tool call is a no-op execution — the grant's
        # DENIED status is the record; the S3 card renders the refusal; the
        # model sees it via context on the next turn. (A blocked board task,
        # if any, keeps its own board_task-subject grant lifecycle.)
        return
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
