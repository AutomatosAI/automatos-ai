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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db
from core.models.approval_grants import (
    ApprovalGrant,
    GrantStatus,
    KIND_QUESTION,
    SUBJECT_BOARD_TASK,
)
from core.models.core import BoardTask
from modules.policy.ai_act import oversight_for_risk

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/approval-grants", tags=["approval-grants"])


def grant_with_oversight(grant: ApprovalGrant) -> Dict[str, Any]:
    """A grant dict enriched with its EU-AI-Act Art.14 oversight tier + rationale.

    PRD-196 S1 (P2-15, governance I.1): the approvals inbox card shows *why* a
    human is in the loop. Rather than let the client invent a rationale, the list
    response carries the pure ``oversight_for_risk`` mapping for the grant's risk
    tier — fail-safe to human-in-the-loop when the risk is unknown/absent
    (mirroring ``ai_act.py``'s conservative fallback).
    """
    data = grant.to_dict()
    data["oversight"] = oversight_for_risk(grant.risk_tier).to_dict()
    return data


def _grant_payload(db: Session, grant: ApprovalGrant) -> Dict[str, Any]:
    """List payload for one grant. Question-kind rows carry their blocked
    cascade (PRD-225) so the Questions tab renders the downstream work stuck
    behind the ask without a second round-trip."""
    data = grant_with_oversight(grant)
    if getattr(grant, "kind", None) == KIND_QUESTION:
        if grant.subject_type == SUBJECT_BOARD_TASK:
            from services.ask_cascade import board_task_cascade_detail

            data["cascade"] = board_task_cascade_detail(
                db, grant.workspace_id, grant.subject_id
            )
        else:
            data["cascade"] = {"total": 0, "tasks": []}
    return data


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
    kind: Optional[str] = None,
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """List this workspace's approval grants, newest first.

    Filter by ``status`` and/or ``kind`` — the Questions tab reuses THIS route
    with ``kind=question`` (PRD-225: no separate list endpoint). Question rows
    are enriched with their blocked cascade.
    """
    q = db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == ctx.workspace_id)
    if status:
        q = q.filter(ApprovalGrant.status == status)
    if kind:
        q = q.filter(ApprovalGrant.kind == kind)
    rows: List[ApprovalGrant] = q.order_by(ApprovalGrant.requested_at.desc()).limit(200).all()
    return {"grants": [_grant_payload(db, g) for g in rows]}


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
    # PRD-225: a question is answered, never approved — /answer is its only
    # completion path (a yes/no can't stand in for a free-text decision).
    if grant.kind == KIND_QUESTION:
        raise HTTPException(
            status_code=422,
            detail="This is a question — answer it via POST /{id}/answer, not /grant.",
        )
    if grant.status != GrantStatus.PENDING.value:
        raise HTTPException(status_code=422, detail=f"Grant is not pending (status: {grant.status})")

    grant_grant(grant, granted_by=_actor_ref(ctx))
    # COMMIT THE YES BEFORE RESUMING (2026-08-06 incident, grant 77).
    # SessionLocal runs autoflush=False, and the resume re-enters the
    # confirmation gate, whose consume_tool_grant() runs real SQL — an
    # uncommitted GRANTED row is still 'pending' to that query, so the gate
    # re-asked and the human's approval executed nothing. Committing first
    # makes the recorded yes visible to the gate AND durable even if the
    # resume itself crashes; the follow-up commit persists executed_result.
    db.commit()
    # PRD-193 S4: for tool_call subjects this re-dispatches the stored call
    # (consume-then-execute against the now-committed grant).
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
    """Refuse a pending grant. An approval denial FAILS the subject; a question
    denial is a *dismiss* — the subject stays blocked and the asker may re-ask
    (PRD-225 baked decision), so no answer is fabricated and the trail is kept."""
    from core.services.approval_grants import deny_grant

    grant = _load_grant(db, ctx, grant_id)
    if grant.status != GrantStatus.PENDING.value:
        raise HTTPException(status_code=422, detail=f"Grant is not pending (status: {grant.status})")

    is_question = grant.kind == KIND_QUESTION
    deny_grant(grant, revoked_by=_actor_ref(ctx))
    if not is_question:
        # Only an approval denial fails the subject. A dismissed question leaves
        # the parked subject blocked — answering "use your judgment" is the
        # one-click unblock path.
        _fail_subject(db, grant)
    db.commit()
    _audit(db, ctx, "question:dismissed" if is_question else "approval_grant:denied", grant)
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


# ---------------------------------------------------------------------------
# PRD-225 — answer a question and resume the parked subject
# ---------------------------------------------------------------------------

class AnswerRequest(BaseModel):
    """The human's answer: free text and/or one of the offered options."""

    answer_text: Optional[str] = None
    option: Optional[str] = None


@router.post("/{grant_id}/answer")
async def answer_question(
    grant_id: int,
    body: AnswerRequest,
    ctx: RequestContext = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Answer a pending question: record the answer, resume the parked subject
    through the SAME machinery a grant uses, and confirm into chat.

    ``pending → granted`` on a ``kind='question'`` row. The Q&A is appended to the
    subject's execution context (board task ``planning_data.human_qa`` or the
    tool-call resume payload) so the agent's next run reads the answer verbatim.
    """
    grant = _load_grant(db, ctx, grant_id)
    if grant.kind != KIND_QUESTION:
        raise HTTPException(
            status_code=422, detail="This grant is an approval, not a question."
        )
    if grant.status != GrantStatus.PENDING.value:
        raise HTTPException(
            status_code=422, detail=f"Question is not open (status: {grant.status})"
        )

    answer = (body.answer_text or body.option or "").strip()
    if not answer:
        raise HTTPException(status_code=422, detail="answer_text or option is required")
    # A chosen option must be one of the offered choices; free text is unconstrained.
    if body.option and not body.answer_text:
        options = grant.options if isinstance(grant.options, list) else []
        if options and body.option not in options:
            raise HTTPException(
                status_code=422, detail="option is not one of the offered choices"
            )

    await apply_question_answer(db, grant, answer_text=answer, answered_by=_actor_ref(ctx))
    _audit(db, ctx, "question:answered", grant)
    return {"grant": grant.to_dict()}


async def apply_question_answer(
    db: Session, grant: ApprovalGrant, *, answer_text: str, answered_by: str
) -> ApprovalGrant:
    """Record an answer and resume the parked subject — the shared service the
    HTTP endpoint AND the Telegram bridge both call (PRD-225 US-005; the bridge
    is NOT an HTTP self-call). Assumes the caller validated kind='question' +
    status pending.

    ``pending → granted``; the Q&A is appended to the subject's execution context
    and the work resumes through the EXISTING ``_requeue_subject``.
    """
    now = datetime.now(timezone.utc)
    grant.answer_text = answer_text
    grant.answered_by = answered_by
    grant.answered_at = now
    grant.status = GrantStatus.GRANTED.value
    _record_answer_on_subject(db, grant, answer_text, now)

    # Commit the answer BEFORE resuming — mirror grant_approval's 2026-08-06
    # ordering so the resume path sees a committed, non-pending row.
    db.commit()
    await _requeue_subject(db, grant)
    db.commit()

    _confirm_answer_into_chat(db, grant)
    return grant


def _record_answer_on_subject(
    db: Session, grant: ApprovalGrant, answer: str, now: datetime
) -> None:
    """Append the Q&A into the subject's execution context (rebuild-don't-mutate
    the JSONB) so the agent's next run reads the human's answer verbatim."""
    entry = {
        "q": grant.question_md,
        "a": answer,
        "answered_by": grant.answered_by,
        "at": now.isoformat(),
    }
    if grant.subject_type == SUBJECT_BOARD_TASK:
        try:
            task = db.query(BoardTask).get(int(grant.subject_id))
        except (TypeError, ValueError):
            task = None
        if task is not None:
            pdata = dict(task.planning_data or {})
            qa = list(pdata.get("human_qa") or [])
            qa.append(entry)
            task.planning_data = {**pdata, "human_qa": qa}
        return
    # tool_call / playbook_run: the resume payload carries the Q&A.
    details = dict(grant.details or {})
    qa = list(details.get("human_qa") or [])
    qa.append(entry)
    grant.details = {**details, "human_qa": qa}


def _confirm_answer_into_chat(db: Session, grant: ApprovalGrant) -> None:
    """Fail-soft chat confirmation that the answered work is resuming."""
    try:
        from services.chat_messenger import deliver_background_message

        subject_label = f"{grant.subject_type.replace('_', ' ')} {grant.subject_id}"
        deliver_background_message(
            db,
            workspace_id=grant.workspace_id,
            text=f"Answered — resuming {subject_label}.",
            source={"origin": "question_answer", "grant_id": grant.id},
            link_type="question",
            link_id=str(grant.id),
        )
    except Exception:  # noqa: BLE001 — confirmation must never fail the answer
        logger.warning(
            "[approval_grants.api] answer confirmation failed for grant %s",
            grant.id, exc_info=True,
        )


async def _requeue_subject(db: Session, grant: ApprovalGrant) -> None:
    """On grant, resume the blocked subject.

    - ``board_task``: return the blocked task to the dispatch queue (unchanged).
    - ``tool_call`` (PRD-193 S4, P2-12): re-dispatch the stored call through
      the spine — or, for board-originated asks, ride the existing board
      re-queue so the re-run completes into the now-active grant (S2).
    - ``playbook_run`` (PRD-204 S7/S8): the watcher's corrective-action
      grants -- ``details.watch_action`` discriminates (rerun / replan /
      reassign / spawn_agent); the stored spec launches and the supervising
      watch follows the work. First real wiring of SUBJECT_PLAYBOOK_RUN.
    """
    from core.models.approval_grants import SUBJECT_PLAYBOOK_RUN, SUBJECT_TOOL_CALL

    if grant.subject_type == SUBJECT_TOOL_CALL:
        await _resume_tool_call(db, grant)
        return
    if grant.subject_type == SUBJECT_PLAYBOOK_RUN:
        from services.watch_rerun import resume_playbook_run_grant

        await resume_playbook_run_grant(db, grant)
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
    from core.models.approval_grants import SUBJECT_PLAYBOOK_RUN, SUBJECT_TOOL_CALL

    if grant.subject_type == SUBJECT_PLAYBOOK_RUN:
        # PRD-204 S7: no launch; the supervising watch parks for a human.
        from services.watch_rerun import fail_playbook_run_grant

        fail_playbook_run_grant(db, grant)
        return
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
