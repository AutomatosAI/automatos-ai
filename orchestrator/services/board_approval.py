"""PRD-181 S2 (F060) — board-task approval gate.

Closes the review gap "governance covers missions only — board tasks Auto
creates have no ceiling or approval gate". A board task about to execute is run
through the **same** ``evaluate_approval`` primitive missions use (PRD-163):

  - the workspace policy auto-approves ⇒ the task runs (no grant needed);
  - the policy asks (always_ask, or over the dollar ceiling) ⇒ a durable,
    revocable, expiring :class:`ApprovalGrant` is created and the task is
    **blocked** until a human grants it — not hard-blocked, not auto-allowed.

The grant is the tool-agnostic record the future scheduled/webhook agents share.
Every grant creation is audited (governance action), and — PRD-196 S2 (C.8) —
announced to the workspace's admins via an ``approval_pending`` notification,
so a blocked task never waits silently.

This module is the decision glue only; the dispatch wiring (moving the board task
to ``blocked`` and re-queuing it on grant) lives in ``api.board_tasks`` and
``services.board_dispatcher`` which call :func:`evaluate_board_task_approval`.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional, Set
from uuid import UUID

logger = logging.getLogger(__name__)

# Strong refs to fire-and-forget notification tasks (the api/webhooks.py
# background-task idiom) so the loop cannot GC them mid-flight.
_NOTIFY_TASKS: Set["asyncio.Task"] = set()


@dataclass(frozen=True)
class BoardApprovalOutcome:
    """Result of evaluating a board task against the workspace approval policy."""

    requires_approval: bool
    reason: str
    grant: Optional[Any]  # ApprovalGrant when a grant was created, else None
    policy: str
    estimated_cost_usd: float


def _audit_governance(db: Any, workspace_id: UUID | str, action: str, **details: Any) -> None:
    """Write a governance AuditLog row (best-effort). Patchable in tests.

    Actor is the system (a dispatcher decision, no human principal), so the S1
    nullable-user audit path is used.
    """
    try:
        from core.workspaces.audit import AuditService

        AuditService(db).log(
            workspace_id=str(workspace_id) if workspace_id is not None else None,
            user_id=None,
            actor_type="system",
            action=action,
            resource_type="approval_grant",
            resource_id=str(details.get("grant_id")) if details.get("grant_id") else None,
            resource_name=details.get("subject_id"),
            details=details,
        )
    except Exception:
        logger.warning("[board_approval] governance audit failed for %s", action, exc_info=True)


def evaluate_board_task_approval(
    db: Any,
    *,
    workspace_id: UUID | str,
    task_id: int | str,
    estimated_cost_usd: float = 0.0,
    agent_id: Optional[int] = None,
    risk_tier: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
    _policy_override: Optional[str] = None,
    _ceiling_override: Optional[float] = None,
) -> BoardApprovalOutcome:
    """Decide whether a board task needs approval; create + audit a grant if so.

    ``_policy_override`` / ``_ceiling_override`` are test seams that bypass the
    workspace-settings read so the gate can be exercised without a DB row.
    """
    from core.services.approval_grants import (
        DEFAULT_TTL_SECONDS,
        create_grant,
        find_pending_grant,
    )
    from core.models.approval_grants import SUBJECT_BOARD_TASK

    # Resolve the approval decision. Reuse the mission primitive unless a test
    # override is supplied.
    if _policy_override is not None:
        decision = _decide_from_override(_policy_override, _ceiling_override, estimated_cost_usd)
        policy = _policy_override
    else:
        from core.services.approval_policy import evaluate_approval

        d = evaluate_approval(db, workspace_id, estimated_cost_usd)
        decision = d.auto_approve
        policy = d.policy

    if decision:  # auto-approve ⇒ no grant, task runs.
        return BoardApprovalOutcome(
            requires_approval=False,
            reason="auto-approved by workspace policy",
            grant=None,
            policy=policy,
            estimated_cost_usd=estimated_cost_usd,
        )

    # Needs approval. Idempotency: reuse an existing pending grant for this task.
    existing = find_pending_grant(
        db, workspace_id, subject_type=SUBJECT_BOARD_TASK, subject_id=str(task_id)
    )
    if existing is not None:
        return BoardApprovalOutcome(
            requires_approval=True,
            reason="pending approval grant already exists",
            grant=existing,
            policy=policy,
            estimated_cost_usd=estimated_cost_usd,
        )

    reason = f"board task requires approval under '{policy}' policy"
    grant = create_grant(
        db, workspace_id,
        subject_type=SUBJECT_BOARD_TASK,
        subject_id=str(task_id),
        risk_tier=risk_tier,
        agent_id=agent_id,
        reason=reason,
        estimated_cost_usd=estimated_cost_usd,
        ttl_seconds=ttl_seconds if ttl_seconds is not None else DEFAULT_TTL_SECONDS,
    )
    _audit_governance(
        db, workspace_id, "approval_grant:created",
        grant_id=getattr(grant, "id", None),
        subject_type=SUBJECT_BOARD_TASK,
        subject_id=str(task_id),
        policy=policy,
        estimated_cost_usd=estimated_cost_usd,
        risk_tier=risk_tier,
    )
    # PRD-196 S2 (C.8): a pending grant must never wait silently — tell the
    # workspace's humans. Only on fresh creation (the reuse branch above was
    # already announced when its grant was created).
    _notify_approval_pending(
        workspace_id,
        grant_id=getattr(grant, "id", None),
        subject_id=str(task_id),
        risk_tier=risk_tier,
        reason=reason,
        estimated_cost_usd=estimated_cost_usd,
    )
    return BoardApprovalOutcome(
        requires_approval=True,
        reason=reason,
        grant=grant,
        policy=policy,
        estimated_cost_usd=estimated_cost_usd,
    )


async def _dispatch_approval_pending(
    workspace_id: str,
    grant_id: Any,
    subject_id: str,
    risk_tier: Optional[str],
    reason: str,
    estimated_cost_usd: float,
) -> None:
    """Dispatch ``approval_pending`` through the canonical notification seam.

    Owns its session: by the time the loop runs this, the creating caller's
    transaction is finished (and its session may be closed).
    """
    from core.database.database import SessionLocal
    from core.services.notification_dispatcher import NotificationDispatcher

    db = SessionLocal()
    try:
        message = reason
        if estimated_cost_usd:
            message = f"{reason} (est. ${estimated_cost_usd:.2f})"
        if risk_tier:
            message = f"{message} [risk: {risk_tier}]"
        await NotificationDispatcher(db, workspace_id).dispatch(
            event_type="approval_pending",
            title=f"Approval needed: board task {subject_id}",
            message=message,
            link_type="approval_grant",
            link_id=str(grant_id) if grant_id is not None else None,
            status="action_required",
        )
    except Exception:
        logger.warning(
            "[board_approval] approval_pending dispatch failed for grant %s",
            grant_id, exc_info=True,
        )
    finally:
        db.close()


def _notify_approval_pending(
    workspace_id: UUID | str,
    *,
    grant_id: Any,
    subject_id: str,
    risk_tier: Optional[str],
    reason: str,
    estimated_cost_usd: float,
) -> None:
    """Schedule the pending-grant notification without blocking the gate.

    The gate is sync and its live caller runs on the event loop, so the
    dispatch is scheduled as a tracked task there; with no running loop
    (scripts, sync tests) it runs inline. Never raises — a notification fault
    must not wedge board execution (same posture as the gate's caller).
    """
    try:
        coro = _dispatch_approval_pending(
            str(workspace_id), grant_id, subject_id, risk_tier, reason, estimated_cost_usd,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
            return
        task = loop.create_task(coro)
        _NOTIFY_TASKS.add(task)
        task.add_done_callback(_NOTIFY_TASKS.discard)
    except Exception:
        logger.warning("[board_approval] approval_pending scheduling failed", exc_info=True)


def _decide_from_override(
    policy: str, ceiling: Optional[float], estimated_cost_usd: float
) -> bool:
    """Pure re-expression of the approval decision for the test seam.

    Mirrors ``approval_policy.evaluate_approval`` without a DB: always_ask never
    auto-approves; auto_below_budget approves at/below the ceiling; full_auto
    approves (the §12.3 gate is out of scope for the seam).
    """
    if policy == "auto_below_budget":
        return ceiling is not None and estimated_cost_usd <= ceiling
    if policy == "full_auto":
        return True
    return False  # always_ask (default / fail-safe)
