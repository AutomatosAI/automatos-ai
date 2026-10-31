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
from typing import Any, Optional, Set, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)

# Strong refs to fire-and-forget notification tasks (the api/webhooks.py
# background-task idiom) so the loop cannot GC them mid-flight.
_NOTIFY_TASKS: Set["asyncio.Task"] = set()

# The reason a task the policy lets run gives (no rule asked, so no rule sentence).
AUTO_APPROVED_REASON = "auto-approved by workspace policy"


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

    PRD-268 US-C20b: a task the policy holds gives the rule that asked, in words: the English
    sentence for ``approval_policy.<policy>`` (:func:`held_reason`) is the outcome's reason, the
    grant's and its bell's. The ticket's blocked line shows it, and the approval view reads the
    policy back from it.

    ``_policy_override`` / ``_ceiling_override`` are test seams that bypass the
    workspace-settings read so the gate can be exercised without a DB row.
    """
    auto_approve, policy = _decide(db, workspace_id, estimated_cost_usd, _policy_override, _ceiling_override)
    if auto_approve:  # auto-approve ⇒ no grant, task runs.
        return _outcome(False, AUTO_APPROVED_REASON, None, policy, estimated_cost_usd)

    reason = held_reason(policy)
    # Needs approval. Idempotency: reuse an existing pending grant for this task.
    existing = _pending_grant(db, workspace_id, task_id)
    if existing is not None:
        return _outcome(True, reason, existing, policy, estimated_cost_usd)

    held = _HeldTask(
        workspace_id=workspace_id, task_id=str(task_id), agent_id=agent_id, risk_tier=risk_tier,
        policy=policy, reason=reason, estimated_cost_usd=estimated_cost_usd,
    )
    return _outcome(True, reason, _open_grant(db, held, ttl_seconds), policy, estimated_cost_usd)


def held_reason(policy: str) -> str:
    """The reason a task ``policy`` holds gives: the English rule sentence for
    ``approval_policy.<policy>`` (``rule_sentences``' fallback for a policy it doesn't know),
    never the policy key."""
    from modules.policy.rule_sentences import approval_policy_rule, rule_sentence
    from modules.proactive.locale import ENGLISH

    return rule_sentence(approval_policy_rule(policy), ENGLISH)


@dataclass(frozen=True)
class _HeldTask:
    """A board task the approval policy holds, as its new grant records it."""

    workspace_id: UUID | str
    task_id: str
    agent_id: Optional[int]
    risk_tier: Optional[str]
    policy: str
    reason: str
    estimated_cost_usd: float


def _outcome(
    requires_approval: bool, reason: str, grant: Optional[Any], policy: str, estimated_cost_usd: float,
) -> BoardApprovalOutcome:
    """The gate's answer, every field named."""
    return BoardApprovalOutcome(
        requires_approval=requires_approval,
        reason=reason,
        grant=grant,
        policy=policy,
        estimated_cost_usd=estimated_cost_usd,
    )


def _decide(
    db: Any,
    workspace_id: UUID | str,
    estimated_cost_usd: float,
    policy_override: Optional[str],
    ceiling_override: Optional[float],
) -> Tuple[bool, str]:
    """Whether the workspace policy auto-approves the task, and that policy.

    Reuses the mission primitive unless a test override is supplied.
    """
    if policy_override is not None:
        return _decide_from_override(policy_override, ceiling_override, estimated_cost_usd), policy_override
    from core.services.approval_policy import evaluate_approval

    decision = evaluate_approval(db, workspace_id, estimated_cost_usd)
    return decision.auto_approve, decision.policy


def _pending_grant(db: Any, workspace_id: UUID | str, task_id: int | str) -> Optional[Any]:
    """The task's pending approval grant in the workspace, if one already waits."""
    from core.models.approval_grants import SUBJECT_BOARD_TASK
    from core.services.approval_grants import find_pending_grant

    return find_pending_grant(db, workspace_id, subject_type=SUBJECT_BOARD_TASK, subject_id=str(task_id))


def _open_grant(db: Any, held: _HeldTask, ttl_seconds: Optional[int]) -> Any:
    """Create the held task's grant, audit it and announce it to the workspace's humans."""
    from core.models.approval_grants import SUBJECT_BOARD_TASK
    from core.services.approval_grants import DEFAULT_TTL_SECONDS, create_grant

    grant = create_grant(
        db, held.workspace_id,
        subject_type=SUBJECT_BOARD_TASK,
        subject_id=held.task_id,
        risk_tier=held.risk_tier,
        agent_id=held.agent_id,
        reason=held.reason,
        estimated_cost_usd=held.estimated_cost_usd,
        ttl_seconds=ttl_seconds if ttl_seconds is not None else DEFAULT_TTL_SECONDS,
    )
    _audit_governance(
        db, held.workspace_id, "approval_grant:created",
        grant_id=getattr(grant, "id", None),
        subject_type=SUBJECT_BOARD_TASK,
        subject_id=held.task_id,
        policy=held.policy,
        estimated_cost_usd=held.estimated_cost_usd,
        risk_tier=held.risk_tier,
    )
    # PRD-196 S2 (C.8): a pending grant must never wait silently — tell the
    # workspace's humans. Only on fresh creation (a reused grant was already
    # announced when it was created).
    _notify_approval_pending(
        held.workspace_id,
        grant_id=getattr(grant, "id", None),
        subject_id=held.task_id,
        risk_tier=held.risk_tier,
        reason=held.reason,
        estimated_cost_usd=held.estimated_cost_usd,
    )
    return grant


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
    transaction is finished (and its session may be closed). The dispatcher
    never commits, so this commits its bell row (closing the session alone
    would roll it back).
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
        db.commit()
    except Exception:
        logger.warning(
            "[board_approval] approval_pending dispatch failed for grant %s",
            grant_id, exc_info=True,
        )
        db.rollback()
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
