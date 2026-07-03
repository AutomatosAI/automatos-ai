"""PRD-181 S2 (F060) — board-task approval gate.

Closes the review gap "governance covers missions only — board tasks Auto
creates have no ceiling or approval gate". A board task about to execute is run
through the **same** ``evaluate_approval`` primitive missions use (PRD-163):

  - the workspace policy auto-approves ⇒ the task runs (no grant needed);
  - the policy asks (always_ask, or over the dollar ceiling) ⇒ a durable,
    revocable, expiring :class:`ApprovalGrant` is created and the task is
    **blocked** until a human grants it — not hard-blocked, not auto-allowed.

The grant is the tool-agnostic record the future scheduled/webhook agents share.
Every grant creation is audited (governance action).

This module is the decision glue only; the dispatch wiring (moving the board task
to ``blocked`` and re-queuing it on grant) lives in ``api.board_tasks`` and
``services.board_dispatcher`` which call :func:`evaluate_board_task_approval`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


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
    return BoardApprovalOutcome(
        requires_approval=True,
        reason=reason,
        grant=grant,
        policy=policy,
        estimated_cost_usd=estimated_cost_usd,
    )


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
