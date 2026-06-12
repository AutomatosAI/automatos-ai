"""Mission approval policy — per-workspace control over when a mission plan
auto-approves vs waits for a human (PRD-163 S3).

Single canonical reader/writer for ``workspace.settings.approval_policy``. Mirrors
the ``auto_autonomy`` service pattern (settings live on the Workspace JSON column;
the caller owns the transaction).

Settings shape (``workspace.settings.approval_policy``):

    {
      "policy": "always_ask" | "auto_below_budget" | "full_auto",
      "approval_dollar_ceiling": 5.0,          # $ ceiling for auto_below_budget
      "auto_proceed_after_seconds": null | 30  # countdown auto-proceed (Devin-style)
    }

Policy levels (OpenHands-style):

    always_ask (default)
        Every plan waits for explicit human approval.
    auto_below_budget
        Auto-approve when the plan's estimated cost is at/below the $ ceiling;
        otherwise wait for a human.
    full_auto
        Auto-approve everything — but ONLY when the §12.3 autonomy gate
        (``auto_autonomy.is_full_autonomy``) is on. With the gate off, full_auto
        downgrades to "ask" (fail-safe: never run unsupervised without the gate).

Fail-safe: an unreadable / corrupt setting falls back to ``always_ask``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

ALWAYS_ASK = "always_ask"
AUTO_BELOW_BUDGET = "auto_below_budget"
FULL_AUTO = "full_auto"
VALID_POLICIES = frozenset({ALWAYS_ASK, AUTO_BELOW_BUDGET, FULL_AUTO})

DEFAULTS: Dict[str, Any] = {
    "policy": ALWAYS_ASK,
    "approval_dollar_ceiling": None,
    "auto_proceed_after_seconds": None,
}


def load_approval_policy(db: Session, workspace_id: UUID | str) -> Dict[str, Any]:
    """Return the workspace's ``approval_policy`` settings merged onto defaults."""
    from core.models.workspaces import Workspace

    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if ws is None:
        return dict(DEFAULTS)

    cfg = (ws.settings or {}).get("approval_policy") or {}
    policy = cfg.get("policy")
    if not isinstance(policy, str) or policy not in VALID_POLICIES:
        policy = ALWAYS_ASK

    ceiling = cfg.get("approval_dollar_ceiling")
    ceiling = float(ceiling) if isinstance(ceiling, (int, float)) else None

    countdown = cfg.get("auto_proceed_after_seconds")
    countdown = int(countdown) if isinstance(countdown, (int, float)) and countdown > 0 else None

    return {
        "policy": policy,
        "approval_dollar_ceiling": ceiling,
        "auto_proceed_after_seconds": countdown,
    }


def set_approval_policy(
    db: Session,
    workspace_id: UUID | str,
    *,
    policy: Optional[str] = None,
    approval_dollar_ceiling: Optional[float] = None,
    auto_proceed_after_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Persist approval-policy fields to ``workspace.settings.approval_policy``.

    Only provided fields are changed. Caller owns the transaction (stages + flushes).
    """
    from core.models.workspaces import Workspace
    from sqlalchemy.orm.attributes import flag_modified

    if policy is not None and policy not in VALID_POLICIES:
        raise ValueError(f"invalid policy {policy!r}; expected one of {sorted(VALID_POLICIES)}")

    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if ws is None:
        raise ValueError(f"workspace {workspace_id} not found")

    settings = dict(ws.settings or {})
    current = dict(settings.get("approval_policy") or {})
    if policy is not None:
        current["policy"] = policy
    if approval_dollar_ceiling is not None:
        current["approval_dollar_ceiling"] = float(approval_dollar_ceiling)
    if auto_proceed_after_seconds is not None:
        current["auto_proceed_after_seconds"] = int(auto_proceed_after_seconds)
    settings["approval_policy"] = current
    ws.settings = settings
    flag_modified(ws, "settings")
    db.flush()
    return load_approval_policy(db, workspace_id)


@dataclass(frozen=True)
class ApprovalDecision:
    """Outcome of evaluating a plan against the workspace approval policy."""

    auto_approve: bool
    reason: str
    policy: str
    ceiling: Optional[float]
    estimated_cost: float
    countdown_seconds: Optional[int]  # when awaiting, auto-proceed after N s (else None)

    def audit_snapshot(self) -> Dict[str, Any]:
        return {
            "auto_approved": self.auto_approve,
            "policy": self.policy,
            "approval_dollar_ceiling": self.ceiling,
            "estimated_cost_usd": round(self.estimated_cost, 4),
            "countdown_seconds": self.countdown_seconds,
            "reason": self.reason,
        }


def evaluate_approval(
    db: Session,
    workspace_id: UUID | str,
    estimated_cost_usd: float,
    *,
    override_auto_approve: bool = False,
) -> ApprovalDecision:
    """Decide whether a plan auto-approves under the workspace policy.

    :param estimated_cost_usd: the plan's estimated dollar cost.
    :param override_auto_approve: a per-request 'auto-approve this one' from chat —
        forces auto-approval regardless of policy.
    """
    cfg = load_approval_policy(db, workspace_id)
    policy = cfg["policy"]
    ceiling = cfg["approval_dollar_ceiling"]
    countdown = cfg["auto_proceed_after_seconds"]

    def _ask(reason: str) -> ApprovalDecision:
        return ApprovalDecision(False, reason, policy, ceiling, estimated_cost_usd, countdown)

    def _approve(reason: str) -> ApprovalDecision:
        return ApprovalDecision(True, reason, policy, ceiling, estimated_cost_usd, None)

    if override_auto_approve:
        return _approve("per-request override (auto-approve this one)")

    if policy == FULL_AUTO:
        from core.services.auto_autonomy import is_full_autonomy

        if is_full_autonomy(db, workspace_id):
            return _approve("full_auto policy with autonomy gate on")
        # Fail-safe: full_auto without the §12.3 gate must NOT run unsupervised.
        return _ask("full_auto requires the autonomy gate, which is off")

    if policy == AUTO_BELOW_BUDGET:
        if ceiling is not None and estimated_cost_usd <= ceiling:
            return _approve(f"cost ${estimated_cost_usd:.2f} <= ceiling ${ceiling:.2f}")
        return _ask(
            f"cost ${estimated_cost_usd:.2f} exceeds ceiling "
            f"${ceiling:.2f}" if ceiling is not None else "no budget ceiling set"
        )

    return _ask("policy=always_ask")
