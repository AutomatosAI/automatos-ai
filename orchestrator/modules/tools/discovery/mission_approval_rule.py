"""PRD-268 §6.11.4 — a Mission plan's card names the approval policy that held it.

While a new plan awaits approval, ``create_mission``'s answer carries ``approval_policy``, so the
card's ``ApprovalView@1`` (``modules/tools/formatting/approval_view.py``) says the rule in words
(``approval_policy.<policy>``, ``modules/policy/rule_sentences.py``). The policy is named only
when it would hold this plan: ``evaluate_approval`` is asked again, as the coordinator asked it
(``services/coordinator_service.py``), with the plan's recorded estimate
(``approval_estimated_cost_usd`` on the run) under the surface the run was stamped from
(``origin_surface``: a widget-born plan is never autonomous). A plan asked for as a plan only, or
one the policy would have let start (the day's spend ceiling held it), names none, and its card
reads the plain fallback sentence. The reads run in a savepoint, so a failed one never aborts the
turn's transaction; it names none, and is logged.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

__all__ = ["APPROVAL_POLICY", "names_the_approval_policy", "policy_holding"]

APPROVAL_POLICY = "approval_policy"
ESTIMATE = "approval_estimated_cost_usd"  # stamped on the run's config when it parks (coordinator_service)
PLAN_ONLY = "plan_only"  # the coordinator parks such a plan whatever the policy
Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]


def names_the_approval_policy(handler: Handler) -> Handler:
    """A Mission tool's answer for a plan awaiting approval names the policy that held it."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        result = await handler(db, workspace_id, params)
        if not (isinstance(result, dict) and result.get("awaiting_approval") and result.get("mission_id")):
            return result
        policy = policy_holding(db, workspace_id, result["mission_id"])
        return {**result, APPROVAL_POLICY: policy} if policy else result
    return wrapped


def policy_holding(db: Session, workspace_id: Any, mission_id: Any) -> Optional[str]:
    """The approval policy holding the Mission's plan, or None: it would start, it was asked for as a
    plan only, or the read failed (logged). The card and ``GET /api/v1/approval-grants/{id}/view`` read it."""
    try:
        return _holding_policy(db, workspace_id, mission_id)
    except Exception:
        logger.exception("[mission_approval_rule] could not read the policy holding mission %s", mission_id)
        return None


def _holding_policy(db: Session, workspace_id: Any, mission_id: Any) -> Optional[str]:
    """The workspace's approval policy when it would hold this plan; None when it would let it start."""
    from core.models.orchestration import OrchestrationRun
    from core.security.surface import origin_surface
    from core.services.approval_policy import evaluate_approval

    with db.begin_nested():
        config = (db.query(OrchestrationRun.config)
                  .filter(OrchestrationRun.id == mission_id, OrchestrationRun.workspace_id == workspace_id)
                  .scalar()) or {}
        if config.get(PLAN_ONLY):
            return None
        estimate = config.get(ESTIMATE)
        with origin_surface(config):
            decision = evaluate_approval(db, workspace_id,
                                         float(estimate) if isinstance(estimate, (int, float)) else 0.0)
    return None if decision.auto_approve else decision.policy
