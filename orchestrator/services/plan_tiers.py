"""PRD-222 W2·S1 (US-023) — plan-tier resolution + assignment.

The single service seam for the config-driven tier contract (``config.PLAN_TIERS``):
read which tiers are assignable, resolve a tier, derive the ``plan_limits`` a tier
implies, and assign a plan to a workspace. Reused by US-025's proposal-acceptance
tool path so that plan changes flow through ONE writer (FR-4 auditability).

Assignment writes the tier's limits into ``workspaces.plan_limits`` as a NEW dict
(PRD-220 rebuild-don't-mutate), MERGED onto whatever the workspace already carries
so unmanaged keys survive, under the keys the LIVE consumers already read:

  * ``max_members`` — the seat key. ``core/workspaces/invitations.py`` enforces it
    and the onboarding checklist gates the invite item on it, so the tier's
    ``seats`` lands HERE, not in a parallel ``seats`` key.
  * ``max_agents``  — documented concurrency cap (0 = unlimited).
  * ``budget``      — ``{window, max_cost_usd}``, read by ``modules/policy/budget.py``;
    written only when the tier sets a positive ``budget_usd`` (0 = custom / no
    ceiling, so an existing explicit budget is preserved on merge).

``mission_concurrency`` / ``watcher_limit`` / ``marketplace_depth`` are stored too
(the tier's declared limits) but have no enforcement consumer yet — this wave adds
NO quota hardening. No billing anywhere: ``display_price_usd`` is a label only
(PRD §12 Q5). ``enterprise`` is coming-soon and is rejected by :func:`assign_plan`.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _tiers(tiers: Optional[dict] = None) -> dict:
    """The tier map to resolve against — an injected override (tests / callers
    that already loaded an env override) or the module ``config.PLAN_TIERS``."""
    if tiers is not None:
        return tiers
    from config import PLAN_TIERS

    return PLAN_TIERS


def get_tier(plan: str, tiers: Optional[dict] = None) -> Optional[dict]:
    """The tier config for ``plan`` (any tier, incl. non-assignable), or None."""
    return _tiers(tiers).get(plan)


def is_assignable(plan: str, tiers: Optional[dict] = None) -> bool:
    """True only for a tier a workspace can actually be put on — present,
    ``assignable``, and not a coming-soon placeholder (``enterprise``)."""
    tier = _tiers(tiers).get(plan)
    return bool(tier) and bool(tier.get("assignable")) and not tier.get("coming_soon")


def assignable_tiers(tiers: Optional[dict] = None) -> dict:
    """The subset of tiers a workspace can be assigned to (excludes enterprise)."""
    resolved = _tiers(tiers)
    return {name: t for name, t in resolved.items() if is_assignable(name, resolved)}


def plan_limits_for_tier(plan: str, tiers: Optional[dict] = None) -> dict:
    """The ``plan_limits`` fragment a tier implies, keyed for the LIVE consumers.

    Raises :class:`ValueError` for a non-assignable / unknown plan — the caller
    must never write limits for a tier that cannot be assigned.
    """
    resolved = _tiers(tiers)
    if not is_assignable(plan, resolved):
        raise ValueError(f"plan {plan!r} is not assignable")
    tier = resolved[plan]
    limits: dict[str, Any] = {
        "max_members": tier["seats"],
        "max_agents": tier["max_agents"],
        "mission_concurrency": tier["mission_concurrency"],
        "watcher_limit": tier["watcher_limit"],
        "marketplace_depth": tier["marketplace_depth"],
    }
    budget_usd = tier.get("budget_usd") or 0
    if budget_usd and budget_usd > 0:
        limits["budget"] = {"window": "month", "max_cost_usd": float(budget_usd)}
    return limits


def assign_plan(db: Any, workspace: Any, plan: str, tiers: Optional[dict] = None) -> dict:
    """Set ``workspace.plan`` and merge the tier's limits into ``plan_limits``.

    Rebuild-don't-mutate (PRD-220): a NEW ``plan_limits`` dict is built from the
    existing one (so keys this tier does not manage survive) with the tier's
    limits overlaid, then reassigned so SQLAlchemy marks the JSONB column dirty.
    ``db is None`` performs the in-memory assignment only — the escape hatch for
    pure-logic tests (mirrors ``services/onboarding_state._persist``). Returns the
    new ``plan_limits``. Raises :class:`ValueError` for a non-assignable plan.
    """
    limits = plan_limits_for_tier(plan, tiers)  # validates assignability first
    new_limits = dict(workspace.plan_limits or {})
    new_limits.update(limits)
    workspace.plan = plan
    workspace.plan_limits = new_limits
    if db is not None:
        db.add(workspace)
        db.commit()
    logger.info("[plan_tiers] assigned plan=%s to workspace=%s", plan, getattr(workspace, "id", "?"))
    return new_limits
