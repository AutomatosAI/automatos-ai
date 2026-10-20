"""Auto records the plan a person chooses; it never puts them on it (PRD-265 US-006, FR-4).

Before billing, ``platform_update_onboarding`` with ``plan`` wrote the plan and
its limits straight onto the workspace — so Auto could hand anyone Business, free.
Every plan is paid now (Own keys included), and a workspace moves plan only on a
verified payment or a platform admin's action. So the tool's ``plan`` is a
CHOICE: it is validated, stamped as the ``plan_accepted`` funnel event, and
answered with how the owner buys it:

- while billing is closed (the beta), nothing is charged and nothing changes —
  the workspace runs on its trial credits, then on the owner's own AI keys;
- once billing opens, a self-serve plan (Own keys, Pro) is bought on the Billing
  tab; Business is by invite.

:func:`plan_is_chosen_never_assigned` wraps the handler, so the plan never
reaches the stage machine's writer.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

PLAN_KEY = "plan"
_INVITE_COPY = (
    "Business is by invite during the beta: the owner asks Automatos for an invite. "
    "Nothing changes in the workspace until then."
)
_BETA_COPY = (
    "Billing isn't open yet — beta pricing, nothing is charged until it opens. Until then the "
    "workspace runs on its trial credits and then on the owner's own AI keys; the plan starts "
    "when billing opens."
)


def _refusal(plan: str) -> Optional[Dict[str, Any]]:
    from services.plan_tiers import is_assignable

    if is_assignable(plan):
        return None
    return {"success": False, "error": (
        "Enterprise is coming soon — it can't be chosen yet. Choose Own keys, Pro or Business."
        if plan == "enterprise" else f"'{plan}' isn't a plan — choose Own keys, Pro or Business."
    )}


def how_to_buy(plan: str) -> Dict[str, Any]:
    """What happens next for a chosen ``plan`` (no state changes)."""
    from config import config
    from services.plan_tiers import get_tier, is_self_serve, purchase_path

    name = (get_tier(plan) or {}).get("display_name") or plan.title()
    if not is_self_serve(plan):
        return {"plan": plan, "name": name, "buy": None, "message": _INVITE_COPY}
    if not config.BILLING_OPEN:
        return {"plan": plan, "name": name, "buy": None, "message": _BETA_COPY}
    path = purchase_path(plan)
    return {"plan": plan, "name": name, "buy": path,
            "message": f"The owner buys {name} on the Billing tab ({path}); it starts once paid."}


def record_plan_choice(db: Any, workspace_id: UUID, raw_plan: Any) -> Dict[str, Any]:
    """Validate ``raw_plan``, stamp ``plan_accepted`` and answer how to buy it.
    The workspace's plan and limits are never written here."""
    from core.models.workspaces import Workspace
    from services.onboarding_state import public_snapshot, record_plan_event
    from services.plan_tiers import tier_key

    plan = tier_key(raw_plan)
    refused = _refusal(plan) if plan else {"success": False, "error": "Name a plan: Own keys, Pro or Business."}
    if refused:
        return refused
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first() if db is not None else None
    if workspace is None:
        return {"success": False, "error": "workspace not found"}
    try:
        record_plan_event(db, workspace, "plan_accepted", plan)
    except Exception:  # noqa: BLE001 — a clean tool error, never a crash; nothing is half-written (FR-4)
        logger.exception("[update_onboarding] recording the plan choice failed")
        db.rollback()
        return {"success": False, "error": "The plan choice couldn't be recorded — try again."}
    return {"success": True, "data": public_snapshot(workspace), PLAN_KEY: how_to_buy(plan)}


def plan_is_chosen_never_assigned(
    handler: Callable[..., Awaitable[Dict[str, Any]]],
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """Wrap ``update_onboarding``: a ``plan`` is taken out of the call and recorded
    as a choice; whatever else the call carries runs as before."""
    @functools.wraps(handler)
    async def wrapped(db: Any, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
        given = params if isinstance(params, dict) else {}
        if given.get(PLAN_KEY) is None:
            return await handler(db, workspace_id, params)
        rest = {k: v for k, v in given.items() if k != PLAN_KEY}
        if not any(v is not None for v in rest.values()):
            return record_plan_choice(db, workspace_id, given[PLAN_KEY])
        result = await handler(db, workspace_id, rest)
        if not result.get("success"):
            return result
        choice = record_plan_choice(db, workspace_id, given[PLAN_KEY])
        if not choice.get("success"):
            return {**result, "plan_error": choice.get("error")}
        return {**result, "data": choice["data"], PLAN_KEY: choice[PLAN_KEY]}
    return wrapped
