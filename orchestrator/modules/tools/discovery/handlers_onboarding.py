"""Onboarding handlers for PlatformActionExecutor (PRD-222 W1S3).

Delegates all state changes to ``services.onboarding_state`` — the single writer
of the stage machine — and returns the client-safe ``{stage, trial}`` snapshot.
Invalid transitions are returned as clear errors, never raised as crashes.
"""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.workspaces import Workspace
from services.onboarding_state import (
    InvalidStageTransition,
    advance_onboarding_stage,
    get_onboarding,
    public_snapshot,
    record_plan_event,
    set_segment,
)

logger = logging.getLogger(__name__)


async def update_onboarding(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Advance the onboarding spine, record segment answers, and/or set the plan.

    Params: ``advance_to`` (next stage), ``segment`` ({business, goal, comfort}),
    and/or ``plan`` (the accepted tier — basic/pro/business only). At least one is
    required. Setting a plan writes plan + plan_limits through the US-023 helper
    (the single writer) and stamps the ``plan_accepted`` funnel event; advancing
    to the proposal stamps ``plan_recommended``. Returns ``{success, data:
    {stage, trial}}``.
    """
    advance_to = params.get("advance_to")
    segment = params.get("segment")
    plan = params.get("plan")

    if not advance_to and not segment and not plan:
        return {
            "success": False,
            "error": "Provide advance_to, segment, or plan — at least one is required.",
        }

    workspace = (
        db.query(Workspace).filter(Workspace.id == workspace_id).first()
    )
    if not workspace:
        return {"success": False, "error": "workspace not found"}

    # Reject a non-assignable plan BEFORE any write — honest coming-soon copy.
    if plan is not None:
        from services.plan_tiers import is_assignable

        if not is_assignable(plan):
            return {
                "success": False,
                "error": (
                    f"'{plan}' can't be assigned yet — Enterprise is coming soon. "
                    "Choose basic, pro, or business."
                ),
            }

    # ATOMICITY (FR-4, RVW-2): every writer below runs with commit=False and the
    # WHOLE tool call is committed ONCE at the end. A state change (plan +
    # plan_limits, or a stage advance) and its funnel audit stamp therefore land
    # together or not at all — a durable change can never be reported success:False
    # (nothing committed on failure) nor left without its recorded audit event.
    try:
        if advance_to:
            # A single write advances the stage and merges any segment answers.
            advance_onboarding_stage(db, workspace, advance_to, segment=segment, commit=False)
        elif segment:
            set_segment(db, workspace, segment, commit=False)

        # Funnel: reaching the proposal records the plan Auto will recommend
        # (derived from the stored segment). Auditable, FR-4.
        if advance_to == "proposal":
            from services.plan_tiers import recommend_plan

            rec_plan, _reason = recommend_plan(get_onboarding(workspace).get("segment") or {})
            record_plan_event(db, workspace, "plan_recommended", rec_plan, commit=False)

        # Accepting a plan writes plan + plan_limits through the US-023 helper and
        # stamps plan_accepted — state changes only through this tool (FR-4).
        if plan is not None:
            from services.plan_tiers import assign_plan

            assign_plan(db, workspace, plan, commit=False)
            record_plan_event(db, workspace, "plan_accepted", plan, commit=False)

        # The single commit — all deferred writes land as one transaction.
        if db is not None:
            db.commit()
    except InvalidStageTransition as exc:
        if db is not None:
            db.rollback()
        return {"success": False, "error": str(exc)}
    except ValueError as exc:
        # e.g. set_segment called with no recognised keys.
        if db is not None:
            db.rollback()
        return {"success": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - surface a clean tool error, never crash
        logger.error("[update_onboarding] failed: %s", exc, exc_info=True)
        if db is not None:
            db.rollback()
        return {"success": False, "error": str(exc)}

    return {"success": True, "data": public_snapshot(workspace)}
