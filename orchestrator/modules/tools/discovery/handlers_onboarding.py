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
    public_snapshot,
    set_segment,
)

logger = logging.getLogger(__name__)


async def update_onboarding(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Advance the onboarding spine and/or record segment answers.

    Params: ``advance_to`` (next stage) and/or ``segment`` ({business, goal,
    comfort}). At least one is required. Returns ``{success, data: {stage, trial}}``.
    """
    advance_to = params.get("advance_to")
    segment = params.get("segment")

    if not advance_to and not segment:
        return {
            "success": False,
            "error": "Provide advance_to or segment — at least one is required.",
        }

    workspace = (
        db.query(Workspace).filter(Workspace.id == workspace_id).first()
    )
    if not workspace:
        return {"success": False, "error": "workspace not found"}

    try:
        if advance_to:
            # A single write advances the stage and merges any segment answers.
            advance_onboarding_stage(db, workspace, advance_to, segment=segment)
        else:
            set_segment(db, workspace, segment)
    except InvalidStageTransition as exc:
        return {"success": False, "error": str(exc)}
    except ValueError as exc:
        # e.g. set_segment called with no recognised keys.
        return {"success": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - surface a clean tool error, never crash
        logger.error("[update_onboarding] failed: %s", exc, exc_info=True)
        db.rollback()
        return {"success": False, "error": str(exc)}

    return {"success": True, "data": public_snapshot(workspace)}
