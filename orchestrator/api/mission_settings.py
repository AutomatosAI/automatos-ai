"""F282 (night 8): "one setting ... a switch on the form and on the mission page".

Auto wrote the owner's wish in eight spellings (``SAYS_CHECK_EACH_STEP`` in
modules/coordination/owner_checks.py), so whether a mission checked each step
depended on how Auto phrased it, Auto sometimes said it was on when it wasn't,
and the New mission form had no switch of its own — the owner could only ask
Auto. The New mission form's switch sets the setting at creation; this route
is what the mission page's matching switch calls to change it afterwards.
Its own router: api/missions.py is over 800 lines and does not grow.
"""
from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.missions import _get_run_for_workspace
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.orchestration_enums import RunState, TERMINAL_RUN_STATES
from modules.coordination.owner_checks import CHECK_EACH_STEP, with_step_checks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/missions", tags=["missions"])


class MissionSettingsRequest(BaseModel):
    check_each_step: bool


@router.patch("/{mission_id}/settings", dependencies=[Depends(require_workspace_permission("missions:update"))])
def update_mission_settings(
    mission_id: UUID,
    body: MissionSettingsRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Turn the owner's check of each step on or off on a mission that can still
    run. The setting is written as ``check_each_step`` from here on, whatever
    spelling the mission started with (``with_step_checks``). A mission already
    completed, failed or cancelled refuses: there is nothing left for the
    setting to change."""
    try:
        run = _get_run_for_workspace(db, mission_id, ctx.workspace_id)
        if RunState(run.state) in TERMINAL_RUN_STATES:
            raise HTTPException(status_code=409,
                                detail=f"Mission is {run.state}; its step-check setting can no longer change.")
        run.config = with_step_checks(run.config, on=body.check_each_step)
        db.commit()
        return {"id": str(run.id), CHECK_EACH_STEP: body.check_each_step}
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        logger.error("Failed to update settings for mission %s: %s", mission_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
