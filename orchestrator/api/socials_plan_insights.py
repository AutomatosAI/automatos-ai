"""
Socials plan insights (PRD-251C C7, C9; US-C404, US-C407)
=========================================================

* ``GET /api/socials/plans/{plan_id}/proposals``: Auto's proposals from the plan's results
  (``modules/socials/proposals.py``), each a change the Plan page applies by one click
  through ``PUT /api/socials/plans/{plan_id}``. Nothing is applied here.
* ``GET /api/socials/plans/{plan_id}/health``: what needs the owner now, each item with the
  one action that fixes it (``modules/socials/plan_health.py``).

Reads only. Another workspace's plan is a 404. ``api/socials_plans.py`` includes this
router, so it takes the Socials router's prefix and gate. Plain ``def`` routes (F105).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.workspaces import Workspace
from modules.socials import plan_health, proposals

router = APIRouter()


def _plans_api() -> Any:
    """``api/socials_plans.py``: it includes this router, so it is imported when a request runs."""
    from api import socials_plans

    return socials_plans


@router.get("/plans/{plan_id}/proposals")
def get_social_plan_proposals(
    plan_id: UUID, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """The plan's proposals: ``{proposals: [{id, kind, title, why, changes}]}``."""
    plan = _plans_api().load_plan(db, ctx, plan_id)
    since = datetime.now(timezone.utc) - timedelta(days=proposals.LOOKBACK_DAYS)
    return {"proposals": [item.to_dict() for item in proposals.proposals(plan, proposals.read_posts(db, plan, since))]}


@router.get("/plans/{plan_id}/health")
def get_social_plan_health(
    plan_id: UUID, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """The plan's health: ``{items: [{id, title, detail, action: {kind, label}}]}``, most pressing first."""
    plan = _plans_api().load_plan(db, ctx, plan_id)
    workspace = db.get(Workspace, ctx.workspace_id)
    return {"items": [item.to_dict() for item in plan_health.health_for(db, plan, workspace, datetime.now(timezone.utc))]}
