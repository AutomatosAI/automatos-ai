"""
Approve the week (PRD-251C C2; US-C205)
=======================================

``POST /api/socials/plans/{plan_id}/batches/{batch_key}/approve`` (``socials:approve``): a
weekly or monthly plan's batch, approved in one sitting. The body lists every post the
approver was shown, each with the ``content_hash`` of the version shown. Each post of the
batch still waiting for approval with that hash is approved as a single approval would
approve it (D6, D7), and with its slot still ahead it is scheduled into it. A post that
changed since it was shown is left and listed, to approve on its own after a second look; so
is a post of another batch or another plan, and each of the batch's posts the approver was
not shown.

It runs the series approval core (``campaigns.approve_series``) without the workspace's
series approval switch (O2): every post is still approved by its own hash. That switch stays
for campaigns that are not plans. Another workspace's plan is a 404.

``api/socials_plans.py`` includes this router, so it takes the Socials router's prefix and
gate. A plain ``def`` (F105).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from api.socials_campaigns import ShownPostRequest
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from modules.socials import campaigns, schedule_jobs, service

router = APIRouter()

CAN_REVIEW = Depends(require_workspace_permission("socials:approve"))
BATCH_KEY = re.compile(r"^\d{4}-(W\d{2}|\d{2})$")
BAD_BATCH_KEY = "batch_key is a week (2026-W42) or a month (2026-11)"


class ApproveBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    posts: List[ShownPostRequest] = Field(..., min_length=1, max_length=campaigns.SERIES_MAX_POSTS)
    comment: Optional[str] = Field(None, max_length=service.COMMENT_MAX_CHARS)


def _plans_api() -> Any:
    """``api/socials_plans.py``: it includes this router, so it is imported when a request runs."""
    from api import socials_plans

    return socials_plans


def _sync_jobs(db: Session, workspace_id: UUID, approved: List[Dict[str, Any]]) -> None:
    """Each approved post's publish job follows its slot now (the reconcile pass would too)."""
    for row in approved:
        post = service.get_post(db, workspace_id, UUID(row["id"]))
        if post is not None:
            schedule_jobs.sync_job(post)


@router.post("/plans/{plan_id}/batches/{batch_key}/approve", dependencies=[CAN_REVIEW])
def approve_social_plan_batch(
    plan_id: UUID,
    batch_key: str,
    body: ApproveBatchRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Approve the week: ``{plan_id, batch_key, approved, left}`` (the module docstring)."""
    plans_api = _plans_api()
    plan = plans_api.load_plan(db, ctx, plan_id)
    if not BATCH_KEY.match(batch_key):
        raise HTTPException(status_code=422, detail=BAD_BATCH_KEY)
    actor = plans_api._posts_api()._actor(ctx)
    shown = [campaigns.ShownPost(item.post_id, item.content_hash, item.override_unsourced) for item in body.posts]
    try:
        result = campaigns.approve_series(db, plan, actor, shown, comment=body.comment, batch_key=batch_key)
    except campaigns.CampaignNotFound as exc:  # the plan was deleted while the approval ran
        raise HTTPException(status_code=404, detail=plans_api.PLAN_NOT_FOUND) from exc
    except service.SocialsError as exc:
        plans_api._raise_for(exc)
    _sync_jobs(db, ctx.workspace_id, result["approved"])
    return {"plan_id": str(plan.id), "batch_key": batch_key, "approved": result["approved"], "left": result["left"]}
