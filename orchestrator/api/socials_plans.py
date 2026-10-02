"""
Socials plans (PRD-251B B6, B7; US-B202, US-B208)
=================================================

A plan is a campaign of kind ``plan`` (``modules/socials/plans.py``):

* ``GET /api/socials/plans``: the workspace's plans, newest first, each with its bank's
  counts; ``POST`` creates one (name, dates, timezone and cadence required).
* ``GET /api/socials/plans/{plan_id}``: the plan with its bank's counts; ``PUT`` changes
  any of its fields, each checked (known channels and formats, lengths the chosen
  template declares, times and days); an ended plan is read-only.
* ``POST /api/socials/plans/{plan_id}/pause`` · ``/resume`` · ``/end``.
* ``POST /api/socials/plans/{plan_id}/research``: **Research again** (US-B204): the
  workspace's Content bank research playbook runs for the plan now (202 with its
  execution id); 409 when the Socials package that carries it is not installed.
* ``GET /api/socials/plans/{plan_id}/slots?start&end``: the planned and made slots in a
  window of at most 62 days; ``PUT .../slots/{slot_key}`` moves a planned slot or skips
  it (``slot_overrides``), the cadence untouched.

The content bank's routes are ``api/socials_topics.py``, included here. Every read and
write is scoped to the caller's workspace: another workspace's plan is a 404. Every route
is a plain ``def`` (F105). This router has no prefix and no gate of its own:
``api/socials.py`` includes it in the Socials router, whose ``require_socials_enabled``
answers 404 unless both switches are on. Never mount it in the app directly.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List, NoReturn, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from api.socials_topics import router as topics_router
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.socials import SocialCampaign
from modules.socials import campaigns, plan_store, plans, service
from services import socials_plan_research

router = APIRouter()
router.include_router(topics_router)

CAN_CREATE = Depends(require_workspace_permission("documents:create"))
CAN_UPDATE = Depends(require_workspace_permission("documents:update"))
PLAN_NOT_FOUND = "Plan not found"
SLOT_KEY_MAX_CHARS = 160


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CadenceRow(_Strict):
    id: Optional[str] = Field(None, max_length=32)
    channels: List[str] = Field(..., min_length=1, max_length=plans.MAX_ROW_CHANNELS)
    format: str
    length_seconds: Optional[int] = None
    template_id: Optional[UUID] = None
    days: List[str] = Field(..., min_length=1, max_length=7)
    time: str


class PlanSources(_Strict):
    knowledge: bool = True
    deliverables: bool = True
    website: bool = True
    github: bool = False
    notes: Optional[str] = Field(None, max_length=plans.NOTES_MAX_CHARS)
    never_say: List[str] = Field(default_factory=list, max_length=plans.MAX_NEVER_SAY)


class PlanMake(_Strict):
    time: str = plans.DEFAULT_MAKE_TIME
    video_days_early: int = plans.DEFAULT_VIDEO_DAYS_EARLY
    max_per_day: Optional[int] = None
    visual_mix: Optional[Dict[str, int]] = None


class PlanResearch(_Strict):
    enabled: bool = True
    day: str = "mon"
    time: str = "06:00"


class PlanFields(_Strict):
    name: Optional[str] = Field(None, min_length=1, max_length=campaigns.CAMPAIGN_NAME_MAX_CHARS)
    goal: Optional[str] = Field(None, max_length=plans.GOAL_MAX_CHARS)
    audience: Optional[str] = Field(None, max_length=plans.AUDIENCE_MAX_CHARS)
    starts_on: Optional[date] = None
    ends_on: Optional[date] = None
    timezone: Optional[str] = Field(None, max_length=64)
    cadence: Optional[List[CadenceRow]] = Field(None, max_length=plans.MAX_CADENCE_ROWS)
    sources: Optional[PlanSources] = None
    make: Optional[PlanMake] = None
    research: Optional[PlanResearch] = None
    late_policy: Optional[str] = None
    approval_mode: Optional[str] = None


class SlotMoveRequest(_Strict):
    # The new time (with its timezone), or null to put the slot back; skip=true skips it.
    to: Optional[datetime] = None
    skip: bool = False


def _fields(body: PlanFields) -> Dict[str, Any]:
    """The fields the body set, as plain data (a row's template id as text)."""
    fields = body.model_dump(exclude_unset=True, mode="python")
    if fields.get("cadence") is not None:
        fields["cadence"] = [{**row, "template_id": str(row["template_id"]) if row.get("template_id") else None} for row in fields["cadence"]]
    return fields


def _posts_api() -> Any:
    """``api/socials.py``: its router includes this one, so it is imported when a request runs."""
    from api import socials

    return socials


def _raise_for(exc: Exception) -> NoReturn:
    if isinstance(exc, plans.PlanNotFound):
        raise HTTPException(status_code=404, detail=PLAN_NOT_FOUND)
    if isinstance(exc, socials_plan_research.ResearchUnavailable):
        raise HTTPException(status_code=409, detail=str(exc))
    _posts_api()._raise_for(exc)


def load_plan(db: Session, ctx: RequestContext, plan_id: UUID) -> SocialCampaign:
    plan = plan_store.get_plan(db, ctx.workspace_id, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail=PLAN_NOT_FOUND)
    return plan


def plan_view(db: Session, plan: SocialCampaign) -> Dict[str, Any]:
    counts = plan_store.bank_counts(db, [plan.id]).get(plan.id, {"topics": 0, "unused": 0})
    return {**plan.to_dict(), "bank": counts}


def _saved(db: Session, plan: SocialCampaign) -> Dict[str, Any]:
    db.commit()
    db.refresh(plan)
    return plan_view(db, plan)


@router.get("/plans")
def list_social_plans(db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    """The workspace's plans, newest first, each with ``bank`` ({topics, unused})."""
    rows = plan_store.list_plans(db, ctx.workspace_id)
    counts = plan_store.bank_counts(db, [plan.id for plan in rows])
    plans_out = [{**plan.to_dict(), "bank": counts.get(plan.id, {"topics": 0, "unused": 0})} for plan in rows]
    return {"plans": plans_out, "total": len(plans_out)}


@router.post("/plans", status_code=201, dependencies=[CAN_CREATE])
def create_social_plan(
    body: PlanFields, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)
) -> Dict[str, Any]:
    """A new active plan (name, starts_on, ends_on, timezone and cadence required)."""
    try:
        plan = plan_store.create_plan(db, workspace_id=ctx.workspace_id, created_by=_posts_api()._actor(ctx), fields=_fields(body))
    except service.SocialsError as exc:
        db.rollback()
        _raise_for(exc)
    return _saved(db, plan)


@router.get("/plans/{plan_id}")
def get_social_plan(plan_id: UUID, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    return plan_view(db, load_plan(db, ctx, plan_id))


@router.put("/plans/{plan_id}", dependencies=[CAN_UPDATE])
def update_social_plan(
    plan_id: UUID, body: PlanFields, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)
) -> Dict[str, Any]:
    """Change any of the plan's fields, each checked; an ended plan is read-only (422)."""
    plan = load_plan(db, ctx, plan_id)
    try:
        plan_store.update_plan(db, plan, _fields(body))
    except service.SocialsError as exc:
        db.rollback()
        _raise_for(exc)
    return _saved(db, plan)


def _set_status(db: Session, ctx: RequestContext, plan_id: UUID, status: str) -> Dict[str, Any]:
    plan = load_plan(db, ctx, plan_id)
    try:
        plan_store.set_status(plan, status)
    except service.SocialsError as exc:
        _raise_for(exc)
    return _saved(db, plan)


@router.post("/plans/{plan_id}/pause", dependencies=[CAN_UPDATE])
def pause_social_plan(plan_id: UUID, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    """Stop making posts (and researching) until resumed; posts already made stay."""
    return _set_status(db, ctx, plan_id, plans.PAUSED)


@router.post("/plans/{plan_id}/resume", dependencies=[CAN_UPDATE])
def resume_social_plan(plan_id: UUID, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    return _set_status(db, ctx, plan_id, plans.ACTIVE)


@router.post("/plans/{plan_id}/end", dependencies=[CAN_UPDATE])
def end_social_plan(plan_id: UUID, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    """End the plan for good: no more posts are made; it stays readable."""
    return _set_status(db, ctx, plan_id, plans.ENDED)


@router.get("/plans/{plan_id}/slots")
def list_social_plan_slots(
    plan_id: UUID,
    start: datetime = Query(..., description="ISO datetime with its timezone"),
    end: datetime = Query(..., description="ISO datetime with its timezone, at most 62 days after start"),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """The plan's slots in [start, end): ``planned`` ones (with the topic pinned to their
    day) and ``made`` ones (with their post)."""
    plan = load_plan(db, ctx, plan_id)
    try:
        start, end = plans.window(start, end)
    except service.SocialsError as exc:
        _raise_for(exc)
    return {"plan_id": str(plan.id), "slots": plan_store.plan_slots(db, plan, start, end)}


@router.put("/plans/{plan_id}/slots/{slot_key:path}", dependencies=[CAN_UPDATE])
def move_social_plan_slot(
    plan_id: UUID,
    slot_key: str,
    body: SlotMoveRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Move a planned slot to ``to``, put it back (``to`` null), or skip it (B208). A slot
    whose post is made already moves with the post, never here (409)."""
    plan = load_plan(db, ctx, plan_id)
    if len(slot_key) > SLOT_KEY_MAX_CHARS:
        raise HTTPException(status_code=422, detail="slot_key is too long")
    if plan_store.made_posts(db, plan, [slot_key]):
        raise HTTPException(status_code=409, detail="This slot's post is made: move the post instead")
    try:
        moved = plan_store.move_slot(plan, slot_key, to=body.to, skip=body.skip)
    except service.SocialsError as exc:
        _raise_for(exc)
    db.commit()
    return moved


@router.post("/plans/{plan_id}/research", status_code=202, dependencies=[CAN_UPDATE])
def research_social_plan(plan_id: UUID, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    """Research again (US-B204): the plan's research run starts now; its execution id."""
    plan = load_plan(db, ctx, plan_id)
    if plan.status == plans.ENDED:
        raise HTTPException(status_code=422, detail="An ended plan is not researched")
    try:
        execution_id = socials_plan_research.launch(
            db, plan, triggered_by=f"user:{_posts_api()._actor(ctx)}", now=datetime.now(timezone.utc)
        )
    except service.SocialsError as exc:
        db.rollback()
        _raise_for(exc)
    return {"plan_id": str(plan.id), "execution_id": execution_id}
