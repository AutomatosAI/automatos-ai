"""
Socials content bank (PRD-251B B8; US-B203)
===========================================

A plan's topics (``modules/socials/topics.py``):

* ``GET /api/socials/plans/{plan_id}/topics``: the bank, unused topics first, each with
  ``repeat`` when the workspace's history holds a post close to it ({post_id, note}, PRD-251C
  US-C106), ``research_note``: why research cannot run in the workspace, or null when it
  can (US-C101, ``services/socials_research_setup.py``), and ``research_last_run_at``: when
  the plan's research last started (US-C207).
* ``POST`` adds a topic; ``PUT .../topics/{topic_id}`` edits it; ``DELETE`` removes it;
  ``PUT .../topics/{topic_id}/pin`` pins it to a day (or unpins it).

A fact without a source, a title the bank already holds, or a "never say" phrase of the
plan is refused with 422 and the reason. A person's topic close to one in any of the
workspace's banks, or to a recent post, is added with a ``warning`` naming it (PRD-251C
US-C104: research's would be refused). Plain ``def`` routes (F105), scoped to the
caller's workspace through the plan. ``api/socials_plans.py`` includes this router, so
it takes the Socials router's prefix and gate.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, NoReturn, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from modules.socials import repeats, service, topics
from services import socials_research_setup

router = APIRouter()

CAN_UPDATE = Depends(require_workspace_permission("documents:update"))
TOPIC_NOT_FOUND = "Topic not found"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FactSource(_Strict):
    kind: str
    ref: str = Field(..., max_length=topics.REF_MAX_CHARS)
    label: str = Field(..., max_length=topics.LABEL_MAX_CHARS)


class Fact(_Strict):
    text: str = Field(..., max_length=topics.FACT_MAX_CHARS)
    # Optional here so the bank's own check answers with its reason: every fact names a source.
    source: Optional[FactSource] = None


class TopicFields(_Strict):
    title: Optional[str] = Field(None, max_length=topics.TITLE_MAX_CHARS)
    angle: Optional[str] = Field(None, max_length=topics.ANGLE_MAX_CHARS)
    facts: Optional[List[Fact]] = Field(None, max_length=topics.MAX_FACTS)
    formats: Optional[List[str]] = None
    pinned_on: Optional[date] = None


class PinRequest(_Strict):
    pinned_on: Optional[date] = None


def _plans_api() -> Any:
    """``api/socials_plans.py``: it includes this router, so it is imported when a request runs."""
    from api import socials_plans

    return socials_plans


def _raise_for(exc: Exception) -> NoReturn:
    if isinstance(exc, topics.TopicNotFound):
        raise HTTPException(status_code=404, detail=TOPIC_NOT_FOUND)
    _plans_api()._raise_for(exc)


def _commit(db: Session, topic: Any) -> Dict[str, Any]:
    db.commit()
    db.refresh(topic)
    return topic.to_dict()


@router.get("/plans/{plan_id}/topics")
def list_social_plan_topics(plan_id: UUID, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    plan = _plans_api().load_plan(db, ctx, plan_id)
    bank = topics.list_topics(db, plan)
    notes = repeats.post_notes(db, plan, bank)
    rows = [{**topic.to_dict(), "repeat": notes.get(topic.id)} for topic in bank]
    return {
        "topics": rows, "total": len(rows), "unused": sum(1 for row in rows if not row["used_at"]),
        "research_note": socials_research_setup.research_note(db, plan.workspace_id),
        "research_last_run_at": (plan.research or {}).get("last_run_at"),
    }


@router.post("/plans/{plan_id}/topics", status_code=201, dependencies=[CAN_UPDATE])
def add_social_plan_topic(
    plan_id: UUID, body: TopicFields, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)
) -> Dict[str, Any]:
    """A person's topic in the bank, checked (the module docstring), with a ``warning`` when it is
    close to what the workspace has: they may repeat on purpose."""
    plans_api = _plans_api()
    plan = plans_api.load_plan(db, ctx, plan_id)
    warning = repeats.warning_for(db, plan, body.title)
    try:
        topic = topics.add_topic(db, plan, body.model_dump(exclude_unset=True), created_by=plans_api._posts_api()._actor(ctx))
    except service.SocialsError as exc:
        db.rollback()
        _raise_for(exc)
    return {**_commit(db, topic), "warning": warning}


@router.put("/plans/{plan_id}/topics/{topic_id}", dependencies=[CAN_UPDATE])
def update_social_plan_topic(
    plan_id: UUID, topic_id: UUID, body: TopicFields, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)
) -> Dict[str, Any]:
    plan = _plans_api().load_plan(db, ctx, plan_id)
    try:
        topic = topics.get_topic(db, plan, topic_id)
        fields = body.model_dump(exclude_unset=True)
        pinned = fields.pop("pinned_on", topic.pinned_on)
        topics.update_topic(db, plan, topic, fields)
        if pinned != topic.pinned_on:
            topics.pin(topic, pinned)
    except service.SocialsError as exc:
        db.rollback()
        _raise_for(exc)
    return _commit(db, topic)


@router.put("/plans/{plan_id}/topics/{topic_id}/pin", dependencies=[CAN_UPDATE])
def pin_social_plan_topic(
    plan_id: UUID, topic_id: UUID, body: PinRequest, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)
) -> Dict[str, Any]:
    """Pin the topic to a day (that day's slots take it first), or unpin it (null)."""
    plan = _plans_api().load_plan(db, ctx, plan_id)
    try:
        topic = topics.pin(topics.get_topic(db, plan, topic_id), body.pinned_on)
    except service.SocialsError as exc:
        _raise_for(exc)
    return _commit(db, topic)


@router.delete("/plans/{plan_id}/topics/{topic_id}", status_code=204, dependencies=[CAN_UPDATE])
def delete_social_plan_topic(
    plan_id: UUID, topic_id: UUID, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid)
) -> Response:
    plan = _plans_api().load_plan(db, ctx, plan_id)
    try:
        topic = topics.get_topic(db, plan, topic_id)
    except service.SocialsError as exc:
        _raise_for(exc)
    db.delete(topic)
    db.commit()
    return Response(status_code=204)
