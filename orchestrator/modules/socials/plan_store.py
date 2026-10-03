"""PRD-251B Wave 2 (US-B202, US-B208): plans in the database.

Create, update, pause, resume and end a plan (a ``social_campaigns`` row of kind
``plan``); read its slots in a window with what was made for each; move or skip a
planned slot (``slot_overrides``). Every read is scoped to the workspace. Nothing here
commits: the caller (the api, the make tick) owns the transaction.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence
from uuid import UUID

from sqlalchemy import func

from core.models.socials import SocialCampaign, SocialPost, SocialTopic
from modules.socials import campaigns, plans
from modules.socials.template_gallery import gallery

# What a status change may come from (B6): an ended plan stays ended.
STATUS_FROM = {
    plans.PAUSED: (plans.ACTIVE,),
    plans.ACTIVE: (plans.PAUSED,),
    plans.ENDED: (plans.ACTIVE, plans.PAUSED),
}


def templates_of(db: Any, workspace_id: UUID) -> Dict[str, plans.TemplateInfo]:
    """The workspace's social templates a cadence row may name: format and declared lengths."""
    return {entry["id"]: (entry["format"], list(entry["durations"])) for entry in gallery(db, workspace_id)}


def get_plan(db: Any, workspace_id: UUID, plan_id: UUID) -> Optional[SocialCampaign]:
    return (
        db.query(SocialCampaign)
        .filter(SocialCampaign.id == plan_id, SocialCampaign.workspace_id == workspace_id, SocialCampaign.kind == plans.PLAN)
        .first()
    )


def require_plan(db: Any, workspace_id: UUID, plan_id: UUID) -> SocialCampaign:
    plan = get_plan(db, workspace_id, plan_id)
    if plan is None:
        raise plans.PlanNotFound(str(plan_id))
    return plan


def list_plans(db: Any, workspace_id: UUID) -> List[SocialCampaign]:
    return (
        db.query(SocialCampaign)
        .filter(SocialCampaign.workspace_id == workspace_id, SocialCampaign.kind == plans.PLAN)
        .order_by(SocialCampaign.created_at.desc())
        .all()
    )


def active_plans(db: Any) -> List[SocialCampaign]:
    """Every active plan of every workspace: the make tick's list."""
    return db.query(SocialCampaign).filter(SocialCampaign.kind == plans.PLAN, SocialCampaign.status == plans.ACTIVE).all()


def _apply(plan: SocialCampaign, fields: Mapping[str, Any], templates: Mapping[str, plans.TemplateInfo]) -> None:
    clean = plans.validate_fields(fields, templates)
    starts = fields.get("starts_on", plan.starts_on)
    ends = fields.get("ends_on", plan.ends_on)
    if "starts_on" in fields or "ends_on" in fields:
        clean["starts_on"], clean["ends_on"] = plans.validate_dates(starts, ends)
    if "name" in fields:
        clean["name"] = campaigns.validate_name(fields["name"])
    if "approval_mode" in fields:
        clean["approval_mode"] = campaigns.validate_mode(fields["approval_mode"])
    for key, value in clean.items():
        setattr(plan, key, value)


def create_plan(db: Any, *, workspace_id: UUID, created_by: str, fields: Mapping[str, Any]) -> SocialCampaign:
    """A new active plan from ``fields`` (name, dates, timezone and cadence required); not committed."""
    missing = [key for key in ("name", "starts_on", "ends_on", "timezone", "cadence") if fields.get(key) in (None, "", [])]
    if missing:
        raise plans.InvalidPlan(f"a plan needs {', '.join(missing)}")
    plan = SocialCampaign(
        workspace_id=workspace_id, created_by=created_by, name="", kind=plans.PLAN, status=plans.ACTIVE,
        approval_mode=campaigns.PER_POST, approved_hash_set=[],
        sources=plans.validate_sources(None), make=plans.validate_make(None), research=plans.validate_research(None),
        late_policy=plans.SKIP, slot_overrides={},
    )
    _apply(plan, fields, templates_of(db, workspace_id))
    db.add(plan)
    db.flush()
    return plan


def update_plan(db: Any, plan: SocialCampaign, fields: Mapping[str, Any]) -> SocialCampaign:
    """``plan`` with ``fields`` changed, each checked; an ended plan is read-only."""
    if plan.status == plans.ENDED:
        raise plans.InvalidPlan("an ended plan cannot change")
    _apply(plan, fields, templates_of(db, plan.workspace_id))
    return plan


def set_status(plan: SocialCampaign, status: str) -> SocialCampaign:
    """Pause, resume or end (B6). A move the plan cannot make is refused (422)."""
    if plan.status not in STATUS_FROM.get(status, ()):
        raise plans.InvalidPlan(f"a {plan.status} plan cannot become {status}")
    plan.status = status
    return plan


def made_posts(db: Any, plan: SocialCampaign, keys: Optional[Sequence[str]] = None) -> Dict[str, SocialPost]:
    """The plan's posts made for a slot, by slot key (``keys`` narrows the read)."""
    query = db.query(SocialPost).filter(SocialPost.campaign_id == plan.id, SocialPost.slot_key.isnot(None))
    if keys is not None:
        if not keys:
            return {}
        query = query.filter(SocialPost.slot_key.in_(list(keys)))
    return {post.slot_key: post for post in query.all()}


def taken_keys(db: Any, plan: SocialCampaign) -> set:
    return {key for (key,) in db.query(SocialPost.slot_key).filter(SocialPost.campaign_id == plan.id, SocialPost.slot_key.isnot(None))}


def pinned_topics(db: Any, plan: SocialCampaign) -> Dict[str, SocialTopic]:
    """The plan's unused topics pinned to a day, by that day (ISO)."""
    rows = (
        db.query(SocialTopic)
        .filter(SocialTopic.campaign_id == plan.id, SocialTopic.pinned_on.isnot(None), SocialTopic.used_at.is_(None))
        .order_by(SocialTopic.created_at)
        .all()
    )
    pinned: Dict[str, SocialTopic] = {}
    for topic in rows:
        pinned.setdefault(topic.pinned_on.isoformat(), topic)
    return pinned


def _post_summary(post: SocialPost) -> Dict[str, Any]:
    moment = post.scheduled_for or post.planned_for
    return {"id": str(post.id), "title": post.title, "status": post.status, "at": moment.isoformat() if moment else None}


def plan_slots(db: Any, plan: SocialCampaign, start: datetime, end: datetime) -> List[Dict[str, Any]]:
    """The plan's slots in [start, end): each ``planned`` (no post yet, with the topic
    pinned to its day, if any) or ``made`` (with its post)."""
    slots = plans.expand_slots(plan, start, end)
    made = made_posts(db, plan, [slot.key for slot in slots])
    pinned = pinned_topics(db, plan)
    rows = []
    for slot in slots:
        post = made.get(slot.key)
        topic = pinned.get(slot.local_date.isoformat())
        rows.append({
            **slot.to_dict(),
            "state": "made" if post is not None else "planned",
            "post": _post_summary(post) if post is not None else None,
            "topic": {"id": str(topic.id), "title": topic.title} if topic is not None and post is None else None,
        })
    return rows


def move_slot(plan: SocialCampaign, key: str, *, to: Optional[datetime], skip: bool) -> Dict[str, Any]:
    """Move a planned slot to ``to`` or skip it (B208): ``slot_overrides`` holds it; the
    cadence is untouched. A move stays within MAX_MOVE_DAYS of the slot's own time."""
    slot = plans.slot_for_key(plan, key)
    if slot is None and not (plan.slot_overrides or {}).get(key, {}).get("skip"):
        raise plans.InvalidPlan("no planned slot has that key")
    overrides = dict(plan.slot_overrides or {})
    if skip:
        overrides[key] = {"skip": True}
    elif to is None:
        overrides.pop(key, None)
    else:
        if to.tzinfo is None:
            raise plans.InvalidPlan("to must carry a timezone")
        original = plans.local_to_utc(*_key_day_clock(key), plans.zone_of(plan))
        if abs(to.astimezone(timezone.utc) - original).days > plans.MAX_MOVE_DAYS:
            raise plans.InvalidPlan(f"a slot moves at most {plans.MAX_MOVE_DAYS} days")
        overrides[key] = {"to": to.astimezone(timezone.utc).isoformat()}
    plan.slot_overrides = overrides
    moved = plans.slot_for_key(plan, key)
    return {"key": key, "skipped": moved is None, "slot": moved.to_dict() if moved else None}


def _key_day_clock(key: str):
    _row, day, clock = plans.parse_slot_key(key)
    return day, clock


def bank_counts(db: Any, plan_ids: Sequence[UUID]) -> Dict[UUID, Dict[str, int]]:
    """Per plan: its topics and how many are unused."""
    if not plan_ids:
        return {}
    rows = (
        db.query(SocialTopic.campaign_id, func.count(SocialTopic.id), func.count(SocialTopic.used_at))
        .filter(SocialTopic.campaign_id.in_(list(plan_ids)))
        .group_by(SocialTopic.campaign_id)
        .all()
    )
    return {plan_id: {"topics": total, "unused": total - used} for plan_id, total, used in rows}
