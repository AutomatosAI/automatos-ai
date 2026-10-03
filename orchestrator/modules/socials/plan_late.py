"""PRD-251B Wave 2 (B11; US-B206): a plan's slot that passes.

What happens is the plan's ``late_policy``:

* ``skip`` (and any post that no plan made): an unapproved post ends ``missed``, once,
  and the workspace is told (``schedule_jobs.pass_planned_slots``); one approved after
  its slot passed stays approved, for a person to schedule.
* ``next_slot``: the post moves to the next free slot of the same cadence row, and the
  workspace is told where. Unapproved, it keeps waiting for approval there; approved
  late, it is scheduled into it. With no free slot left in the plan, it is a ``skip``.

The leader's reconcile pass calls both: :func:`moved_on` for each unapproved post
whose slot passed, :func:`schedule_late_approvals` once per pass. Each write is a
compare-and-set, so two passes racing on one post leave one move.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple

from config import config
from core.models.socials import SocialCampaign, SocialPost
from modules.socials import plan_notify, plan_store, plans, service

logger = logging.getLogger(__name__)

LATE_ACTOR = "socials-plan"
ACTION_NEXT_SLOT = "next_slot"
MOVED_NOTE = "Its slot passed before approval; the plan moves a late post to the next free slot: {slot}."
SCHEDULED_NOTE = "Approved after its slot passed; the plan schedules a late post into the next free slot: {slot}."
# A made post moves to a slot at least this far ahead: it is made already, so no make lead.
LATE_LEAD = timedelta(minutes=15)


def next_slot_of(db: Any, post: SocialPost, now: datetime) -> Optional[Tuple[SocialCampaign, plans.Slot]]:
    """The plan and the next free slot of the post's row, when its plan moves late posts on."""
    if post.campaign_id is None or not post.slot_key:
        return None
    plan = db.get(SocialCampaign, post.campaign_id)
    if plan is None or plan.kind != plans.PLAN or plan.late_policy != plans.NEXT_SLOT or plan.status == plans.ENDED:
        return None
    parsed = plans.parse_slot_key(post.slot_key)
    if parsed is None:
        return None
    slot = plans.next_free_slot(plan, parsed[0], now + LATE_LEAD, plan_store.taken_keys(db, plan))
    return (plan, slot) if slot is not None else None


def _moved(post: SocialPost, plan: SocialCampaign, slot: plans.Slot, note: str) -> None:
    """The post on ``slot``: its key and planned time, with the move in its review log."""
    post.slot_key = slot.key
    service.set_planned_for(post, slot.at, plan.timezone)
    entry = {
        "at": datetime.now(timezone.utc).isoformat(), "by": LATE_ACTOR, "action": ACTION_NEXT_SLOT,
        "comment": note.format(slot=slot.at.isoformat(timespec="minutes")), "planned_for": slot.at.isoformat(),
    }
    post.review_log = [*(post.review_log or []), entry]


def moved_on(db: Any, post: SocialPost, now: datetime) -> bool:
    """An unapproved post whose slot passed: moved to its row's next free slot when its
    plan says ``next_slot`` (committed, the workspace told). ``True`` when the pass should
    leave the post alone now: moved, or another writer got there first."""
    found = next_slot_of(db, post, now)
    if found is None:
        return False
    plan, slot = found
    status, content_hash = post.status, post.content_hash
    _moved(post, plan, slot, MOVED_NOTE)
    if not service.claim_unchanged(db, post, status=status, content_hash=content_hash):
        db.rollback()
        return True
    db.commit()
    logger.info("[Socials] post %s moved to the next slot of its plan: %s", post.id, slot.key)
    plan_notify.notify_slot_moved(post.workspace_id, post.id, post.title or "", slot.at)
    return True


def schedule_late_approvals(db: Any, now: datetime) -> int:
    """Each post approved after its plan's slot passed, scheduled into the row's next free
    slot when the plan says ``next_slot``. How many it scheduled."""
    cutoff = now - timedelta(seconds=config.SOCIALS_MISFIRE_GRACE_SECONDS)
    ids = [row.id for row in db.query(SocialPost.id).filter(
        SocialPost.status == service.APPROVED, SocialPost.slot_key.isnot(None),
        SocialPost.planned_for.isnot(None), SocialPost.planned_for < cutoff,
    ).all()]
    scheduled = 0
    for post_id in ids:
        post = db.get(SocialPost, post_id)
        found = next_slot_of(db, post, now) if post is not None and post.status == service.APPROVED else None
        if found is None:
            continue
        plan, slot = found
        content_hash = post.content_hash
        try:
            _moved(post, plan, slot, SCHEDULED_NOTE)
            service.schedule(post, LATE_ACTOR, slot.at, plan.timezone or "UTC")
        except service.SocialsError:
            logger.exception("[Socials] post %s could not be scheduled into its plan's next slot", post_id)
            db.rollback()
            continue
        if not service.claim_unchanged(db, post, status=service.APPROVED, content_hash=content_hash):
            db.rollback()
            continue
        db.commit()
        scheduled += 1
        plan_notify.notify_slot_moved(post.workspace_id, post.id, post.title or "", slot.at)
    return scheduled
