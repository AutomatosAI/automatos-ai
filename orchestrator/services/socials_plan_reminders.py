"""PRD-251C Wave 2 (C3; US-C206): the evening-before reminder.

At ``make.remind_at`` (20:00 unless the plan says otherwise, in the plan's timezone) the plan
tick reminds whoever approves of the next day's posts still waiting for a person: a draft the
plan could not finish, a post waiting for approval, or one sent back for changes. One reminder
per plan per day, none when every post of the next day is approved, linked to the week's review
in the Queue (``plan_notify.notify_review``). A post that passes its slot unapproved follows the
plan's late policy as before.

It runs on the tick's own session factory, after the slots are made.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Callable

from core.models.socials import SocialPost
from modules.socials import plan_notify, plan_store, plans, service

logger = logging.getLogger(__name__)

REMINDER = ("social_plan_reminder", "Tomorrow's posts need you: ", "action_required")
WAITING = (service.DRAFT, service.NEEDS_APPROVAL, service.CHANGES_REQUESTED)


def _evening(plan: Any, now: datetime) -> bool:
    """Whether the plan's reminder time has come today, in its timezone."""
    local = now.astimezone(plans.zone_of(plan))
    return local.strftime("%H:%M") >= plans.make_settings(plan)["remind_at"]


def waiting_tomorrow(db: Any, plan: Any, now: datetime) -> int:
    """How many of the plan's posts planned for tomorrow (its timezone) still wait for a person."""
    zone = plans.zone_of(plan)
    tomorrow = now.astimezone(zone).date() + timedelta(days=1)
    start, end = plans.local_to_utc(tomorrow, "00:00", zone), plans.local_to_utc(tomorrow + timedelta(days=1), "00:00", zone)
    return (
        db.query(SocialPost.id)
        .filter(SocialPost.campaign_id == plan.id, SocialPost.status.in_(WAITING))
        .filter(SocialPost.planned_for >= start, SocialPost.planned_for < end)
        .count()
    )


def _remind(db: Any, plan: Any, now: datetime) -> bool:
    if not _evening(plan, now):
        return False
    count = waiting_tomorrow(db, plan, now)
    if count == 0 or not plan_notify.once_today(plan, REMINDER[0], now.astimezone(plans.zone_of(plan)).date()):
        return False
    db.commit()
    waits = "post for tomorrow still waits" if count == 1 else "posts for tomorrow still wait"
    plan_notify.notify_review(plan.workspace_id, plan.id, REMINDER, f"{plan.name}: {count} {waits} for approval")
    return True


def remind_due(session_factory: Callable[[], Any], now: datetime) -> int:
    """The tick's reminder pass: each active plan whose evening has come. How many were reminded."""
    db = session_factory()
    reminded = 0
    try:
        for plan in plan_store.active_plans(db):
            try:
                reminded += int(_remind(db, plan, now))
            except Exception:  # noqa: BLE001 — one plan never stops the pass; logged
                logger.exception("[Socials] plan %s: the evening reminder could not be sent", plan.id)
                db.rollback()
    finally:
        db.close()
    return reminded
