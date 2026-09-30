"""PRD-251 D10 (US-306): a scheduled post publishes at its slot, once, or is missed.

Each scheduled post has one APScheduler ``DateTrigger`` job, ``social-publish-<post_id>``,
on the unified scheduler (``services/scheduler.py``), which only the worker holding
the fcntl lock runs. The job is :func:`fire_scheduled_post`, a module-level function
with plain string args (the post and its workspace), never a closure: the
``RedisJobStore`` pickles it.

* **Keeping jobs and posts in step.** A request usually lands on a worker without the
  scheduler. After a committed write, :func:`sync_job` registers, moves or removes
  the post's job when this worker hosts the scheduler; either way the leader's
  reconcile pass (:func:`reconcile`, from ``services/schedule_reconcile.py`` every
  ``RECONCILE_INTERVAL_SECONDS``) registers a job for every scheduled post, moves one
  whose slot changed, and removes one whose post is no longer scheduled
  (unscheduled, edited, rejected, published).
* **The fire.** The job claims the post like publish now (``publisher.begin_publish``:
  the approval guard first, then the compare-and-set), so a job fired twice, or a
  fire racing a person's publish now, publishes once; then the US-301 publisher
  runs. A job registered late fires at once (``misfire_grace_time=None``) and decides.
* **A missed slot.** Fired more than ``SOCIALS_MISFIRE_GRACE_SECONDS`` after its slot
  (the backend was down), with Socials off for the workspace, or with an approval
  that no longer matches, the post goes to ``missed`` and the workspace is told
  (``social_post_missed``). Stale content is never posted silently. A missed post can
  be rescheduled or published now while its approval still matches.
* **A lost publish.** The same pass ends a post still publishing well past a run's
  own limit (``SOCIALS_PUBLISH_RUN_MAX_SECONDS`` plus :data:`LOST_MARGIN_SECONDS`),
  whose run died with its process or never ended it: no run can still hold it
  (:func:`end_lost_publishes`, ``publish_records.end_lost``). The boot reaper does
  the same for the posts it finds at boot.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy import update

from config import config
from core.models.socials import SocialPost
from core.models.workspaces import Workspace
from modules.socials import notify, publish_lifecycle, publisher, service
from modules.socials.publish_records import PublishJob, end_lost
from modules.socials.publishing import run_publish
from modules.socials.settings import socials_off_reason

logger = logging.getLogger(__name__)

JOB_PREFIX = "social-publish-"
SCHEDULER_ACTOR = "scheduler"
# A fire this early is a slot moved later since the job was registered: its job moves.
EARLY_TOLERANCE_SECONDS = 5
MISSED_LATE = (
    "Its slot passed {late} minutes before it could publish (the grace is {grace} minutes): "
    "nothing was published. Reschedule it or publish it now."
)
MISSED_OFF = "Socials was off for this workspace at its slot ({why}): nothing was published."
# Past a run's own limit, before a publishing post counts as lost: its end is written.
LOST_MARGIN_SECONDS = 300
MISSED_STALE = "Its approval no longer matched its content at its slot ({why}): nothing was published."


def job_id(post_id: Any) -> str:
    return f"{JOB_PREFIX}{post_id}"


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


# ---- registering --------------------------------------------------------------


def register(scheduler: Any, post_id: Any, workspace_id: Any, slot: datetime) -> None:
    """The post's job at ``slot``, replacing any it had."""
    from apscheduler.triggers.date import DateTrigger

    scheduler.add_job(
        fire_scheduled_post,
        DateTrigger(run_date=_utc(slot)),
        args=[str(post_id), str(workspace_id)],
        id=job_id(post_id),
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=None,  # a late job still fires: it decides publish or missed
    )


def unregister(scheduler: Any, post_id: Any) -> bool:
    """Remove the post's job; ``False`` when it had none."""
    from apscheduler.jobstores.base import JobLookupError

    try:
        scheduler.remove_job(job_id(post_id))
    except JobLookupError:
        return False
    return True


def _leader() -> Any:
    """This worker's running scheduler, or ``None`` (another worker holds the lock)."""
    from services.scheduler import get_unified_scheduler

    scheduler = get_unified_scheduler().apscheduler
    return scheduler if scheduler is not None and getattr(scheduler, "running", False) else None


def sync_job(post: Any) -> None:
    """After a committed write: the post's job as its status and slot say, when this
    worker hosts the scheduler (else the reconcile pass does it). Never raises."""
    try:
        scheduler = _leader()
        if scheduler is None:
            return
        if post.status == service.SCHEDULED and post.scheduled_for is not None:
            register(scheduler, post.id, post.workspace_id, post.scheduled_for)
        else:
            unregister(scheduler, post.id)
    except Exception:  # noqa: BLE001 — the reconcile pass catches up; the write stands
        logger.exception("[Socials] the job of post %s was not synced; the reconcile pass will", getattr(post, "id", None))


def _run_date(job: Any) -> Optional[datetime]:
    run_date = getattr(getattr(job, "trigger", None), "run_date", None)
    return _utc(run_date) if isinstance(run_date, datetime) else None


def reconcile(scheduler: Any, db: Any) -> Dict[str, Any]:
    """The leader's pass: a job for every scheduled post at its slot, and none for a
    post that is not scheduled. Idempotent."""
    if scheduler is None or not getattr(scheduler, "running", False):
        return {"added": 0, "moved": 0, "removed": 0, "skipped": True}
    rows = (
        db.query(SocialPost.id, SocialPost.workspace_id, SocialPost.scheduled_for)
        .filter(SocialPost.status == service.SCHEDULED, SocialPost.scheduled_for.isnot(None))
        .all()
    )
    wanted = {job_id(row.id): row for row in rows}
    jobs = {str(job.id): job for job in scheduler.get_jobs() if str(job.id).startswith(JOB_PREFIX)}
    added = moved = removed = 0
    for key, row in wanted.items():
        job = jobs.get(key)
        if job is None or _run_date(job) != _utc(row.scheduled_for):
            register(scheduler, row.id, row.workspace_id, row.scheduled_for)
            added, moved = (added + 1, moved) if job is None else (added, moved + 1)
    for key in set(jobs) - set(wanted):
        scheduler.remove_job(key)
        removed += 1
    ended = end_lost_publishes(db, datetime.now(timezone.utc))
    return {"added": added, "moved": moved, "removed": removed, "ended": ended}


def end_lost_publishes(db: Any, now: datetime) -> int:
    """End each post publishing since before a run's limit plus the margin: its run
    is gone. A compare-and-set per post, so a run that ends it first wins. How many
    it ended."""
    cutoff = now - timedelta(seconds=config.SOCIALS_PUBLISH_RUN_MAX_SECONDS + LOST_MARGIN_SECONDS)
    ids = [row.id for row in db.query(SocialPost.id).filter(
        SocialPost.status == service.PUBLISHING, SocialPost.updated_at < cutoff,
    ).all()]
    ended = 0
    for post_id in ids:
        post = db.get(SocialPost, post_id)
        if post is None or post.status != service.PUBLISHING:
            continue
        content_hash = post.content_hash
        status = end_lost(post, SCHEDULER_ACTOR)
        if not service.claim_unchanged(db, post, status=service.PUBLISHING, content_hash=content_hash):
            db.rollback()
            continue
        db.commit()
        ended += 1
        logger.warning("[Socials] post %s was still publishing with no run left: it ended %s", post_id, status)
    return ended


# ---- the fire -------------------------------------------------------------------


def _missed_reason(db: Any, post: SocialPost, slot: datetime, now: datetime) -> Optional[str]:
    grace = config.SOCIALS_MISFIRE_GRACE_SECONDS
    if now - slot > timedelta(seconds=grace):
        late = int((now - slot).total_seconds() // 60)
        return MISSED_LATE.format(late=late, grace=grace // 60)
    off = socials_off_reason(db.get(Workspace, post.workspace_id))
    return MISSED_OFF.format(why=off) if off else None


def _miss(db: Any, post: SocialPost, reason: str) -> bool:
    status, content_hash = post.status, post.content_hash
    publish_lifecycle.miss(post, SCHEDULER_ACTOR, reason)
    if not service.claim_unchanged(db, post, status=status, content_hash=content_hash):
        db.rollback()
        return False
    db.commit()
    logger.warning("[Socials] post %s missed its slot: %s", post.id, reason)
    return True


def _hold_slot(db: Any, post: SocialPost) -> bool:
    """Lock the post's row, but only while it is still scheduled for the slot this
    fire read: a reschedule that committed first wins (its job fires at the new slot),
    and one that comes after waits for this fire's claim, then finds the post moved on."""
    table = SocialPost.__table__
    with db.no_autoflush:
        result = db.execute(
            update(table)
            .where(
                table.c.id == post.id,
                table.c.status == service.SCHEDULED,
                table.c.scheduled_for == post.scheduled_for,
            )
            .values(status=table.c.status)
        )
    return result.rowcount == 1


def _claim(db: Any, post: SocialPost, now: datetime) -> Tuple[Optional[PublishJob], bool]:
    slot = _utc(post.scheduled_for)
    if slot > now + timedelta(seconds=EARLY_TOLERANCE_SECONDS):
        return None, False  # moved later since this job was registered
    if not _hold_slot(db, post):
        db.rollback()
        return None, False  # rescheduled (or unscheduled) since this fire read it
    reason = _missed_reason(db, post, slot, now)
    if reason is None and not publish_lifecycle.approval_matches(post):
        reason = MISSED_STALE.format(why="the content changed after it was approved")
    if reason is not None:
        return None, _miss(db, post, reason)
    try:
        return publisher.begin_publish(db, post, SCHEDULER_ACTOR), False
    except service.SocialsError as exc:  # another worker claimed it first
        db.rollback()
        logger.info("[Socials] the scheduled publish of post %s was not claimed: %s", post.id, exc)
        return None, False


def begin_scheduled(
    factory: Callable[[], Any], post_id: UUID, workspace_id: UUID, now: datetime,
) -> Tuple[Optional[PublishJob], Optional[str]]:
    """The fire's decision: the publish to run, or ``(None, title)`` when the post was
    missed (the workspace is told), or ``(None, None)`` when there is nothing to do."""
    db = factory()
    try:
        post = service.get_post(db, workspace_id, post_id)
        if post is None or post.status != service.SCHEDULED or post.scheduled_for is None:
            return None, None  # unscheduled, edited, published or deleted since
        job, missed = _claim(db, post, now)
        return job, (post.title or "" if missed else None)
    finally:
        db.close()


def _default_session_factory() -> Callable[[], Any]:
    from core.database.database import SessionLocal

    return SessionLocal


async def fire_scheduled_post(
    post_id: str,
    workspace_id: str,
    *,
    session_factory: Optional[Callable[[], Any]] = None,
    clock: Optional[Callable[[], datetime]] = None,
) -> Optional[str]:
    """The ``social-publish-<post_id>`` job: publish the post if its slot is due and
    its approval stands, else mark it missed. The status the post ended in, or
    ``None`` when there was nothing to publish."""
    factory = session_factory or _default_session_factory()
    post, workspace = UUID(post_id), UUID(workspace_id)
    now = clock() if clock else datetime.now(timezone.utc)
    job, missed_title = await asyncio.to_thread(begin_scheduled, factory, post, workspace, now)
    if missed_title is not None:
        await notify.dispatch_publish_outcome(workspace, post, missed_title, service.MISSED, session_factory=factory)
        return service.MISSED
    if job is None:
        return None
    return await run_publish(job, session_factory=factory)
