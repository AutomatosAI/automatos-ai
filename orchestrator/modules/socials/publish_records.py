"""PRD-251 D8 (US-301): what a publish writes, each in its own short transaction.

The publish runs in the background (``publishing.py``) and talks to the platforms
for minutes; no database session or row lock is held across those calls. Each write
opens a session, commits and closes:

* :func:`load_work`: the post is still ``publishing`` the version the publish was
  started for, and its approval still matches (D6); the targets not yet published,
  each with its steps and what its sources read.
* :func:`begin_attempt`: a compare-and-set on the target, pending or failed →
  uploading, counting the attempt. A published target is never run again, and two
  runs never run one target at once.
* :func:`fail_attempt` / :func:`record_published`: the attempt's end: the
  platform's message, or the receipt (remote id, permalink, when).
* :func:`finish`: a target left uploading (a timeout, an unexpected error) fails
  with the reason, and the post ends by its targets (``publish_lifecycle``), a
  compare-and-set on ``publishing`` and the hash the publish started from.
* :func:`end_lost`: a post whose run is gone (the process restarted, or the run
  was lost) ends the same way, its uploading targets failing with
  :data:`LOST_UPLOADING`: the boot reaper and the leader's reconcile tick
  (``schedule_jobs.end_lost_publishes``) call it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Collection, List, Optional, Sequence, Tuple
from uuid import UUID

from sqlalchemy import update

from core.models.socials import SocialPostTarget
from modules.socials import media_urls, publish_lifecycle, service
from modules.socials.capabilities import ChannelStep
from modules.socials.publish_lifecycle import TARGET_FAILED, TARGET_PENDING, TARGET_PUBLISHED, TARGET_UPLOADING
from modules.socials.publish_sources import TargetContext, context_for, steps_of

logger = logging.getLogger(__name__)

NOTES = "notes"  # a target's action_plan key: what its receipt says beside the id and link
REMOTE_ID_MAX_CHARS = 255
PERMALINK_MAX_CHARS = 1000
LOST_UPLOADING = (
    "The publish was lost (the server restarted, or the run stopped) while this channel was uploading. "
    "The platform may have taken it: check the channel before you retry."
)


@dataclass(frozen=True)
class PublishJob:
    """One publish (or retry) of one post, as the background task runs it."""

    post_id: UUID
    workspace_id: UUID
    actor: str
    content_hash: str
    title: str = ""


@dataclass(frozen=True)
class TargetWork:
    target_id: UUID
    steps: Tuple[ChannelStep, ...]
    context: TargetContext


def _post(db: Any, job: PublishJob) -> Any:
    """The post, when it is still publishing the version the job started for."""
    post = service.get_post(db, job.workspace_id, job.post_id)
    if post is None or post.status != service.PUBLISHING or post.content_hash != job.content_hash:
        return None
    return post


def load_work(factory: Callable[[], Any], job: PublishJob) -> Optional[List[TargetWork]]:
    """The targets to publish, or ``None`` when the post moved on or its approval no
    longer matches its content (then nothing is published)."""
    db = factory()
    try:
        post = _post(db, job)
        if post is None:
            logger.warning("[Socials] post %s is no longer publishing; nothing is published", job.post_id)
            return None
        if not publish_lifecycle.approval_matches(post):
            # Nothing is published, and the run ends the post (finish) rather than
            # leaving it publishing.
            logger.warning("[Socials] post %s is not publishing its approved version; nothing is published", job.post_id)
            return []
        files = media_urls.resolve_post_media(db, post)
        pending = sorted((t for t in post.targets if t.status != TARGET_PUBLISHED), key=lambda t: (t.toolkit, t.post_kind))
        return [TargetWork(t.id, steps_of(t.action_plan), context_for(post, t, files)) for t in pending]
    finally:
        db.close()


def begin_attempt(factory: Callable[[], Any], target_id: UUID) -> bool:
    """pending or failed → uploading, one more attempt: ``False`` when the target is
    published or another run holds it."""
    table = SocialPostTarget.__table__
    db = factory()
    try:
        result = db.execute(
            update(table)
            .where(table.c.id == target_id, table.c.status.in_([TARGET_PENDING, TARGET_FAILED]))
            .values(status=TARGET_UPLOADING, attempts=table.c.attempts + 1, error=None)
        )
        db.commit()
        return result.rowcount == 1
    finally:
        db.close()


def _write(factory: Callable[[], Any], target_id: UUID, notes: Sequence[str], **values: Any) -> None:
    """Write the end of an attempt onto a target this run holds (uploading)."""
    db = factory()
    try:
        target = db.get(SocialPostTarget, target_id)
        if target is None or target.status != TARGET_UPLOADING:
            db.rollback()
            logger.warning("[Socials] target %s is no longer uploading; its attempt is not recorded", target_id)
            return
        for name, value in values.items():
            setattr(target, name, value)
        plan = dict(target.action_plan or {})
        plan[NOTES] = list(notes)
        target.action_plan = plan
        db.commit()
    finally:
        db.close()


def fail_attempt(factory: Callable[[], Any], target_id: UUID, message: str, notes: Sequence[str]) -> None:
    _write(factory, target_id, notes, status=TARGET_FAILED, error=message)


def record_published(
    factory: Callable[[], Any], target_id: UUID, remote_id: Optional[str], permalink: Optional[str], notes: Sequence[str],
) -> None:
    _write(
        factory,
        target_id,
        notes,
        status=TARGET_PUBLISHED,
        error=None,
        remote_id=remote_id[:REMOTE_ID_MAX_CHARS] if remote_id else None,
        permalink=permalink if permalink and len(permalink) <= PERMALINK_MAX_CHARS else None,
        published_at=datetime.now(timezone.utc),
    )


def fail_unfinished(post: Any, reason: str, held: Optional[Collection[UUID]] = None) -> None:
    """A target the publish left uploading fails with ``reason``: those in ``held``
    (the targets this run claimed), or every one (the boot reaper: no run is left)."""
    for target in post.targets:
        if target.status == TARGET_UPLOADING and (held is None or target.id in held):
            target.status = TARGET_FAILED
            target.error = reason


def end_lost(post: Any, actor: str) -> str:
    """End a post whose publish run is gone: a target it left uploading fails with
    :data:`LOST_UPLOADING`, a published one keeps its receipt, and the post ends by
    its targets. The status it ended in; the caller commits."""
    fail_unfinished(post, LOST_UPLOADING)
    return publish_lifecycle.finish_publish(post, actor)


def finish(factory: Callable[[], Any], job: PublishJob, unfinished: str, held: Collection[UUID]) -> Optional[str]:
    """End the publish on the post: the status it ended in, or ``None`` when the post
    moved on, or another run still holds one of its targets (nothing is written then:
    that run ends it). A target this run claimed and left uploading fails with
    ``unfinished``."""
    db = factory()
    try:
        post = _post(db, job)
        if post is None:
            db.rollback()
            logger.warning("[Socials] post %s moved on while it published; its end is not recorded", job.post_id)
            return None
        fail_unfinished(post, unfinished, held)
        if any(target.status == TARGET_UPLOADING for target in post.targets):
            db.rollback()
            logger.warning("[Socials] post %s: another run is still publishing a target; it ends the post", job.post_id)
            return None
        ended = publish_lifecycle.finish_publish(post, job.actor)
        if not service.claim_unchanged(db, post, status=service.PUBLISHING, content_hash=job.content_hash):
            db.rollback()
            return None
        db.commit()
        return ended
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
