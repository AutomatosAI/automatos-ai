"""PRD-251 D6, D8 (US-301): the publish lifecycle, as pure logic beside ``service.py``.

``service.TRANSITIONS`` holds the moves; this module is what each one checks and
records, so ``service.py`` (over 800 lines) only gains its table rows:

* ``start_publish``: approved or scheduled → publishing, only while the approval
  matches the content as it is NOW (``service.assert_publishable``, D6) and the post
  has channels to publish to.
* ``start_retry``: failed or partially_published → publishing, re-running the failed
  targets, under the same approval check (``assert_retryable``). A post that failed
  its render, or whose content changed since its approval, is not retried.
* ``miss``: scheduled → missed, a slot that passed with nothing published (D10).
* ``finish_publish``: publishing → published (every target published),
  partially_published (some) or failed (none), with each target's outcome in
  ``review_log``. A target the publish never tried (it ran out of time, or was
  lost) fails saying so, so Retry publishes it.

Targets move pending → uploading → published | failed (``social_post_targets``); the
publisher writes them (``modules/socials/publish_records.py``). No FastAPI and no
database here: the api maps the exceptions (NotPublishable → 409).
"""
from __future__ import annotations

from typing import Any, Dict, List

from core.models.socials import SocialPost
from modules.socials import service
from modules.socials.service import NotPublishable

TARGET_PENDING = "pending"
TARGET_UPLOADING = "uploading"
TARGET_PUBLISHED = "published"
TARGET_FAILED = "failed"
RETRYABLE_STATUSES = frozenset({service.FAILED, service.PARTIALLY_PUBLISHED})
NO_CHANNELS = "the post has no channels to publish to: choose them in the composer first"
NOTHING_TO_RETRY = "the post has no failed channel to retry"
NOT_TRIED = "The publish ended before this channel was tried: retry to publish it."


def approval_matches(post: Any) -> bool:
    """Whether the post's approval binds to its content as it is NOW (D6)."""
    approved_hash = getattr(post, "approved_hash", None)
    current = getattr(post, "content_hash", None)
    return bool(approved_hash) and approved_hash == current and approved_hash == service.compute_content_hash(post)


def _targets(post: Any) -> List[Any]:
    return list(getattr(post, "targets", None) or [])


def assert_retryable(post: Any) -> None:
    """Pass only a failed or partially published post that still holds its
    approval (D6) and has a target that failed to publish."""
    if post.status not in RETRYABLE_STATUSES:
        raise NotPublishable(f"a post that is {post.status.replace('_', ' ')} cannot be retried")
    if not approval_matches(post):
        raise NotPublishable("the content changed after it was approved: approve it again to publish")
    if not any(target.status == TARGET_FAILED for target in _targets(post)):
        raise NotPublishable(NOTHING_TO_RETRY)


def start_publish(post: SocialPost, actor: str) -> SocialPost:
    """approved or scheduled → publishing. The approval guard runs first."""
    service.assert_publishable(post)
    target = service._target(post, service.ACTION_PUBLISH)
    if not _targets(post):
        raise NotPublishable(NO_CHANNELS)
    post.status = target
    service._log(post, actor, service.ACTION_PUBLISH, None, channels=_channels(post))
    return post


def start_retry(post: SocialPost, actor: str) -> SocialPost:
    """failed or partially_published → publishing, for the failed targets."""
    assert_retryable(post)
    target = service._target(post, service.ACTION_RETRY)
    failed = [_name(t) for t in _targets(post) if t.status == TARGET_FAILED]
    post.status = target
    service._log(post, actor, service.ACTION_RETRY, None, channels=failed)
    return post


def miss(post: SocialPost, actor: str, reason: str) -> SocialPost:
    """scheduled → missed (D10): its slot passed and nothing was published. The
    approval stands, so it can be rescheduled or published now."""
    target = service._target(post, service.ACTION_MISSED)
    slot = post.scheduled_for.isoformat() if post.scheduled_for else None
    post.status = target
    service._log(post, actor, service.ACTION_MISSED, reason[: service.COMMENT_MAX_CHARS], scheduled_for=slot)
    return post


def _name(target: Any) -> str:
    return f"{target.toolkit} {target.post_kind}"


def _channels(post: Any) -> List[str]:
    return [_name(target) for target in sorted(_targets(post), key=lambda t: (t.toolkit, t.post_kind))]


def outcome_action(targets: List[Any]) -> str:
    """How a publish ends: every target published, some, or none."""
    published = [t for t in targets if t.status == TARGET_PUBLISHED]
    if targets and len(published) == len(targets):
        return service.ACTION_PUBLISHED
    return service.ACTION_PARTIALLY_PUBLISHED if published else service.ACTION_PUBLISH_FAILED


def _outcome(target: Any) -> Dict[str, Any]:
    return {
        "channel": _name(target),
        "status": target.status,
        "remote_id": target.remote_id,
        "permalink": target.permalink,
        "error": target.error,
    }


def _summary(action: str, targets: List[Any]) -> str:
    published = sum(1 for t in targets if t.status == TARGET_PUBLISHED)
    if action == service.ACTION_PUBLISHED:
        return f"Published to {published} of {len(targets)} channels."
    if action == service.ACTION_PARTIALLY_PUBLISHED:
        return f"Published to {published} of {len(targets)} channels; retry the others."
    return "Nothing was published: " + "; ".join(f"{_name(t)}: {t.error or 'failed'}" for t in targets if t.error)


def finish_publish(post: SocialPost, actor: str) -> str:
    """publishing → published, partially_published or failed, by its targets'
    statuses, each target's outcome in ``review_log``. The status it ended in. A
    target still pending was never tried: it fails, so a retry publishes it."""
    targets = sorted(_targets(post), key=lambda t: (t.toolkit, t.post_kind))
    for target in targets:
        if target.status == TARGET_PENDING:
            target.status, target.error = TARGET_FAILED, NOT_TRIED
    action = outcome_action(targets)
    post.status = service._target(post, action)
    summary = _summary(action, targets)[: service.COMMENT_MAX_CHARS]
    service._log(post, actor, action, summary, targets=[_outcome(t) for t in targets])
    return post.status
