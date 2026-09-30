"""PRD-251 S0.3a, US-301: the publish seam — the ONE way a post leaves Automatos.

A publish starts here, and nowhere else:

* :func:`begin_publish` (publish now, a scheduled slot) and :func:`begin_retry` (the
  failed targets of a failed or partially published post) run the approval guard
  FIRST (``service.assert_publishable`` / ``publish_lifecycle.assert_retryable``,
  D6): a post whose approval no longer matches its content publishes nothing and
  makes no Composio call, whatever the policy plane mode.
* Then the post is claimed: a compare-and-set on the status and the content hash the
  request loaded (``service.claim_unchanged``), approved | scheduled → publishing.
  Two workers, a double click or a job fired twice claim it once; the others get
  :class:`service.StaleContent` (409) and publish nothing.
* :func:`launch` runs the publish in the background (``launch_guarded``), and
  ``publishing.run_publish`` publishes each target through the workspace's own
  Composio connections.
"""
from __future__ import annotations

from typing import Any, Callable

from core.models.socials import SocialPost
from core.utils.background_tasks import launch_guarded
from modules.socials import publish_lifecycle, service
from modules.socials.publish_records import PublishJob
from modules.socials.publishing import run_publish


def _claim(db: Any, post: SocialPost, status: str, content_hash: str) -> None:
    """Commit the move to publishing only if the post's row still has the status and
    hash the request loaded; otherwise roll back and raise ``StaleContent``."""
    post_id, workspace_id = post.id, post.workspace_id
    if not service.claim_unchanged(db, post, status=status, content_hash=content_hash):
        db.rollback()
        current = service.get_post(db, workspace_id, post_id)
        if current is None:
            raise service.PostNotFound()
        raise service.StaleContent(service.compute_content_hash(current))
    db.commit()
    db.refresh(post)


def _begin(db: Any, post: SocialPost, actor: str, start: Callable[[SocialPost, str], SocialPost]) -> PublishJob:
    status, content_hash = post.status, post.content_hash
    start(post, actor)  # the approval guard runs first, before anything is written
    _claim(db, post, status, content_hash)
    return PublishJob(post.id, post.workspace_id, actor, content_hash, post.title or "")


def begin_publish(db: Any, post: SocialPost, actor: str) -> PublishJob:
    """Claim an approved or scheduled post for publishing: the job to run."""
    return _begin(db, post, actor, publish_lifecycle.start_publish)


def begin_retry(db: Any, post: SocialPost, actor: str) -> PublishJob:
    """Claim a failed or partially published post to publish its failed targets again."""
    return _begin(db, post, actor, publish_lifecycle.start_retry)


def launch(job: PublishJob) -> None:
    """Run the publish in the background; it writes its own end. Call it on the
    event loop (a threadpool route goes through ``anyio.from_thread``)."""
    launch_guarded(run_publish(job), subsystem="socials", operation="publish", workspace_id=job.workspace_id)
