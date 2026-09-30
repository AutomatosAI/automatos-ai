"""PRD-251 Wave 3 (review, P251W3) — a publish that is lost, cannot start, or races.

On the harness in ``tests/test_prd251w3_publisher.py`` (Composio mocked):

* the boot reaper ends a post its process lost mid-publish, keeping its receipts;
  the leader's reconcile tick ends one lost inside the boot cutoff, once no run can
  still hold it (``schedule_jobs.end_lost_publishes``);
* a run that cannot load its work still ends the post, each untried target failing
  as "not tried", so Retry publishes it;
* the default executor holds no session across the run: one per Composio call;
* an attempt ending while its post is ended as lost does not overwrite that end
  (the target rows are locked, proven on the CI Postgres);
* with channels, the title is approved content: YouTube publishes it.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

import tests.test_prd251w3_publisher as harness  # noqa: E402
from config import config  # noqa: E402
from core.composio import tool_executor  # noqa: E402
from core.models.socials import SocialPost  # noqa: E402
from modules.socials import publish_lifecycle, publish_records, publisher, service  # noqa: E402
from modules.socials.publishing import run_publish  # noqa: E402
from tests.test_prd251w3_publisher import (  # noqa: E402
    AUTHOR,
    LINKEDIN,
    REVIEWER,
    SHARE_URN,
    VIDEO,
    WS,
    FakeExecutor,
    _approved_post,
    _claim,
    _post,
    _publish,
    _runtime,
    _target,
    _targets,
)

env = harness.env
pg_engine = harness.pg_engine


def test_the_boot_reaper_ends_a_lost_publish_and_keeps_its_receipts(env):
    from core.boot import reaper

    env.media = [VIDEO]
    post_id = _approved_post(env, _target("linkedin", "text"), _target("linkedin", "video"))
    _claim(env, post_id)
    with env.factory() as db:
        post = service.get_post(db, WS, post_id)
        rows = {(t.toolkit, t.post_kind): t for t in post.targets}
        rows["linkedin", "text"].status, rows["linkedin", "text"].remote_id = "published", SHARE_URN
        rows["linkedin", "video"].status = "uploading"
        db.commit()
    with env.factory() as db:
        later = datetime.now(timezone.utc) + timedelta(hours=2)
        cutoff = later - timedelta(minutes=config.BOOT_REAPER_STALE_MINUTES)
        assert reaper._reap_social_publishes(db, cutoff, later) == 1
        db.commit()
    post = _post(env, post_id)
    targets = _targets(env, post_id)
    assert post.status == "partially_published" and post.review_log[-1]["action"] == "partially_published"
    assert targets["linkedin", "text"]["remote_id"] == SHARE_URN
    assert targets["linkedin", "video"]["status"] == "failed" and "may have taken it" in targets["linkedin", "video"]["error"]



def _publishing_since(env, post_id, since):
    with env.factory() as db:
        db.get(SocialPost, post_id).updated_at = since
        db.commit()


def test_the_reconcile_tick_ends_a_publish_lost_inside_the_boot_cutoff(env):
    """A restart inside BOOT_REAPER_STALE_MINUTES leaves a post the boot reaper does
    not touch: the leader's pass ends it once no run can still hold it."""
    from modules.socials import schedule_jobs

    env.media = [VIDEO]
    lost = _approved_post(env, _target("linkedin", "text"), _target("linkedin", "video"))
    live = _approved_post(env, _target("linkedin", "text"))
    _claim(env, lost)
    _claim(env, live)
    now = datetime.now(timezone.utc)
    limit = config.SOCIALS_PUBLISH_RUN_MAX_SECONDS + schedule_jobs.LOST_MARGIN_SECONDS
    _publishing_since(env, lost, now - timedelta(seconds=limit + 60))
    _publishing_since(env, live, now - timedelta(seconds=limit - 60))  # its run may still be going
    with env.factory() as db:
        post = db.get(SocialPost, lost)
        {t.post_kind: t for t in post.targets}["video"].status = "uploading"
        db.commit()

    with env.factory() as db:
        assert schedule_jobs.end_lost_publishes(db, now) == 1

    assert _post(env, live).status == "publishing"
    post, targets = _post(env, lost), _targets(env, lost)
    assert post.status == "failed" and post.review_log[-1]["by"] == schedule_jobs.SCHEDULER_ACTOR
    assert "may have taken it" in targets["linkedin", "video"]["error"]
    assert targets["linkedin", "text"]["error"] == publish_lifecycle.NOT_TRIED
    with env.factory() as db:
        assert schedule_jobs.end_lost_publishes(db, now) == 0  # once


def test_a_publish_that_cannot_start_still_ends_the_post_and_retry_publishes_the_untried(env, monkeypatch):
    from modules.socials import publishing

    post_id = _approved_post(env, _target("linkedin", "text"))

    def broken(factory, job):
        raise RuntimeError("storage is down")

    with monkeypatch.context() as patched:
        patched.setattr(publishing, "load_work", broken)
        executor = FakeExecutor(LINKEDIN)
        assert _publish(env, post_id, executor) == "failed"
    assert executor.actions == []
    target = _targets(env, post_id)["linkedin", "text"]
    assert target["status"] == "failed" and target["error"] == publish_lifecycle.NOT_TRIED and target["attempts"] == 0
    assert [n["event_type"] for n in env.notices] == ["social_post_failed"]

    assert _publish(env, post_id, FakeExecutor(LINKEDIN), begin=publisher.begin_retry) == "published"



def test_the_default_executor_opens_a_session_per_call_and_closes_it(env, monkeypatch):
    """No session, or open transaction, is held across a publish that talks to the
    platforms for minutes: each Composio call has its own, closed when it ends."""
    post_id = _approved_post(env, _target("linkedin", "text"))
    script = FakeExecutor(LINKEDIN)
    sessions = []

    class Executor:
        def __init__(self, db):
            self.db = db
            sessions.append(SimpleNamespace(db=db, closed=False))

        async def execute_with_uploads(self, action, params, **kwargs):
            assert not sessions[-1].closed
            return await script.execute_with_uploads(action, params, **kwargs)

    def tracked():
        db = env.factory()
        close = db.close

        def closing():
            for s in sessions:
                if s.db is db:
                    s.closed = True
            close()

        db.close = closing
        return db

    monkeypatch.setattr(tool_executor, "ComposioToolExecutor", Executor)
    job = _claim(env, post_id)
    ended = asyncio.run(run_publish(job, session_factory=tracked, runtime=_runtime(env)))

    assert ended == "published" and script.actions == ["LINKEDIN_GET_MY_INFO", "LINKEDIN_CREATE_LINKED_IN_POST"]
    assert len(sessions) == 2 and all(s.closed for s in sessions)


# ---------------------------------------------------------------------------
# @integration: an attempt racing a lost end, on the CI Postgres
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_an_attempt_ending_while_its_post_is_ended_as_lost_does_not_overwrite_the_end(pg_engine, monkeypatch):
    """The sweep ends a post whose run looked gone, failing its uploading target; the
    run, still alive, then records the target published. The target row is locked on
    both sides, so the recorded end stands and the late attempt is not written over it."""
    from core.models.socials import SocialPostTarget
    from core.models.workspaces import Workspace

    monkeypatch.setattr(publish_records.media_urls, "resolve_post_media", lambda db, post: [])
    factory = sessionmaker(bind=pg_engine)
    workspace_id = uuid.uuid4()
    with factory() as db:
        db.add(Workspace(id=workspace_id, name="w3-lost-race", plan="basic", plan_limits={}, settings={}))
        db.commit()
        post = service.create_draft(db, workspace_id=workspace_id, created_by=AUTHOR, title="Race", copy={"base": "Race."})
        db.flush()
        service.update_post(post, AUTHOR, {"targets": [_target("linkedin", "text")]})
        service.submit(post, AUTHOR)
        service.approve(post, REVIEWER, content_hash=post.content_hash)
        db.commit()
        publisher.begin_publish(db, post, REVIEWER)
        post_id, target_id = post.id, post.targets[0].id
    try:
        assert publish_records.begin_attempt(factory, target_id)
        ending = factory()
        post = service.get_post(ending, workspace_id, post_id)
        publish_records.end_lost(post, "scheduler")  # locks the target rows until commit
        late = threading.Thread(target=publish_records.record_published, args=(factory, target_id, "urn:li:share:1", None, []))
        late.start()
        late.join(1)
        assert late.is_alive()  # it waits for the end to commit
        ending.commit()
        ending.close()
        late.join(10)
        with factory() as db:
            target = db.get(SocialPostTarget, target_id)
            assert target.status == "failed" and target.remote_id is None
            assert service.get_post(db, workspace_id, post_id).status == "failed"
    finally:
        with factory() as db:
            db.execute(sa.text("DELETE FROM social_posts WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": str(workspace_id)})
            db.execute(sa.text("DELETE FROM workspaces WHERE id = CAST(:ws AS uuid)"), {"ws": str(workspace_id)})
            db.commit()


# ---------------------------------------------------------------------------
# Review (P251W3): the title a channel publishes is approved content
# ---------------------------------------------------------------------------


def test_a_title_change_after_approval_voids_it_when_the_post_has_channels(env):
    post_id = _approved_post(env, _target("youtube", "video"))
    with env.factory() as db:
        post = service.get_post(db, WS, post_id)
        service.update_post(post, "agent:7", {"title": "A title nobody approved"}, agent="Social Media Director")
        db.commit()
        assert post.status == "needs_approval" and post.approved_hash != post.content_hash
    with pytest.raises(service.NotPublishable):
        _claim(env, post_id)


def test_a_title_is_not_content_for_a_post_with_no_channels():
    post = service.create_draft(SimpleNamespace(add=lambda obj: None), workspace_id=WS, created_by=AUTHOR, title="One")
    before = post.content_hash
    service.update_post(post, AUTHOR, {"title": "Two"})
    assert post.content_hash == before
