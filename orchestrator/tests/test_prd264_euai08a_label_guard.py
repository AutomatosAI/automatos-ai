"""PRD-264 EUAI-08a (EU AI Act Art. 50(4) ¶1): a video goes out only with the label the post sets.

media-render burns the AI-generated label into a video at render time; the caption label is
decided again at publish. So a video rendered with the label off and then switched on (or the
other way round), or rendered before the render record said whether it burned the label in,
would publish a caption and a video that disagree. ``ai_media.label_mismatch`` decides it from
the render record of the files the post carries (a record without ``labelled`` counts as no
label), and every gate refuses such a video until it is rendered again:

* ``service.approve`` (``NotPublishable``, 409 at the route), and a series approval leaves the
  post with the reason ``render_again``;
* publish now (``service.assert_publishable``) and Retry (``publish_lifecycle.assert_retryable``);
* the scheduled fire, which ends the post missed and publishes nothing.

A video without AI media, a still and an uploaded file need no burned-in label.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251w3_publisher as publisher_harness  # noqa: E402
import tests.test_prd251w3_schedule as schedule_harness  # noqa: E402
from core.media_render_bundle import ai_inputs  # noqa: E402
from modules.socials import ai_media, campaigns, deepfake_label, publish_lifecycle, schedule_jobs, service  # noqa: E402
from tests.test_prd251w3_publisher import LINKEDIN, WS, FakeExecutor  # noqa: E402

env = publisher_harness.env
fire = schedule_harness.fire

AUTHOR, REVIEWER = "member-1", "reviewer-1"
BASE = "Three weeks to go."
VIDEO_SHA = "e" * 64
VIDEO = {"9:16": [{"deliverable_id": "d-video-1", "name": "video-9x16.mp4", "sha256": VIDEO_SHA, "bytes": 10,
                   "content_type": "video/mp4"}]}
SPOKEN = {"audio": {"voice": {"lines": [{"id": "l01", "at": 0.3, "text": BASE}]}}, ai_media.INPUTS: ai_inputs(voice=True)}
SILENT = {"audio": {"music": {"track": "calm"}}, ai_media.INPUTS: ai_inputs()}
# A render record from before EUAI-08a: Kokoro spoke the script, and the record says nothing of the label.
BEFORE_THE_LABEL = {"footage": [], "stills": [], "own": [], "voice": "kokoro", "files": [VIDEO_SHA]}
SHOWS_NONE = ai_media.LABEL_MISMATCH.format(was="off", now="on")


class FakeDb:
    def add(self, row):
        self.row = row


def _render(post, bundle=SPOKEN, *, says=True):
    """The post rendered from ``bundle`` the way the render path labels it (``deepfake_label.with_label``),
    back waiting for approval; ``says`` False drops ``labelled`` from the record, as a render before EUAI-08a left it."""
    service.start_render(post, AUTHOR)
    record = ai_media.render_record(post, deepfake_label.with_label(bundle, post), VIDEO)
    if not says:
        record = {key: value for key, value in record.items() if key != ai_media.LABELLED}
    return service.finish_render(post, "render", VIDEO, report={"ai_media": record})


def _rendered(bundle=SPOKEN, setting=None, *, says=True):
    """A video post whose label a person set to ``setting`` (``None``: nobody did), then rendered."""
    post = service.create_draft(FakeDb(), workspace_id=uuid.uuid4(), created_by=AUTHOR, title="Countdown",
                                copy={"base": BASE}, format="video")
    if setting is not None:
        deepfake_label.set_label(post, AUTHOR, setting)
    return _render(post, bundle, says=says)


def _render_done(record):
    return {"at": datetime.now(timezone.utc).isoformat(), "by": "render", "action": "render_done", "comment": None,
            "report": {"ai_media": record}}


def _as_rendered(env, post_id, record):
    """The committed post as a video render recording ``record`` left it, its approval moved onto it
    (as an approval given before this guard existed)."""
    with env.factory() as db:
        post = service.get_post(db, WS, post_id)
        post.format, post.media = "video", VIDEO
        post.review_log = [*(post.review_log or []), _render_done(record)]
        post.content_hash = post.approved_hash = service.compute_content_hash(post)
        db.commit()


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("before, after, was, now", [(False, True, "off", "on"), (None, False, "on", "off")])
def test_a_video_whose_label_was_switched_after_its_render_is_not_approved_until_rendered_again(before, after, was, now):
    post = _rendered(setting=before)
    deepfake_label.set_label(post, AUTHOR, after)
    assert post.status == service.NEEDS_APPROVAL

    reason = ai_media.LABEL_MISMATCH.format(was=was, now=now)
    assert ai_media.label_mismatch(post) == reason
    with pytest.raises(service.NotPublishable) as refused:
        service.approve(post, REVIEWER, content_hash=post.content_hash)
    assert str(refused.value) == reason and post.status == service.NEEDS_APPROVAL
    receipt = ai_media.labels_for(post)["deepfake_label"]
    assert receipt["video_labelled"] is (was == "on")
    assert receipt["line"].endswith(
        f"The video was rendered with the label {was}: render it again before it is approved or published."
    )

    _render(post)  # rendered again: the burned-in label follows the post's setting now
    assert ai_media.label_mismatch(post) is None
    service.approve(post, REVIEWER, content_hash=post.content_hash)
    assert post.status == service.APPROVED


def test_a_video_rendered_before_the_label_existed_shows_none_and_is_neither_approved_nor_published():
    post = _rendered(says=False)  # on by default: a synthetic voice spoke its script
    assert ai_media.burned_label(post) is False
    with pytest.raises(service.NotPublishable) as refused:
        service.approve(post, REVIEWER, content_hash=post.content_hash)
    assert str(refused.value) == SHOWS_NONE

    post.status, post.approved_hash = service.APPROVED, post.content_hash  # approved before this guard existed
    for guard in (service.assert_publishable, lambda p: publish_lifecycle.start_publish(p, REVIEWER)):
        with pytest.raises(service.NotPublishable) as refused:
            guard(post)
        assert str(refused.value) == SHOWS_NONE
    assert post.status == service.APPROVED
    post.status = service.FAILED
    with pytest.raises(service.NotPublishable) as refused:
        publish_lifecycle.assert_retryable(post)
    assert str(refused.value) == SHOWS_NONE


def test_a_video_without_ai_media_a_still_and_an_upload_need_no_burned_label():
    own = _rendered(SILENT, says=False)  # no AI media: the label is off, and the video shows none
    assert ai_media.burned_label(own) is False and ai_media.label_mismatch(own) is None
    service.approve(own, REVIEWER, content_hash=own.content_hash)
    assert own.status == service.APPROVED

    still = SimpleNamespace(media={"1:1": [{"sha256": "f" * 64, "content_type": "image/png"}]}, review_log=[],
                            format="image", footage=None, voice=None)
    assert ai_media.burned_label(still) is None and ai_media.label_mismatch(still) is None
    upload = SimpleNamespace(media={"upload": ["d-upload-1"]}, review_log=[], format="video", footage=None, voice=None)
    assert ai_media.label_mismatch(upload) is None  # the person's own file: Deliverable ids, no rendered video


def test_a_series_approval_leaves_such_a_video_saying_why(monkeypatch):
    post = _rendered(says=False)
    campaign_id = uuid.uuid4()
    post.campaign_id = campaign_id
    monkeypatch.setattr(campaigns.service, "get_post", lambda db, workspace_id, post_id: post)
    monkeypatch.setattr(campaigns.post_sources, "unresolved", lambda db, workspace_id, sources: {})
    item = campaigns.ShownPost(post_id=post.id, content_hash=post.content_hash)

    done, left = campaigns._approve_one(object(), (post.workspace_id, campaign_id), REVIEWER, item, None)

    assert done is None and post.status == service.NEEDS_APPROVAL
    assert (left["reason"], left["message"]) == (campaigns.LEFT_RENDER_AGAIN, SHOWS_NONE)
    assert campaigns.LEFT_RENDER_AGAIN == "render_again"  # frontend/lib/api-client.ts SocialSeriesLeftReason


# ---------------------------------------------------------------------------
# Publish now, Retry and the scheduled fire, over the real publisher
# ---------------------------------------------------------------------------


def test_publish_now_and_retry_refuse_it_and_a_render_with_the_label_publishes(env):
    post_id = publisher_harness._approved_post(env, publisher_harness._target("linkedin", "text"))
    _as_rendered(env, post_id, BEFORE_THE_LABEL)
    with pytest.raises(service.NotPublishable, match="render it again"):
        publisher_harness._claim(env, post_id)
    assert publisher_harness._post(env, post_id).status == "approved"

    _as_rendered(env, post_id, {**BEFORE_THE_LABEL, ai_media.LABELLED: True})  # rendered again, with the label
    assert publisher_harness._publish(env, post_id, FakeExecutor(LINKEDIN)) == "published"


def test_a_scheduled_video_without_its_label_is_missed_and_nothing_is_published(env, fire):
    slot = schedule_harness._slot(1)
    post_id = schedule_harness._scheduled_post(env, slot)
    _as_rendered(env, post_id, BEFORE_THE_LABEL)

    assert fire.run(post_id, slot) == "missed" and fire.executor.calls == []
    entry = publisher_harness._post(env, post_id).review_log[-1]
    assert (entry["action"], entry["comment"]) == ("missed", schedule_jobs.MISSED_UNLABELLED.format(why=SHOWS_NONE))
