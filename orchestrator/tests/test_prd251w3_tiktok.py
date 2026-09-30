"""PRD-251 Wave 3, US-304 (S3.3d) — TikTok: the file uploaded, privacy from the
creator's own levels, the AI label, the publish status polled.

With the executor mocked (the US-301 harness): ``TIKTOK_QUERY_CREATOR_INFO`` first
(the privacy levels this creator may use), then ``TIKTOK_UPLOAD_VIDEO`` with the FILE
(never the URL-pull ``TIKTOK_PUBLISH_VIDEO`` / ``TIKTOK_POST_PHOTO``: they fetch from a
domain the app owner verified, and Composio owns the app), then
``TIKTOK_FETCH_PUBLISH_STATUS`` until ``PUBLISH_COMPLETE``.

* ``privacy_level`` is the target's option when the creator may use it, else the most
  private level allowed; never a public default the person did not choose.
* ``is_aigc`` is true when the post's footage was generated (a slot recorded done,
  D12) or the target's option says so.
* The receipt stores the publish id. A status that reports a failure fails the
  target with TikTok's reason; one that never finishes fails it saying so.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251w3_publisher as harness  # noqa: E402
from api.socials_targets import kind_options  # noqa: E402
from config import config  # noqa: E402
from modules.socials import service  # noqa: E402
from modules.socials.capabilities import SEEDED_CHANNELS, ChannelKind, parse_channel_adapters  # noqa: E402
from tests.test_prd251w3_publisher import VIDEO, WS, FakeExecutor, _approved_post, _publish, _target, _targets, ok  # noqa: E402

env = harness.env

PUBLISH_ID = "v_pub_file~v2-1.7428"
ALL_LEVELS = ["PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS", "FOLLOWER_OF_CREATOR", "SELF_ONLY"]


def _tiktok(levels=ALL_LEVELS, statuses=("PROCESSING_UPLOAD", "PUBLISH_COMPLETE")):
    return {
        "TIKTOK_QUERY_CREATOR_INFO": [ok({"creator_username": "harvestclub", "privacy_level_options": list(levels)})],
        "TIKTOK_UPLOAD_VIDEO": [ok({"publish_id": PUBLISH_ID})],
        "TIKTOK_FETCH_PUBLISH_STATUS": [ok({"status": status, "fail_reason": "spam_risk_too_many_posts"}) for status in statuses],
    }


def _upload_params(executor):
    return next(call for call in executor.calls if call.action == "TIKTOK_UPLOAD_VIDEO").params


def _generated_footage(env, post_id):
    with env.factory() as db:
        post = service.get_post(db, WS, post_id)
        post.footage = {"hook": {"prompt": "A harvest table", "status": "done", "toolkit": "fal_ai"}}
        db.commit()


def test_the_sequence_is_creator_info_upload_the_file_then_poll_until_published(env, monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_PUBLISH_POLL_SECONDS", 6)
    env.media = [VIDEO]
    post_id = _approved_post(env, _target("tiktok", "video", privacy_level="FOLLOWER_OF_CREATOR"))
    executor = FakeExecutor(_tiktok())

    assert _publish(env, post_id, executor) == "published"

    assert executor.actions == [
        "TIKTOK_QUERY_CREATOR_INFO", "TIKTOK_UPLOAD_VIDEO", "TIKTOK_FETCH_PUBLISH_STATUS", "TIKTOK_FETCH_PUBLISH_STATUS",
    ]
    upload = executor.calls[1]
    assert upload.upload_params == ("file_to_upload",) and upload.params["file_to_upload"].name == "video.mp4"
    assert upload.params["privacy_level"] == "FOLLOWER_OF_CREATOR"  # the person's choice, allowed
    assert upload.params["publish"] is True and upload.params["caption"] == "Harvest Club opens Friday."
    assert "is_aigc" not in upload.params  # no generated footage, no option: TikTok's own default
    assert executor.calls[2].params == {"publish_id": PUBLISH_ID}
    assert env.sleeps == [6]
    target = _targets(env, post_id)["tiktok", "video"]
    assert (target["status"], target["remote_id"]) == ("published", PUBLISH_ID)


def test_a_privacy_level_the_creator_may_not_use_falls_back_to_the_most_private(env):
    env.media = [VIDEO]
    post_id = _approved_post(env, _target("tiktok", "video", privacy_level="PUBLIC_TO_EVERYONE"))
    executor = FakeExecutor(_tiktok(levels=["FOLLOWER_OF_CREATOR", "SELF_ONLY"]))
    assert _publish(env, post_id, executor) == "published"
    assert _upload_params(executor)["privacy_level"] == "SELF_ONLY"


def test_no_privacy_choice_is_never_a_public_default(env):
    env.media = [VIDEO]
    post_id = _approved_post(env, _target("tiktok", "video"))
    executor = FakeExecutor(_tiktok(levels=["PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS"]))
    assert _publish(env, post_id, executor) == "published"
    assert _upload_params(executor)["privacy_level"] == "MUTUAL_FOLLOW_FRIENDS"  # the most private allowed


def test_a_creator_allowing_only_public_is_refused_unless_the_person_chose_public(env):
    env.media = [VIDEO]
    refused = _approved_post(env, _target("tiktok", "video"))
    executor = FakeExecutor(_tiktok(levels=["PUBLIC_TO_EVERYONE"]))
    assert _publish(env, refused, executor) == "failed"
    assert "TIKTOK_UPLOAD_VIDEO" not in executor.actions
    assert "none of SELF_ONLY, MUTUAL_FOLLOW_FRIENDS, FOLLOWER_OF_CREATOR is allowed" in _targets(env, refused)["tiktok", "video"]["error"]

    chosen = _approved_post(env, _target("tiktok", "video", privacy_level="PUBLIC_TO_EVERYONE"))
    executor = FakeExecutor(_tiktok(levels=["PUBLIC_TO_EVERYONE"]))
    assert _publish(env, chosen, executor) == "published"
    assert _upload_params(executor)["privacy_level"] == "PUBLIC_TO_EVERYONE"


def test_generated_footage_sets_the_ai_label(env):
    env.media = [VIDEO]
    post_id = _approved_post(env, _target("tiktok", "video", is_aigc=False))
    _generated_footage(env, post_id)  # a render setting: the approval stands
    executor = FakeExecutor(_tiktok())
    assert _publish(env, post_id, executor) == "published"
    assert _upload_params(executor)["is_aigc"] is True


def test_the_person_s_ai_label_is_kept_without_generated_footage(env):
    env.media = [VIDEO]
    post_id = _approved_post(env, _target("tiktok", "video", is_aigc=True))
    executor = FakeExecutor(_tiktok())
    assert _publish(env, post_id, executor) == "published"
    assert _upload_params(executor)["is_aigc"] is True


def test_a_status_that_reports_a_failure_fails_the_target_with_tiktok_s_reason(env):
    env.media = [VIDEO]
    post_id = _approved_post(env, _target("tiktok", "video"))
    executor = FakeExecutor(_tiktok(statuses=("FAILED",)))
    assert _publish(env, post_id, executor) == "failed"
    assert executor.actions.count("TIKTOK_UPLOAD_VIDEO") == 1  # a platform's refusal is not retried
    error = _targets(env, post_id)["tiktok", "video"]["error"]
    assert "spam_risk_too_many_posts" in error
    assert "check the channel" not in error  # TikTok said it failed: nothing is live


def test_a_publish_that_never_finishes_fails_saying_so(env, monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_PUBLISH_MAX_WAIT_SECONDS", 30)
    monkeypatch.setattr(config, "SOCIALS_PUBLISH_POLL_SECONDS", 10)
    env.media = [VIDEO]
    post_id = _approved_post(env, _target("tiktok", "video"))
    executor = FakeExecutor(_tiktok(statuses=("PROCESSING_UPLOAD",)))
    assert _publish(env, post_id, executor) == "failed"
    assert executor.actions.count("TIKTOK_FETCH_PUBLISH_STATUS") == 4  # t = 0, 10, 20, 30
    error = _targets(env, post_id)["tiktok", "video"]["error"]
    assert "TIKTOK_FETCH_PUBLISH_STATUS did not finish within 30 seconds" in error
    # The upload (the publish step) ran: the post may be live, so a retry is not blind.
    assert "check the channel before you retry" in error


def test_the_url_pull_actions_are_never_in_a_sequence_and_the_gate_still_refuses_them():
    adapter = SEEDED_CHANNELS["tiktok"]
    actions = {step.action for steps in adapter.kinds.values() for step in steps}
    assert not actions & {"TIKTOK_PUBLISH_VIDEO", "TIKTOK_POST_PHOTO"}
    assert {"TIKTOK_PUBLISH_VIDEO", "TIKTOK_POST_PHOTO"} <= adapter.publish_actions


def test_the_tiktok_kind_still_takes_the_privacy_and_ai_label_options():
    steps = SEEDED_CHANNELS["tiktok"].kinds["video"]
    assert kind_options(ChannelKind("video", True, None, False, steps)) == {"privacy_level", "is_aigc"}


def test_a_choice_must_be_well_formed():
    bad = {"tiktok": {"label": "T", "kinds": {"video": [
        {"id": "upload", "action": "TIKTOK_UPLOAD_VIDEO", "class": "publish", "params": {"privacy_level": {"choose": "$option.p"}},
         "returns": {"id": "publish_id"}},
    ]}}}
    with pytest.raises(ValueError, match="a choice is"):
        parse_channel_adapters(bad)
    unknown = {"tiktok": {"label": "T", "kinds": {"video": [
        {"id": "upload", "action": "TIKTOK_UPLOAD_VIDEO", "class": "publish",
         "params": {"privacy_level": {"choose": "$option.p", "among": "$steps.creator.levels", "else": []}}, "returns": {"id": "publish_id"}},
    ]}}}
    with pytest.raises(ValueError, match="names an earlier step"):
        parse_channel_adapters(unknown)
