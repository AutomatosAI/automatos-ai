"""PRD-251 Wave 3, US-302 (S3.3b) — X: text, image and video, on the one engine.

With the executor mocked (the US-301 harness): a text post is one
``TWITTER_CREATION_OF_A_POST``; an image uploads first (``TWITTER_UPLOAD_MEDIA``) and
the post carries the returned media id; a video takes the chunked upload
(``TWITTER_UPLOAD_LARGE_MEDIA``), is polled until X has processed it
(``TWITTER_GET_MEDIA_UPLOAD_STATUS``), then posted. The receipt holds the post's id and
its link. X needs the customer's own X app ("Waves 2+3 build"): a refusal of the
connection's credentials is not retried and names the channel's setup note. The stale
``TWITTER_CREATE_TWEET`` is never called, and the engine has no X-specific code.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251w3_publisher as harness  # noqa: E402
from config import config  # noqa: E402
from core.composio.post_gate import PLATFORM_PUBLISHER  # noqa: E402
from modules.socials.capabilities import SEEDED_CHANNELS  # noqa: E402
from tests.test_prd251w3_publisher import (  # noqa: E402
    IMAGE, VIDEO, FakeExecutor, _approved_post, _publish, _target, _targets, ok, refused,
)

env = harness.env  # the US-301 fixture: SQLite, the recording executor, notices

TWEET_ID = "1843000000000000001"
MEDIA_ID = "1843000000000000777"
X_LINK = f"https://x.com/i/web/status/{TWEET_ID}"
X = {
    "TWITTER_CREATION_OF_A_POST": [ok({"data": {"id": TWEET_ID, "text": "Harvest Club opens Friday."}})],
    "TWITTER_UPLOAD_MEDIA": [ok({"media_id_string": MEDIA_ID, "media_id": int(MEDIA_ID)})],
    "TWITTER_UPLOAD_LARGE_MEDIA": [ok({"media_id_string": MEDIA_ID, "processing_info": {"state": "pending"}})],
    "TWITTER_GET_MEDIA_UPLOAD_STATUS": [
        ok({"media_id_string": MEDIA_ID, "processing_info": {"state": "in_progress", "progress_percent": 40}}),
        ok({"media_id_string": MEDIA_ID, "processing_info": {"state": "succeeded", "progress_percent": 100}}),
    ],
}


def _receipt(env, post_id, kind):
    target = _targets(env, post_id)["twitter", kind]
    return target["status"], target["remote_id"], target["permalink"]


def test_an_x_text_post_is_one_post_call_and_its_receipt_is_the_tweet(env):
    post_id = _approved_post(env, _target("twitter", "text"), copy={"base": "Base", "channels": {"twitter": "Short for X"}})
    executor = FakeExecutor(X)

    assert _publish(env, post_id, executor) == "published"

    assert executor.actions == ["TWITTER_CREATION_OF_A_POST"]
    (call,) = executor.calls
    assert call.params == {"text": "Short for X"} and call.way_through is PLATFORM_PUBLISHER and call.app_name == "TWITTER"
    assert _receipt(env, post_id, "text") == ("published", TWEET_ID, X_LINK)


def test_an_x_image_uploads_the_file_then_posts_with_its_media_id(env):
    env.media = [IMAGE, VIDEO]
    post_id = _approved_post(env, _target("twitter", "image"))
    executor = FakeExecutor(X)

    assert _publish(env, post_id, executor) == "published"

    assert executor.actions == ["TWITTER_UPLOAD_MEDIA", "TWITTER_CREATION_OF_A_POST"]
    upload, post = executor.calls
    assert upload.upload_params == ("media",) and upload.params["media"].name == "card.png"
    assert upload.params["media_type"] == "image/png"
    assert post.params == {"text": "Harvest Club opens Friday.", "media_media_ids": [MEDIA_ID]}
    assert _receipt(env, post_id, "image") == ("published", TWEET_ID, X_LINK)


def test_an_x_video_is_chunked_polled_until_processed_then_posted(env, monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_PUBLISH_POLL_SECONDS", 3)
    env.media = [VIDEO]
    post_id = _approved_post(env, _target("twitter", "video"))
    executor = FakeExecutor(X)

    assert _publish(env, post_id, executor) == "published"

    assert executor.actions == [
        "TWITTER_UPLOAD_LARGE_MEDIA", "TWITTER_GET_MEDIA_UPLOAD_STATUS", "TWITTER_GET_MEDIA_UPLOAD_STATUS",
        "TWITTER_CREATION_OF_A_POST",
    ]
    upload = executor.calls[0]
    assert upload.upload_params == ("media",)
    assert upload.params["media_type"] == "video/mp4" and upload.params["total_bytes"] == VIDEO.bytes
    assert executor.calls[1].params == {"media_id": MEDIA_ID}
    assert executor.calls[-1].params["media_media_ids"] == [MEDIA_ID]
    assert env.sleeps == [3]  # polled once more after "in_progress"
    assert _receipt(env, post_id, "video") == ("published", TWEET_ID, X_LINK)


def test_a_video_x_fails_to_process_fails_its_target_with_x_s_reason(env):
    env.media = [VIDEO]
    post_id = _approved_post(env, _target("twitter", "video"))
    failed = ok({"processing_info": {"state": "failed", "error": {"name": "InvalidMedia", "message": "Unsupported codec"}}})
    executor = FakeExecutor({**X, "TWITTER_GET_MEDIA_UPLOAD_STATUS": [failed]})

    assert _publish(env, post_id, executor) == "failed"

    assert "TWITTER_CREATION_OF_A_POST" not in executor.actions
    assert "Unsupported codec" in _targets(env, post_id)["twitter", "video"]["error"]


def test_an_auth_refusal_is_not_retried_and_names_the_x_setup_note(env):
    post_id = _approved_post(env, _target("twitter", "text"))
    executor = FakeExecutor({"TWITTER_CREATION_OF_A_POST": [refused("401 Unauthorized: Could not authenticate you")]})

    assert _publish(env, post_id, executor) == "failed"

    assert executor.actions == ["TWITTER_CREATION_OF_A_POST"] and env.sleeps == []
    target = _targets(env, post_id)["twitter", "text"]
    assert target["attempts"] == 1 and "Could not authenticate you" in target["error"]
    assert SEEDED_CHANNELS["twitter"].setup_note in target["error"]


def test_a_channel_without_a_setup_note_keeps_the_platform_message_alone(env):
    post_id = _approved_post(env, _target("linkedin", "text"))
    executor = FakeExecutor({**harness.LINKEDIN, "LINKEDIN_CREATE_LINKED_IN_POST": [refused("401 Unauthorized")]})
    assert _publish(env, post_id, executor) == "failed"
    assert _targets(env, post_id)["linkedin", "text"]["error"] == "LINKEDIN_CREATE_LINKED_IN_POST: 401 Unauthorized"


@pytest.mark.parametrize("kind", ["text", "image", "video"])
def test_no_x_sequence_calls_the_stale_create_tweet(kind):
    actions = [step.action for step in SEEDED_CHANNELS["twitter"].kinds[kind]]
    assert "TWITTER_CREATE_TWEET" not in actions and "TWITTER_CREATION_OF_A_POST" in actions


def test_the_engine_has_no_channel_specific_code():
    engine = [_ORCH / "modules" / "socials" / name for name in (
        "publisher.py", "publishing.py", "publish_steps.py", "publish_sources.py", "publish_records.py",
        "publish_lifecycle.py", "step_results.py",
    )]
    channel = re.compile(r"\b(twitter|linkedin|instagram|tiktok|youtube)\b", re.IGNORECASE)
    for path in engine:
        code = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if not line.strip().startswith("#"))
        body = code.split('"""', 2)[-1]  # after the module docstring
        assert not channel.search(body), f"{path.name} names a channel"


def test_an_x_image_whose_upload_returns_no_media_id_publishes_nothing(env):
    """Final review: the post step reads the upload's id ($steps.media). An upload
    whose answer has it nowhere fails the target; the tweet is never sent without
    its image, and never recorded published."""
    env.media = [IMAGE, VIDEO]
    post_id = _approved_post(env, _target("twitter", "image"))
    executor = FakeExecutor({**X, "TWITTER_UPLOAD_MEDIA": [ok({"data": {"media_key": "3_123"}})]})

    assert _publish(env, post_id, executor) == "failed"

    assert executor.actions == ["TWITTER_UPLOAD_MEDIA"]  # not retried, and no post call
    target = _targets(env, post_id)["twitter", "image"]
    assert target["status"] == "failed" and "TWITTER_UPLOAD_MEDIA returned no id" in target["error"]
