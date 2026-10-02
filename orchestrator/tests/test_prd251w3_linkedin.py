"""PRD-251 Wave 3, US-301 (S3.3a) — LinkedIn: the engine proven with LinkedIn.

With the executor mocked (the harness in ``tests/test_prd251w3_publisher.py``): a
LinkedIn text, image and video post each issue exactly the adapter's sequence, through
``execute_with_uploads`` with ``way_through=PLATFORM_PUBLISHER`` and the step's files
as the call's own upload spec, and store the remote id and permalink. An image goes
through the workspace-scoped image workaround (the executor hands it the staged file);
a video uploads (``LINKEDIN_UPLOAD_VIDEO``) and posts with the returned video URN
(``LINKEDIN_CREATE_VIDEO_POST``).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251w3_publisher as harness  # noqa: E402
from core.composio.post_gate import PLATFORM_PUBLISHER  # noqa: E402
from tests.test_prd251w3_publisher import (  # noqa: E402
    IMAGE, LINKEDIN, ME_URN, SHARE_URN, VIDEO, VIDEO_URN, FakeExecutor, _approved_post, _post, _publish, _target, _targets,
)

env = harness.env


def test_a_linkedin_text_post_runs_the_adapter_sequence_and_stores_its_receipt(env):
    post_id = _approved_post(env, _target("linkedin", "text"), copy={"base": "Base copy", "channels": {"linkedin": "LinkedIn copy"}})
    executor = FakeExecutor(LINKEDIN)

    assert _publish(env, post_id, executor) == "published"

    assert executor.actions == ["LINKEDIN_GET_MY_INFO", "LINKEDIN_CREATE_LINKED_IN_POST"]
    assert all(call.way_through is PLATFORM_PUBLISHER and call.app_name == "LINKEDIN" for call in executor.calls)
    assert executor.calls[1].params == {"author": ME_URN, "commentary": "LinkedIn copy"}  # the channel's own copy
    assert executor.calls[1].upload_params == ()
    target = _targets(env, post_id)["linkedin", "text"]
    assert target["status"] == "published" and target["attempts"] == 1 and target["published_at"]
    assert target["remote_id"] == SHARE_URN
    assert target["permalink"] == f"https://www.linkedin.com/feed/update/{SHARE_URN}/"
    post = _post(env, post_id)
    assert post.status == "published" and post.review_log[-1]["action"] == "published"
    assert [n["event_type"] for n in env.notices] == ["social_post_published"]


def test_a_chosen_author_option_wins_over_the_account_lookup(env):
    post_id = _approved_post(env, _target("linkedin", "text", author="urn:li:organization:42"))
    executor = FakeExecutor(LINKEDIN)
    assert _publish(env, post_id, executor) == "published"
    assert executor.calls[-1].params["author"] == "urn:li:organization:42"


def test_a_linkedin_image_post_hands_its_files_to_the_upload_spec(env):
    env.media = [IMAGE, VIDEO]
    post_id = _approved_post(env, _target("linkedin", "image"))
    executor = FakeExecutor(LINKEDIN)

    assert _publish(env, post_id, executor) == "published"

    assert executor.actions == ["LINKEDIN_GET_MY_INFO", "LINKEDIN_CREATE_LINKED_IN_POST"]
    post_call = executor.calls[1]
    assert post_call.upload_params == ("images",)
    assert [p.name for p in post_call.params["images"]] == ["card.png"]  # only the image, as a staged file
    assert all(isinstance(p, Path) for p in post_call.params["images"])
    assert _targets(env, post_id)["linkedin", "image"]["remote_id"] == SHARE_URN


def test_a_linkedin_video_uploads_then_posts_with_the_returned_video_urn(env):
    env.media = [IMAGE, VIDEO]
    post_id = _approved_post(env, _target("linkedin", "video"))
    executor = FakeExecutor(LINKEDIN)

    assert _publish(env, post_id, executor) == "published"

    assert executor.actions == ["LINKEDIN_UPLOAD_VIDEO", "LINKEDIN_CREATE_VIDEO_POST"]
    upload, post = executor.calls
    assert upload.upload_params == ("file",) and upload.params["file"].name == "video.mp4"
    assert post.params == {"video_urn": VIDEO_URN, "commentary": "Harvest Club opens Friday."}
    target = _targets(env, post_id)["linkedin", "video"]
    assert (target["remote_id"], target["permalink"]) == (SHARE_URN, f"https://www.linkedin.com/feed/update/{SHARE_URN}/")


def test_a_video_target_with_no_video_fails_saying_so_and_calls_nothing_for_it(env):
    env.media = [IMAGE]
    post_id = _approved_post(env, _target("linkedin", "video"))
    executor = FakeExecutor(LINKEDIN)

    assert _publish(env, post_id, executor) == "failed"

    assert executor.calls == []
    assert "needs a video file for file" in _targets(env, post_id)["linkedin", "video"]["error"]


