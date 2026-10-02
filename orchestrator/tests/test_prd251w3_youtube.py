"""PRD-251 Wave 3, US-305 (S3.3e) — YouTube, and the generic adapter.

With the executor mocked (the US-301 harness):

* YouTube: ``YOUTUBE_UPLOAD_VIDEO`` with the video as a FILE (``videoFilePath``, the
  adapter's own upload spec), the title from the post's title, the description from
  its copy, and categoryId, privacyStatus and tags from the target's options (the
  composer writes category_id, privacy_status and tags). The receipt is the video id
  and its watch URL.
* The custom thumbnail (``YOUTUBE_UPDATE_THUMBNAIL``) takes only a link YouTube
  fetches (D9): with public storage it gets a presigned inline link to the post's
  still; without, the optional step is skipped and the receipt says so.
* The generic adapter (D8 "Generic (text + media)"): a connected toolkit's one create
  action publishes the copy and the media through the same engine, and its first
  published target makes the channel verified (the "unverified channel" label goes).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251w3_publisher as harness  # noqa: E402
from api.socials_targets import step_plan  # noqa: E402
from modules.socials import capabilities  # noqa: E402
from modules.socials.media_urls import NeedsPublicStorage  # noqa: E402
from modules.socials.publishing import run_publish  # noqa: E402
from tests.test_prd251w3_publisher import IMAGE, VIDEO, WS, FakeExecutor, _approved_post, _claim, _runtime, _target, _targets, ok  # noqa: E402

env = harness.env

VIDEO_ID = "dQw4w9WgXcQ"
WATCH = f"https://www.youtube.com/watch?v={VIDEO_ID}"
YOUTUBE = {
    "YOUTUBE_UPLOAD_VIDEO": [ok({"id": VIDEO_ID, "snippet": {"title": "Harvest Club"}})],
    "YOUTUBE_UPDATE_THUMBNAIL": [ok({"items": [{"default": {"url": "https://i.ytimg.com/vi/x/default.jpg"}}]})],
}
OPTIONS = {"category_id": "22", "privacy_status": "unlisted", "tags": ["harvest", "club"]}


def _publish(env, post_id, executor, public_link):
    job = _claim(env, post_id)
    links = []

    def build(executor, stager):
        runtime = _runtime(env)(executor, stager)

        def link(post, media):
            links.append((post.id, post.workspace_id, media.name))
            return public_link(post, media)

        runtime.public_link = link
        return runtime

    ended = asyncio.run(run_publish(job, executor=executor, session_factory=env.factory, runtime=build))
    return ended, links


def _no_public_storage(post, media):
    raise NeedsPublicStorage()


def test_the_upload_takes_the_file_the_title_the_copy_and_the_options(env):
    env.media = [VIDEO, IMAGE]
    post_id = _approved_post(env, _target("youtube", "video", **OPTIONS), copy={"base": "Base", "channels": {"youtube": "Description for YouTube"}})
    executor = FakeExecutor(YOUTUBE)

    ended, _ = _publish(env, post_id, executor, lambda post, media: "https://public.example/still.png?sig=1")

    assert ended == "published"
    upload = executor.calls[0]
    assert upload.action == "YOUTUBE_UPLOAD_VIDEO" and upload.upload_params == ("videoFilePath",)
    assert upload.params["videoFilePath"].name == "video.mp4"
    assert {k: v for k, v in upload.params.items() if k != "videoFilePath"} == {
        "title": "Harvest Club", "description": "Description for YouTube",
        "categoryId": "22", "privacyStatus": "unlisted", "tags": ["harvest", "club"],
    }
    target = _targets(env, post_id)["youtube", "video"]
    assert (target["status"], target["remote_id"], target["permalink"]) == ("published", VIDEO_ID, WATCH)


def test_with_public_storage_the_thumbnail_gets_a_presigned_link_to_the_still(env):
    env.media = [VIDEO, IMAGE]
    post_id = _approved_post(env, _target("youtube", "video", **OPTIONS))
    executor = FakeExecutor(YOUTUBE)
    link = "https://media.example/social-media/ws/post/card.png?X-Amz-Signature=abc&response-content-disposition=inline"

    ended, links = _publish(env, post_id, executor, lambda post, media: link)

    assert ended == "published"
    assert executor.actions == ["YOUTUBE_UPLOAD_VIDEO", "YOUTUBE_UPDATE_THUMBNAIL"]
    thumbnail = executor.calls[1]
    assert thumbnail.params == {"videoId": VIDEO_ID, "thumbnailUrl": link} and thumbnail.upload_params == ()
    assert links == [(post_id, WS, "card.png")]  # the post's still, in its own workspace
    assert _targets(env, post_id)["youtube", "video"]["notes"] == []


def test_without_public_storage_the_thumbnail_is_skipped_and_the_receipt_says_so(env):
    env.media = [VIDEO, IMAGE]
    post_id = _approved_post(env, _target("youtube", "video", **OPTIONS))
    executor = FakeExecutor(YOUTUBE)

    ended, _ = _publish(env, post_id, executor, _no_public_storage)

    assert ended == "published"
    assert executor.actions == ["YOUTUBE_UPLOAD_VIDEO"]
    target = _targets(env, post_id)["youtube", "video"]
    assert target["permalink"] == WATCH
    assert len(target["notes"]) == 1 and "Needs public storage" in target["notes"][0]
    assert "YOUTUBE_UPDATE_THUMBNAIL" in target["notes"][0]


def test_a_thumbnail_youtube_refuses_does_not_fail_the_published_video(env):
    env.media = [VIDEO, IMAGE]
    post_id = _approved_post(env, _target("youtube", "video", **OPTIONS))
    executor = FakeExecutor({**YOUTUBE, "YOUTUBE_UPDATE_THUMBNAIL": [harness.refused("403 Forbidden: the channel cannot set custom thumbnails")]})

    ended, _ = _publish(env, post_id, executor, lambda post, media: "https://public.example/card.png")

    assert ended == "published"
    target = _targets(env, post_id)["youtube", "video"]
    assert target["remote_id"] == VIDEO_ID and "cannot set custom thumbnails" in target["notes"][0]


# ---------------------------------------------------------------------------
# The generic adapter
# ---------------------------------------------------------------------------

GENERIC_SLUG = "BLUESKYISH_CREATE_POST"
GENERIC_SCHEMA = {"properties": {"text": {"type": "string"}, "image_file": {"type": "string", "file_uploadable": True}}}


def _generic_target(kind):
    fields = capabilities._generic_fields(GENERIC_SCHEMA)
    storage = capabilities._Storage(public=False, needs="Needs public storage")
    (offered,) = [k for k in capabilities._generic_kinds(GENERIC_SLUG, fields, storage) if k.kind == kind]
    return {"toolkit": "blueskyish", "post_kind": kind, "options": {}, "steps": [step_plan(step) for step in offered.steps]}


def test_a_generic_channel_publishes_through_its_one_create_action_and_becomes_verified(env):
    env.media = [IMAGE]
    post_id = _approved_post(env, _generic_target("image"))
    with env.factory() as db:
        assert "blueskyish" not in capabilities._published_toolkits(db, WS)  # "unverified channel" until now
    executor = FakeExecutor({GENERIC_SLUG: [ok({"id": "at://did:plc:abc/post/1", "url": "https://bsky.example/p/1"})]})

    ended, _ = _publish(env, post_id, executor, _no_public_storage)

    assert ended == "published"
    (call,) = executor.calls
    assert call.action == GENERIC_SLUG and call.upload_params == ("image_file",)
    assert call.params["text"] == "Harvest Club opens Friday." and call.params["image_file"].name == "card.png"
    target = _targets(env, post_id)["blueskyish", "image"]
    assert (target["remote_id"], target["permalink"]) == ("at://did:plc:abc/post/1", "https://bsky.example/p/1")
    with env.factory() as db:
        assert "blueskyish" in capabilities._published_toolkits(db, WS)  # verified: the label goes


def test_a_generic_text_post_sends_the_copy_alone(env):
    post_id = _approved_post(env, _generic_target("text"))
    executor = FakeExecutor({GENERIC_SLUG: [ok({"id": "p-2"})]})
    ended, _ = _publish(env, post_id, executor, _no_public_storage)
    assert ended == "published" and executor.calls[0].params == {"text": "Harvest Club opens Friday."}
    assert _targets(env, post_id)["blueskyish", "text"]["remote_id"] == "p-2"


def test_a_failed_generic_publish_leaves_the_channel_unverified(env):
    post_id = _approved_post(env, _generic_target("text"))
    executor = FakeExecutor({GENERIC_SLUG: [harness.refused("400 Bad Request")]})
    ended, _ = _publish(env, post_id, executor, _no_public_storage)
    assert ended == "failed"
    with env.factory() as db:
        assert "blueskyish" not in capabilities._published_toolkits(db, WS)


def test_the_youtube_adapter_reads_the_options_the_composer_writes():
    steps = capabilities.SEEDED_CHANNELS["youtube"].kinds["video"]
    upload, thumbnail = steps
    assert upload.params["categoryId"] == "$option.category_id" and upload.params["privacyStatus"] == "$option.privacy_status"
    assert upload.files == ("videoFilePath",) and thumbnail.urls == ("thumbnailUrl",) and thumbnail.optional
