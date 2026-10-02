"""PRD-251 Wave 3, US-303 (S3.3c) — Instagram: image, Reel and carousel.

With the executor mocked (the US-301 harness) and the real stager (a fake object
store, a fake media-render): each kind makes its container from the FILE
(``INSTAGRAM_POST_IG_USER_MEDIA`` / ``INSTAGRAM_CREATE_CAROUSEL_CONTAINER``, the
adapter's own upload spec), waits until Instagram has processed it
(``INSTAGRAM_GET_IG_MEDIA`` status_code FINISHED), publishes it
(``INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH`` with the container id) and reads the
permalink. A container that never finishes within ``SOCIALS_PUBLISH_MAX_WAIT_SECONDS``
fails its target and publishes nothing.

Instagram takes JPEG images only, and the stills render as PNG: a still goes through
media-render's ``POST /jpeg`` (it has ffmpeg; the orchestrator adds no image library)
and reaches Instagram as a ``.jpg``. When media-render cannot convert it, the target
fails saying so. The deprecated actions are never in a sequence.
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

import httpx
import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251w3_publisher as harness  # noqa: E402
from config import config  # noqa: E402
from core import media_render_client  # noqa: E402
from core.media_render_client import MediaRenderClient, MediaRenderUnavailable  # noqa: E402
from modules.socials.capabilities import SEEDED_CHANNELS  # noqa: E402
from modules.socials.media_urls import MediaFile  # noqa: E402
from modules.socials.publish_steps import Runtime, Stager  # noqa: E402
from modules.socials.publishing import run_publish  # noqa: E402
from tests.test_prd251w3_publisher import FakeExecutor, _approved_post, _claim, _target, _targets, ok  # noqa: E402

env = harness.env

ACCOUNT, CONTAINER, MEDIA = "17841400000000001", "17900000000000002", "17900000000000003"
LINK = "https://www.instagram.com/p/C0ffee/"
PNG, JPEG = b"\x89PNG-still", b"\xff\xd8\xff-jpeg"


def _still(n=1):
    return MediaFile(aspect="4:5", deliverable_id=str(uuid.uuid4()), name=f"carousel-4x5-0{n}.png", key=f"k/carousel-4x5-0{n}.png",
                     content_type="image/png", bytes=len(PNG))


REEL = MediaFile(aspect="9:16", deliverable_id=str(uuid.uuid4()), name="video-9x16.mp4", key="k/video-9x16.mp4", content_type="video/mp4", bytes=4)
INSTAGRAM = {
    "INSTAGRAM_GET_USER_INFO": [ok({"id": ACCOUNT, "username": "harvestclub"})],
    "INSTAGRAM_POST_IG_USER_MEDIA": [ok({"id": CONTAINER})],
    "INSTAGRAM_CREATE_CAROUSEL_CONTAINER": [ok({"id": CONTAINER})],
    "INSTAGRAM_GET_IG_MEDIA": [
        ok({"id": CONTAINER, "status_code": "IN_PROGRESS"}),
        ok({"id": CONTAINER, "status_code": "FINISHED"}),
        ok({"id": MEDIA, "permalink": LINK}),
    ],
    "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH": [ok({"id": MEDIA})],
}


class ReadingExecutor(FakeExecutor):
    """Also keeps each staged file's bytes, read when the call is made."""

    async def execute_with_uploads(self, action, params, **kwargs):
        self.read = getattr(self, "read", {})
        for name, value in params.items():
            if isinstance(value, Path):
                self.read[(action, name)] = value.read_bytes()
        return await super().execute_with_uploads(action, params, **kwargs)


class FakeStore:
    """The documents bucket: download_file writes the object's bytes."""

    def __init__(self):
        self.keys = []

    def download_file(self, bucket, key, filename):
        self.keys.append(key)
        Path(filename).write_bytes(REEL_BYTES if key.endswith(".mp4") else PNG)


REEL_BYTES = b"mp4!"


def _run(env, post_id, executor, converter):
    job = _claim(env, post_id)
    converted = []

    async def convert(image):
        converted.append(image)
        return await converter(image)

    def build(executor, stager):
        async def sleep(seconds):
            env.sleeps.append(seconds)
            env.clock[0] += seconds

        real = Stager(stager.workdir, client=FakeStore(), converter=convert)
        return Runtime(executor=executor, stager=real, sleep=sleep, clock=lambda: env.clock[0])

    ended = asyncio.run(run_publish(job, executor=executor, session_factory=env.factory, runtime=build))
    return ended, converted


async def to_jpeg(image):
    return JPEG


def _receipt(env, post_id, kind):
    target = _targets(env, post_id)["instagram", kind]
    return target["status"], target["remote_id"], target["permalink"]


def test_an_instagram_image_is_a_jpeg_container_waited_on_then_published(env, monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_PUBLISH_POLL_SECONDS", 4)
    env.media = [_still()]
    post_id = _approved_post(env, _target("instagram", "image"))
    executor = ReadingExecutor(INSTAGRAM)

    ended, converted = _run(env, post_id, executor, to_jpeg)

    assert ended == "published"
    assert executor.actions == [
        "INSTAGRAM_GET_USER_INFO", "INSTAGRAM_POST_IG_USER_MEDIA", "INSTAGRAM_GET_IG_MEDIA", "INSTAGRAM_GET_IG_MEDIA",
        "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH", "INSTAGRAM_GET_IG_MEDIA",
    ]
    container = executor.calls[1]
    assert container.upload_params == ("image_file",)
    assert container.params["image_file"].suffix == ".jpg"
    assert executor.read["INSTAGRAM_POST_IG_USER_MEDIA", "image_file"] == JPEG
    assert container.params["ig_user_id"] == ACCOUNT and container.params["caption"] == "Harvest Club opens Friday."
    assert converted == [PNG]  # the PNG still went to media-render
    assert executor.calls[2].params == {"ig_media_id": CONTAINER, "fields": "status_code,status"}
    assert executor.calls[4].params == {"ig_user_id": ACCOUNT, "creation_id": CONTAINER}
    assert env.sleeps == [4]
    assert _receipt(env, post_id, "image") == ("published", MEDIA, LINK)


def test_a_jpeg_still_is_not_converted_again(env):
    env.media = [MediaFile(aspect="1:1", deliverable_id=str(uuid.uuid4()), name="photo.jpg", key="k/photo.jpg", content_type="image/jpeg", bytes=3)]
    post_id = _approved_post(env, _target("instagram", "image"))
    ended, converted = _run(env, post_id, FakeExecutor(INSTAGRAM), to_jpeg)
    assert ended == "published" and converted == []


def test_a_reel_uploads_its_video_file_unconverted(env):
    env.media = [REEL, _still()]
    post_id = _approved_post(env, _target("instagram", "reel"))
    executor = FakeExecutor(INSTAGRAM)

    ended, converted = _run(env, post_id, executor, to_jpeg)

    assert ended == "published" and converted == []
    container = executor.calls[1]
    assert container.action == "INSTAGRAM_POST_IG_USER_MEDIA" and container.upload_params == ("video_file",)
    assert container.params["media_type"] == "REELS" and container.params["video_file"].name == "video-9x16.mp4"
    assert _receipt(env, post_id, "reel") == ("published", MEDIA, LINK)


def test_a_three_slide_carousel_sends_three_jpegs_to_one_carousel_container(env):
    env.media = [_still(1), _still(2), _still(3)]
    post_id = _approved_post(env, _target("instagram", "carousel"))
    executor = FakeExecutor(INSTAGRAM)

    ended, converted = _run(env, post_id, executor, to_jpeg)

    assert ended == "published" and len(converted) == 3
    container = executor.calls[1]
    assert container.action == "INSTAGRAM_CREATE_CAROUSEL_CONTAINER" and container.upload_params == ("child_image_files",)
    files = container.params["child_image_files"]
    assert [f.name for f in files] == ["carousel-4x5-01.jpg", "carousel-4x5-02.jpg", "carousel-4x5-03.jpg"]
    assert executor.calls[4].params["creation_id"] == CONTAINER
    assert _receipt(env, post_id, "carousel") == ("published", MEDIA, LINK)


def test_a_container_that_never_finishes_fails_clearly_and_publishes_nothing(env, monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_PUBLISH_MAX_WAIT_SECONDS", 20)
    monkeypatch.setattr(config, "SOCIALS_PUBLISH_POLL_SECONDS", 5)
    env.media = [REEL]
    post_id = _approved_post(env, _target("instagram", "reel"))
    executor = FakeExecutor({**INSTAGRAM, "INSTAGRAM_GET_IG_MEDIA": [ok({"id": CONTAINER, "status_code": "IN_PROGRESS"})]})

    ended, _ = _run(env, post_id, executor, to_jpeg)

    assert ended == "failed"
    assert "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH" not in executor.actions
    assert executor.actions.count("INSTAGRAM_GET_IG_MEDIA") == 5  # t = 0, 5, 10, 15, 20
    error = _targets(env, post_id)["instagram", "reel"]["error"]
    assert "INSTAGRAM_GET_IG_MEDIA did not finish within 20 seconds" in error


def test_a_container_instagram_rejects_fails_with_its_status(env):
    env.media = [REEL]
    post_id = _approved_post(env, _target("instagram", "reel"))
    rejected = ok({"id": CONTAINER, "status_code": "ERROR", "status": "Error: the video's aspect ratio is not supported"})
    executor = FakeExecutor({**INSTAGRAM, "INSTAGRAM_GET_IG_MEDIA": [rejected]})

    ended, _ = _run(env, post_id, executor, to_jpeg)

    assert ended == "failed" and "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH" not in executor.actions
    assert "aspect ratio is not supported" in _targets(env, post_id)["instagram", "reel"]["error"]


def test_a_png_media_render_cannot_convert_fails_the_target_saying_so(env):
    env.media = [_still()]
    post_id = _approved_post(env, _target("instagram", "image"))
    executor = FakeExecutor(INSTAGRAM)

    async def unreachable(image):
        raise MediaRenderUnavailable("not_configured", "no media-render service is configured (SOCIALS_RENDER_URL)")

    ended, _ = _run(env, post_id, executor, unreachable)

    assert ended == "failed"
    assert "INSTAGRAM_POST_IG_USER_MEDIA" not in executor.actions  # nothing was sent as a PNG
    error = _targets(env, post_id)["instagram", "image"]["error"]
    assert "must reach the platform as a JPEG" in error and "SOCIALS_RENDER_URL" in error


@pytest.mark.parametrize("kind", ["image", "reel", "carousel"])
def test_no_deprecated_instagram_action_is_in_a_sequence(kind):
    actions = {step.action for step in SEEDED_CHANNELS["instagram"].kinds[kind]}
    assert not actions & {"INSTAGRAM_CREATE_MEDIA_CONTAINER", "INSTAGRAM_CREATE_POST", "INSTAGRAM_GET_POST_STATUS"}


def test_the_client_posts_the_image_to_media_render_s_jpeg_route(monkeypatch):
    seen = []

    def answer(request):
        seen.append((request.method, request.url.path, request.content))
        if request.content == b"bad":
            return httpx.Response(400, json={"error": "not_an_image", "message": "ffmpeg could not convert the image"})
        return httpx.Response(200, content=JPEG, headers={"Content-Type": "image/jpeg"})

    monkeypatch.setattr(config, "SOCIALS_RENDER_URL", "http://media-render:8090")
    client = MediaRenderClient(http=httpx.AsyncClient(transport=httpx.MockTransport(answer)))
    assert asyncio.run(client.to_jpeg(PNG)) == JPEG
    assert seen == [("POST", "/jpeg", PNG)]
    with pytest.raises(media_render_client.MediaRenderError):
        asyncio.run(client.to_jpeg(b"bad"))


def test_a_jpeg_param_must_be_one_of_the_step_s_files():
    from modules.socials.capabilities import parse_channel_adapters

    data = {"instagram": {"label": "I", "kinds": {"image": [
        {"id": "post", "action": "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH", "class": "publish", "params": {"caption": "$copy"},
         "jpeg": ["caption"], "returns": {"id": "id"}},
    ]}}}
    with pytest.raises(ValueError, match="jpeg param"):
        parse_channel_adapters(data)
