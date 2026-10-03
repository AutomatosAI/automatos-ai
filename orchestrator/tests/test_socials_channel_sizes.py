"""Each channel its own shape (3 Oct 2026, Gerard: "different socials have different styles
and sizes ... we need multiple for the different formats").

* ``channel_sizes``: a channel's best shape (X 16:9, Instagram and LinkedIn images 1:1,
  carousels 4:5, reels, stories, shorts and TikTok 9:16); of a template's sizes, the one
  closest in shape; the sizes a post renders (each channel's closest, once, in channel
  order; the template's default with no channel).
* Publishing gives each channel only its own size's files; a carousel's slides stay
  together; with one size, every file as before.
* The render makes every size, the footage and the voice ONCE for all of them (each is
  paid), and stores files only once every size rendered (a later size that fails leaves
  nothing behind); the files of every size join ``media`` by aspect.
* ``POST /render`` hands a still post's job one bundle per size its channels need, and a
  video's one bundle, its default size, as before (a video's minutes, quota hold and wait
  are one render's).
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
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

import tests.test_prd251w1_render_lifecycle as lifecycle  # noqa: E402
from core.models.socials import SocialPostTarget  # noqa: E402
from modules.socials import channel_sizes, render  # noqa: E402
from modules.socials.media_urls import MediaFile  # noqa: E402
from modules.socials.publish_sources import media_for  # noqa: E402

env = lifecycle.env
TEXT_CARD = ["1080x1350", "1080x1920", "1200x628", "1600x900"]
PHOTO_CARD = ["1080x1350", "1080x1920", "1080x1080", "1200x628"]


def _target(toolkit, kind):
    return SimpleNamespace(toolkit=toolkit, post_kind=kind)


# ---------------------------------------------------------------------------
# channel_sizes
# ---------------------------------------------------------------------------


def test_each_channel_has_its_shape_and_a_text_post_none():
    assert channel_sizes.aspect_of("twitter", "image") == "16:9"
    assert channel_sizes.aspect_of("instagram", "reel") == "9:16"
    assert channel_sizes.aspect_of("mastodon", "image") == "1:1"  # a kind's default
    assert channel_sizes.aspect_of("twitter", "text") is None
    assert channel_sizes.ratio("4:5") == channel_sizes.ratio("1080x1350") == 0.8
    assert channel_sizes.ratio("x") is None and channel_sizes.ratio("0:5") is None


@pytest.mark.parametrize("sizes, aspect, expected", [
    (TEXT_CARD, "1:1", "1080x1350"),
    (TEXT_CARD, "16:9", "1600x900"),
    (TEXT_CARD, "9:16", "1080x1920"),
    (PHOTO_CARD, "1:1", "1080x1080"),
    (PHOTO_CARD, "16:9", "1200x628"),
    ([], "1:1", None),
])
def test_the_closest_size_in_shape(sizes, aspect, expected):
    assert channel_sizes.closest(sizes, aspect) == expected


def test_a_post_renders_each_channels_size_once_else_the_default():
    channels = [_target("instagram", "image"), _target("twitter", "image"), _target("linkedin", "image")]
    assert channel_sizes.render_sizes(TEXT_CARD, channels) == ["1080x1350", "1600x900"]
    assert channel_sizes.render_sizes(PHOTO_CARD, channels) == ["1080x1080", "1200x628"]
    assert channel_sizes.render_sizes(TEXT_CARD, [_target("instagram", "reel")]) == ["1080x1920"]
    assert channel_sizes.render_sizes(TEXT_CARD, []) == ["1080x1350"]
    assert channel_sizes.render_sizes(TEXT_CARD, [_target("twitter", "text")]) == ["1080x1350"]
    # two sizes of one shape would write one file name: the first is kept
    assert channel_sizes.render_sizes(["1080x1350", "1920x1080", "3840x2160"], channels) == ["1080x1350", "1920x1080"]


# ---------------------------------------------------------------------------
# Publishing: each channel its own files
# ---------------------------------------------------------------------------


def _file(aspect, name, content_type="image/png"):
    return MediaFile(aspect=aspect, deliverable_id=name, name=name, key=f"k/{name}", content_type=content_type)


def test_each_channel_publishes_its_own_size_and_a_carousel_keeps_its_slides():
    files = [_file("4:5", "image-4x5.png"), _file("16:9", "image-16x9.png")]
    assert [f.name for f in media_for(files, "twitter", "image")] == ["image-16x9.png"]
    assert [f.name for f in media_for(files, "instagram", "image")] == ["image-4x5.png"]
    slides = [_file("4:5", f"slide-{n}-4x5.png") for n in (1, 2)] + [_file("9:16", f"slide-{n}-9x16.png") for n in (1, 2)]
    assert [f.name for f in media_for(slides, "instagram", "carousel")] == ["slide-1-4x5.png", "slide-2-4x5.png"]
    one_size = [_file("4:5", "image-4x5.png"), _file("4:5", "image-4x5-2.png")]
    assert media_for(one_size, "twitter", "image") == tuple(one_size)  # one size: every file, as before
    assert media_for(files, "twitter", "text") == ()


# ---------------------------------------------------------------------------
# The render: every size, footage and voice once
# ---------------------------------------------------------------------------


def _bundle(width, height):
    html = '<div><img data-slot="photo" src="assets/slots/photo.png"></div>'
    return {"composition": {"html": html}, "variables": {"size.width": width, "size.height": height}}


def test_every_size_is_rendered_with_the_footage_and_voice_made_once(monkeypatch):
    made = {"footage": 0, "voice": 0}
    submitted, stored = [], []

    async def footage_links(job, store, factory):
        made["footage"] += 1
        return {"assets/slots/photo.png": "https://storage.test/photo.png"}

    async def voice_links(job, bundle, store, factory):
        made["voice"] += 1
        return {}

    async def submit(client, bundle, deadline):
        submitted.append(bundle)
        return {"id": f"job-{len(submitted)}"}

    async def wait(client, job, accepted, deadline):
        return {"id": accepted["id"], "report": {"check": {"ok": True}}}

    async def store_outputs(client, store, factory, job, finished, music):
        stored.append(finished["id"])
        return {"4:5": [{"name": "image-1080.png"}]} if finished["id"] == "job-1" else {"16:9": [{"name": "image-1600.png"}]}

    for name, fake in (("_footage_links", footage_links), ("_voice_links", voice_links), ("_submit", submit),
                       ("_wait", wait), ("_store_outputs", store_outputs)):
        monkeypatch.setattr(render, name, fake)
    monkeypatch.setattr(render, "_music_of", lambda job, finished: None)
    job = render.RenderJob(
        post_id=uuid.uuid4(), workspace_id=uuid.uuid4(), actor="owner-1", content_hash="h", title="t", format="image",
        bundle=_bundle(1080, 1350), extra_bundles=(_bundle(1600, 900),),
    )

    first, music, media = asyncio.run(render._render_sizes(job, None, None, None, 0.0))

    assert made == {"footage": 1, "voice": 1}
    assert [b["variables"]["size.width"] for b in submitted] == [1080, 1600]
    assert all(b["media"] == [{"path": "assets/slots/photo.png", "url": "https://storage.test/photo.png"}] for b in submitted)
    assert media == {"4:5": [{"name": "image-1080.png"}], "16:9": [{"name": "image-1600.png"}]}
    assert first["id"] == "job-1" and music is None and stored == ["job-1", "job-2"]


def test_a_later_size_that_fails_leaves_nothing_of_an_earlier_one(monkeypatch):
    stored = []

    async def no_links(*args):
        return {}

    async def submit(client, bundle, deadline):
        return {"id": f"job-{bundle['variables']['size.width']}"}

    async def wait(client, job, accepted, deadline):
        if accepted["id"] == "job-1600":
            raise render.RenderFailure("timed_out", "The render did not finish.")
        return {"id": accepted["id"]}

    async def store_outputs(*args):
        stored.append(args)
        return {}

    for name, fake in (("_footage_links", no_links), ("_voice_links", no_links), ("_submit", submit),
                       ("_wait", wait), ("_store_outputs", store_outputs)):
        monkeypatch.setattr(render, name, fake)
    job = render.RenderJob(
        post_id=uuid.uuid4(), workspace_id=uuid.uuid4(), actor="owner-1", content_hash="h", title="t", format="image",
        bundle=_bundle(1080, 1350), extra_bundles=(_bundle(1600, 900),),
    )
    with pytest.raises(render.RenderFailure):
        asyncio.run(render._render_sizes(job, None, None, None, 0.0))
    assert stored == []  # the 1080x1350 file was never stored or registered


# ---------------------------------------------------------------------------
# POST /render: one bundle per size
# ---------------------------------------------------------------------------


def _targets(env, post, channels):
    for toolkit, kind in channels:
        env.session.add(SocialPostTarget(
            post_id=uuid.UUID(post["id"]), toolkit=toolkit, post_kind=kind, action_plan={},
            idempotency_key=f"{post['id']}:{toolkit}", status="pending",
        ))
    env.session.commit()


def _sizes(job):
    return [(b["variables"]["size.width"], b["variables"]["size.height"]) for b in (job.bundle, *job.extra_bundles)]


def _title_card():
    from modules.documents.social_starters import social_starters

    (title_card,) = [s for s in social_starters("social_image") if s["slug"] == "title-card"]
    return title_card["blocks"]


def test_a_still_post_hands_the_job_one_bundle_per_channel_size(env):
    template = lifecycle._template(env, blocks=_title_card(), fmt="social_image")
    post = lifecycle._create(env, template_id=str(template), format="image",
                             variables={"headline": {"value": "MEET|AUTO.", "claim": False}})
    _targets(env, post, [("instagram", "image"), ("twitter", "image"), ("linkedin", "image")])

    _, job = lifecycle._start(env, post)

    assert _sizes(job) == [(1080, 1350), (1600, 900)]


def test_a_video_renders_one_size_as_before(env):
    blocks = {**lifecycle.COMPOSITION, "sizes": ["1080x1920", "1920x1080"]}
    post = lifecycle._create(env, template_id=str(lifecycle._template(env, blocks=blocks)))
    _targets(env, post, [("instagram", "reel"), ("youtube", "video")])

    _, job = lifecycle._start(env, post)

    assert _sizes(job) == [(1080, 1920)]


def test_a_post_with_no_channel_renders_the_default_size_alone(env):
    blocks = {**lifecycle.COMPOSITION, "sizes": ["1080x1920", "1920x1080"]}
    post = lifecycle._create(env, template_id=str(lifecycle._template(env, blocks=blocks)))
    _, job = lifecycle._start(env, post)
    assert (job.bundle["variables"]["size.width"], job.bundle["variables"]["size.height"]) == (1080, 1920)
    assert job.extra_bundles == ()
