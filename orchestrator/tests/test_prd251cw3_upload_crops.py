"""PRD-251C Wave 3, US-C303 (O6, the PRD's recommendation: crop) — a still of your own, cropped
for each channel.

Pinned:

* **The composition** is a valid still template: the picture fills the frame, centred, in the
  one slot only the person's own file fills, and it has nothing else on it.
* **Which posts.** A post's own still (an upload or a Library picture as the whole post, no
  template, an image) is cropped; a video of its own, a template's post or a post with no
  file of its own is not.
* **Files per shape.** One render size per shape its channels need (LinkedIn's square,
  Instagram's story at 9:16, X's 16:9), one square with no channel.
* **The render.** ``POST /posts/{id}/render`` on such a post starts the crop render: its
  bundles, the picture linked from storage into the slot, the original kept beside the
  crops when the render ends; a picture no longer in Deliverables is a 422 and nothing
  starts. The plan maker renders a Library still instead of sending it as it is.
* **Publishing gives each channel its own** crop (``publish_sources.media_for``).
"""
from __future__ import annotations

import asyncio
import functools
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import anyio

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.socials_render_crops as socials_render_crops  # noqa: E402
import api.socials_targets as socials_targets  # noqa: E402
import tests.test_prd251w1_render_lifecycle as lifecycle  # noqa: E402
from core.models.socials import SocialPost  # noqa: E402
from core.social_templates import SOCIAL_IMAGE, validate_social_blocks  # noqa: E402
from modules.socials import publish_sources, render, service, upload_crops  # noqa: E402
from modules.socials.media_urls import MediaFile  # noqa: E402
from services import socials_plan_maker as maker_mod  # noqa: E402
from tests.test_prd251_api import WS_A  # noqa: E402
from tests.test_prd251bw3_plan_visuals import TOPIC, _db, _plan, _PostsApi, _slot  # noqa: E402

env = lifecycle.env
NOW = datetime(2026, 10, 14, 7, 30, tzinfo=timezone.utc)
PICTURE = str(uuid.uuid4())
KEY = "social-media/ws/post/upload-0123456789abcdef.png"


def _target(toolkit, kind):
    return SimpleNamespace(toolkit=toolkit, post_kind=kind)


def _own(**over):
    fields = {"id": uuid.uuid4(), "workspace_id": uuid.uuid4(), "template_id": None, "format": "image",
              "media": {"original": [PICTURE]}, "targets": []}
    return SimpleNamespace(**{**fields, **over})


# ── the composition and which posts ────────────────────────────────────────


def test_the_crop_is_a_valid_still_with_only_the_picture():
    checked = validate_social_blocks(dict(upload_crops.CROP_BLOCKS), SOCIAL_IMAGE)
    assert checked["slots"] == {"photo": {"kind": "image", "path": upload_crops.CROP_PATH, "label": "Your picture", "generate": False}}
    html = upload_crops.CROP_HTML
    assert "object-fit: cover" in html and "object-position: center" in html and 'data-slot="photo"' in html
    assert "{{ brand." not in html and checked["variables_schema"] == {}  # no logo, no words


def test_only_a_still_of_the_posts_own_is_cropped():
    assert upload_crops.own_still(_own()) == PICTURE
    assert upload_crops.own_still(_own(format="video")) is None  # a video of your own goes as it is
    assert upload_crops.own_still(_own(template_id=uuid.uuid4())) is None  # a template's post renders its template
    assert upload_crops.own_still(_own(media={})) is None
    assert upload_crops.own_still(_own(media={"original": [PICTURE], "1:1": [{"deliverable_id": "d-2"}]})) == PICTURE  # cropped before


def test_one_size_per_shape_the_channels_need():
    channels = [_target("linkedin", "image"), _target("instagram", "story"), _target("twitter", "image"), _target("instagram", "image")]
    assert upload_crops.crop_sizes(_own(targets=channels)) == ["1080x1080", "1080x1920", "1600x900"]
    assert upload_crops.crop_sizes(_own()) == ["1080x1080"]
    bundles = upload_crops.crop_bundles(_own(targets=channels))
    assert [(b["variables"]["size.width"], b["variables"]["size.height"]) for b in bundles] == [(1080, 1080), (1080, 1920), (1600, 900)]
    assert all('data-slot="photo"' in b["composition"]["html"] and b["still"] == {"at": [0.0]} for b in bundles)


# ── the render ─────────────────────────────────────────────────────────────


def _own_post(env, monkeypatch, files):
    post = lifecycle._create(env, format="image")
    row = lifecycle._post(env, post["id"])
    row.media, row.template_id = {"original": [PICTURE]}, None
    row.content_hash = service.compute_content_hash(row)
    env.session.commit()
    monkeypatch.setattr(socials_render_crops.media_urls, "resolve_post_media", lambda db, p: list(files))
    return {**post, "id": str(row.id)}


def test_rendering_a_still_of_your_own_starts_its_crops(env, monkeypatch):
    original = MediaFile(aspect="original", deliverable_id=PICTURE, name="upload-0123456789abcdef.png", key=KEY,
                         content_type="image/png", bytes=10)
    post = _own_post(env, monkeypatch, [original])
    resp = env.client.post(f"/api/socials/posts/{post['id']}/render")
    assert resp.status_code == 202, resp.text
    (job,) = env.launched
    assert (job.slot_keys, job.keep_media, job.extra_bundles) == ({upload_crops.CROP_PATH: KEY}, ("original",), ())
    assert 'data-slot="photo"' in job.bundle["composition"]["html"] and job.bundle["still"] == {"at": [0.0]}
    assert lifecycle._post(env, post["id"]).status == "rendering"


def test_a_picture_gone_from_deliverables_is_422_and_nothing_starts(env, monkeypatch):
    post = _own_post(env, monkeypatch, [])
    resp = env.client.post(f"/api/socials/posts/{post['id']}/render")
    assert resp.status_code == 422 and "no longer in the workspace's Deliverables" in resp.json()["detail"]
    assert env.launched == [] and lifecycle._post(env, post["id"]).status == "draft"


class _LinkingStore:
    def presigned_get(self, key, ttl):
        return f"https://storage.test/{key}"


def test_the_picture_is_linked_into_each_size_and_the_original_is_kept(monkeypatch):
    post = _own(targets=[_target("linkedin", "image"), _target("twitter", "image")])
    bundle, other = upload_crops.crop_bundles(post)
    job = render.RenderJob(post_id=post.id, workspace_id=post.workspace_id, actor="u", content_hash="h", title="Mine", format="image",
                           bundle=bundle, extra_bundles=(other,), slot_keys={upload_crops.CROP_PATH: KEY}, keep_media=("original",))
    dressed = asyncio.run(render._dressed(job, _LinkingStore(), None))
    assert [b["media"] for b in dressed] == [[{"path": upload_crops.CROP_PATH, "url": f"https://storage.test/{KEY}"}]] * 2
    row = SocialPost(id=uuid.uuid4(), workspace_id=WS_A, created_by="u", title="Mine", status="rendering", format="image",
                     media={"original": [PICTURE]}, copy={"base": "Ours."}, content_hash="0" * 64)
    crop = {"deliverable_id": "d-1x1", "name": "image-1x1.png", "sha256": "a" * 64, "bytes": 5}
    service.finish_render(row, "u", {"1:1": [crop]}, keep=("original",))
    assert row.media == {"original": [PICTURE], "1:1": [crop]} and row.status == "needs_approval"


def test_publishing_gives_each_channel_its_own_crop():
    def still(aspect, name):
        return MediaFile(aspect=aspect, deliverable_id=name, name=f"{name}.png", key=f"k/{name}.png", content_type="image/png", bytes=5)

    files = [still("original", "upload"), still("1:1", "square"), still("9:16", "story"), still("16:9", "wide")]
    assert [f.name for f in publish_sources.media_for(files, "linkedin", "image")] == ["square.png"]
    assert [f.name for f in publish_sources.media_for(files, "instagram", "story")] == ["story.png"]
    assert [f.name for f in publish_sources.media_for(files, "twitter", "image")] == ["wide.png"]
    assert [f.name for f in publish_sources.media_for(files[:1], "linkedin", "image")] == ["upload.png"]  # not cropped yet: as it is


def test_the_plan_maker_crops_a_library_still_instead_of_sending_it_as_it_is(monkeypatch):
    posts_api = _PostsApi()  # applies the template, format and media as service.update_post does
    monkeypatch.setattr(maker_mod, "_posts_api", lambda: posts_api)
    monkeypatch.setattr(maker_mod, "_propose", lambda db, plan, slot, topic, now, visual_slots=(): {
        "title": "Lisbon", "copy": {"base": "See you there."}, "variables": {}, "sources": {}, "template_id": None, "visual_prompts": {}})
    monkeypatch.setattr(maker_mod, "slot_targets", lambda db, ws, slot: [])
    monkeypatch.setattr(socials_targets, "set_post_targets", lambda db, post, actor, targets, agent=None: None)
    post = SimpleNamespace(id=uuid.uuid4(), template_id=None, format="image", media={})
    db = _db([{"id": "d-9", "title": "Lisbon stand", "summary": ""}])
    run = functools.partial(maker_mod.write, db, SimpleNamespace(id=WS_A), _plan({"library": 100}), _slot(fmt="image"), TOPIC, post, NOW)
    anyio.run(functools.partial(anyio.to_thread.run_sync, run))
    assert post.media == {"original": ["d-9"]} and posts_api.renders == [post.id] and posts_api.submits == []
