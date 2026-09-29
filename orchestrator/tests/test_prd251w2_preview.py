"""PRD-251 Wave 2, US-208 (S2.2b) — the composer's preview render.

``POST /api/socials/posts/{id}/render`` with ``{"preview": true}`` on the S1.1c
render harness (SQLite, media-render mocked by an httpx transport, storage
faked). Pinned:

* a preview renders the first size at half resolution (1080×1920 → 540×960) and
  leaves the post's status, media and content hash as they were; it registers no
  Deliverable, and its files never share a name with the post's media;
* it is stored as the post's ``preview`` (done with its files, or failed with why),
  for the version it rendered;
* its seconds are held against the quota before media-render is called (429 and
  nothing changed when none are left) and booked on the media lane when it ends;
* a preview never voids an approval; one preview at a time; a rendering post is 409;
* the migration's column is nullable JSON outside the content hash.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.socials_preview as preview_api  # noqa: E402
import modules.socials.service as service  # noqa: E402
import tests.test_prd251w1_render_lifecycle as render_harness  # noqa: E402
from modules.socials import preview  # noqa: E402
from tests.test_prd251w1_render_lifecycle import (  # noqa: E402
    WS,
    Deliverables,
    FakeStore,
    Renderer,
    _post,
    _real_renderer_check,
    _renderable,
    _run,
    _start,
    _usage_row,
)

env = render_harness.env
PREVIEW_OUTPUT = {"name": "render.mp4", "aspect": "9:16", "width": 540, "height": 960, "duration": 39.5}


@pytest.fixture
def previews(env, monkeypatch):
    launched = []
    monkeypatch.setattr(preview_api, "_launch_preview", launched.append)
    env.previews = launched
    return env


def _preview(env, post_id):
    return env.client.post(f"/api/socials/posts/{post_id}/render", json={"preview": True})


def _run_preview(job, renderer, store, factory):
    """The harness's _run, through run_preview instead of run_render."""
    original = render_harness.render.run_render
    render_harness.render.run_render = preview.run_preview
    try:
        return _run(job, renderer, store, factory)
    finally:
        render_harness.render.run_render = original


def test_a_preview_starts_without_moving_the_post(previews):
    post = _renderable(previews)
    resp = _preview(previews, post["id"])
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "draft" and body["content_hash"] == post["content_hash"] and body["media"] == {}
    assert body["preview"]["status"] == "rendering" and body["preview"]["content_hash"] == post["content_hash"]
    (job,) = previews.previews
    assert job.preview is True and previews.launched == []  # not the approval render
    # Half resolution, the first size only; the same timing and audio.
    assert job.bundle["variables"]["size.width"] == 540 and job.bundle["variables"]["size.height"] == 960
    assert job.bundle["audio"] == render_harness.COMPOSITION["audio_plan"]


def test_a_finished_preview_is_the_posts_preview_and_nothing_else(previews):
    post = _renderable(previews)
    _preview(previews, post["id"])
    (job,) = previews.previews
    renderer, store = Renderer(outputs=(PREVIEW_OUTPUT,)), FakeStore()
    assert _run_preview(job, renderer, store, previews.factory) is True

    row = _post(previews, post["id"])
    assert row.status == "draft" and row.media == {} and row.content_hash == post["content_hash"]
    assert row.content_hash == service.compute_content_hash(row)
    assert row.preview["status"] == "done" and row.preview["content_hash"] == post["content_hash"]
    (file,) = row.preview["files"]
    assert file["name"] == "preview-video-9x16.mp4"
    assert file["url"] == f"/api/socials/posts/{post['id']}/media/preview-video-9x16.mp4"
    assert (file["width"], file["height"], file["duration"]) == (540, 960, 39.5)
    assert list(store.objects) == [f"social-media/{WS}/{post['id']}/preview-video-9x16.mp4"]
    assert Deliverables.calls == []  # a preview is no Deliverable
    # Its render seconds are booked on the media lane, as any render's.
    (booking,) = previews.booked
    assert booking["units"] == 39.5 and booking["scope"]["request_type"] == "media"


def test_a_failed_preview_says_why_and_the_post_is_untouched(previews):
    post = _renderable(previews)
    _preview(previews, post["id"])
    (job,) = previews.previews
    renderer = Renderer(status="failed", error={"code": "render_timed_out", "message": "too long"})
    assert _run_preview(job, renderer, FakeStore(), previews.factory) is False
    row = _post(previews, post["id"])
    assert row.status == "draft" and row.preview["status"] == "failed" and "too long" in row.preview["error"]
    assert previews.booked == []


def test_past_the_quota_a_preview_is_refused_before_media_render(previews, monkeypatch):
    post = _renderable(previews)
    calls = _real_renderer_check(monkeypatch)
    previews.session.add(_usage_row(WS, 600))  # Basic: 10 minutes, all used
    previews.session.commit()
    resp = _preview(previews, post["id"])
    assert resp.status_code == 429
    assert calls == [] and previews.previews == []
    assert _post(previews, post["id"]).preview is None


def test_a_preview_never_voids_an_approval(previews):
    post = _renderable(previews)
    _, job = _start(previews, post)
    assert _run(job, Renderer(), FakeStore(), previews.factory) is True
    rendered = _post(previews, post["id"])
    approved = previews.client.post(
        f"/api/socials/posts/{post['id']}/approve", json={"content_hash": rendered.content_hash}
    )
    assert approved.status_code == 200, approved.text

    resp = _preview(previews, post["id"])
    assert resp.status_code == 202
    (job,) = previews.previews
    assert _run_preview(job, Renderer(outputs=(PREVIEW_OUTPUT,)), FakeStore(), previews.factory) is True
    row = _post(previews, post["id"])
    assert row.status == "approved" and row.approved_hash == row.content_hash == rendered.content_hash
    assert row.media == rendered.media
    service.assert_publishable(row)


def test_one_preview_at_a_time_and_never_while_rendering(previews):
    post = _renderable(previews)
    assert _preview(previews, post["id"]).status_code == 202
    second = _preview(previews, post["id"])
    assert second.status_code == 409 and "rendering already" in second.json()["detail"]

    other = _renderable(previews)
    _start(previews, other)
    assert _preview(previews, other["id"]).status_code == 409


def test_a_plain_render_is_still_the_approval_render(previews):
    post = _renderable(previews)
    resp = previews.client.post(f"/api/socials/posts/{post['id']}/render", json={"preview": False})
    assert resp.status_code == 202 and resp.json()["status"] == "rendering"
    assert previews.previews == [] and len(previews.launched) == 1


def test_another_workspaces_post_cannot_be_previewed(previews):
    post = _renderable(previews)
    previews.ctx = render_harness._ctx(render_harness.WS_OTHER)
    assert _preview(previews, post["id"]).status_code == 404


@pytest.mark.parametrize("size, half", [("1080x1920", "540x960"), ("1080x1350", "540x674"), ("1920x1080", "960x540")])
def test_half_size_keeps_even_sides(size, half):
    assert preview.half_size(size) == half


def test_the_preview_is_outside_the_content_hash():
    post = service.create_draft(type("S", (), {"add": lambda self, o: None})(), workspace_id=uuid.uuid4(),
                                created_by="u", title="T", copy={"base": "x"})
    before = service.compute_content_hash(post)
    post.preview = {"status": "done", "files": [{"name": "preview-video-9x16.mp4"}]}
    assert service.compute_content_hash(post) == before
