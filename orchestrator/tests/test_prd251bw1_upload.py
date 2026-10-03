"""PRD-251B Wave 1, US-B109 — a post's visual uploaded: ``POST /api/socials/posts/{id}/media``.

On the S1.1c render harness (SQLite, storage faked, DeliverableService recorded). Pinned:

* a PNG and an MP4 within their limits are accepted: stored under
  ``social-media/{workspace}/{post}/upload-<sha>.<ext>``, registered as a Deliverable
  (``source_type`` upload), and the post's media is that file with ``template_id`` null, the
  format the file is and no chosen length (media is content: the hash moves);
* the type is the bytes', never the name's: a text file called .png, an HEIC still and a
  QuickTime movie are 415, and nothing is stored;
* a file over its limit is 413, and nothing is stored;
* a post that cannot change (rendering, archived) is 409 before anything is stored;
  another workspace's post is 404; no file is 422;
* an upload to an approved post voids the approval like any edit;
* the route is a plain ``def``, in the manifest with its method, apiClient posts the file
  to it, and both limits are in config.py and config-surface.json.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import sys
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

from fastapi.routing import APIRoute  # noqa: E402

import api.socials as socials_api  # noqa: E402
import api.socials_media_upload as upload_api  # noqa: E402
import tests.test_prd251w1_render_lifecycle as render_harness  # noqa: E402
from core.models.socials import SocialPost  # noqa: E402
from modules.socials import media_store, service  # noqa: E402
from tests.test_prd251w1_render_lifecycle import WS, WS_OTHER, Deliverables, FakeStore, _create, _ctx, _post  # noqa: E402

env = render_harness.env
ROUTE = "/api/socials/posts/{post_id}/media"
MANIFEST = _ORCH / "reports" / "route-manifest.json"
CONFIG_SURFACE = _ORCH / "reports" / "config-surface.json"
API_CLIENT = _ORCH.parent / "frontend" / "lib" / "api-client.ts"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"pixels " * 64
MP4 = b"\x00\x00\x00\x18ftypisom" + b"frames " * 200
HEIC = b"\x00\x00\x00\x18ftypheic" + b"still " * 40
QUICKTIME = b"\x00\x00\x00\x14ftypqt  " + b"movie " * 40


@pytest.fixture
def uploads(env, monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(media_store, "MediaStore", lambda *args, **kwargs: store)
    env.store = store
    return env


def _upload(env, post_id, data, name="visual.png", content_type="image/png"):
    return env.client.post(f"/api/socials/posts/{post_id}/media", files={"file": (name, data, content_type)})


def _limits(monkeypatch, image=None, video=None):
    if image is not None:
        monkeypatch.setattr(upload_api.config, "SOCIALS_UPLOAD_IMAGE_MAX_BYTES", image)
    if video is not None:
        monkeypatch.setattr(upload_api.config, "SOCIALS_UPLOAD_VIDEO_MAX_BYTES", video)


def test_a_png_becomes_the_posts_visual(uploads):
    post = _create(uploads, format="image", length_seconds=None)
    resp = _upload(uploads, post["id"], PNG)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    sha = hashlib.sha256(PNG).hexdigest()
    key = f"social-media/{WS}/{post['id']}/upload-{sha[:16]}.png"
    assert list(uploads.store.objects) == [key]
    assert uploads.store.objects[key] == (PNG, "image/png")
    (registered,) = Deliverables.calls
    assert registered["source_type"] == "upload" and registered["artifact_type"] == "image"
    assert registered["file_path"] == key and registered["file_size_bytes"] == len(PNG)
    assert registered["extra"]["sha256"] == sha and registered["extra"]["social_post_id"] == post["id"]
    assert body["media"] == {"original": ["d-video-1"]}
    assert body["template_id"] is None and body["format"] == "image" and body["length_seconds"] is None
    assert body["content_hash"] != post["content_hash"]
    row = _post(uploads, post["id"])
    assert row.content_hash == service.compute_content_hash(row)


def test_an_mp4_becomes_a_video_post(uploads):
    post = _create(uploads, format="image")
    resp = _upload(uploads, post["id"], MP4, name="clip.mp4", content_type="video/mp4")
    assert resp.status_code == 200, resp.text
    assert resp.json()["format"] == "video"
    (key,) = uploads.store.objects
    assert key.endswith(".mp4") and uploads.store.objects[key][1] == "video/mp4"
    assert Deliverables.calls[0]["artifact_type"] == "video"


@pytest.mark.parametrize(
    "data, name, declared",
    [
        (b"just some text, honestly", "visual.png", "image/png"),  # the name and the type lie
        (HEIC, "photo.mp4", "video/mp4"),  # an ftyp box, but a still image's brand
        (QUICKTIME, "clip.mp4", "video/mp4"),  # a QuickTime movie is not an MP4
    ],
)
def test_the_type_is_the_bytes_never_the_name(uploads, data, name, declared):
    post = _create(uploads, format="image")
    resp = _upload(uploads, post["id"], data, name=name, content_type=declared)
    assert resp.status_code == 415, resp.text
    assert uploads.store.objects == {} and Deliverables.calls == []
    assert _post(uploads, post["id"]).media == {}


def test_a_file_over_its_limit_is_refused_and_nothing_is_stored(uploads, monkeypatch):
    _limits(monkeypatch, image=len(PNG) - 1, video=len(MP4))
    post = _create(uploads, format="image")
    assert _upload(uploads, post["id"], PNG).status_code == 413
    assert uploads.store.objects == {} and Deliverables.calls == []
    # The video limit is the video's own: an MP4 at exactly its limit is fine.
    assert _upload(uploads, post["id"], MP4, name="clip.mp4", content_type="video/mp4").status_code == 200


def test_a_post_that_cannot_change_refuses_before_anything_is_stored(uploads):
    post = _create(uploads, format="image")
    row = _post(uploads, post["id"])
    row.status = service.RENDERING
    uploads.session.commit()
    assert _upload(uploads, post["id"], PNG).status_code == 409
    assert uploads.store.objects == {} and Deliverables.calls == []


def test_another_workspaces_post_is_404_and_no_file_is_422(uploads):
    post = _create(uploads, format="image")
    uploads.ctx = _ctx(WS_OTHER)
    assert _upload(uploads, post["id"], PNG).status_code == 404
    assert uploads.client.post(f"/api/socials/posts/{post['id']}/media").status_code == 404
    uploads.ctx = _ctx(WS)
    missing = uploads.client.post(f"/api/socials/posts/{post['id']}/media")
    assert missing.status_code == 422 and "Attach the file" in missing.text
    assert uploads.store.objects == {}


def test_an_upload_to_an_approved_post_voids_the_approval(uploads):
    post = _create(uploads, format="image")
    submitted = uploads.client.post(f"/api/socials/posts/{post['id']}/submit").json()
    approved = uploads.client.post(f"/api/socials/posts/{post['id']}/approve", json={"content_hash": submitted["content_hash"]})
    assert approved.status_code == 200 and approved.json()["status"] == "approved"
    resp = _upload(uploads, post["id"], PNG)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "needs_approval" and body["approved_hash"] != body["content_hash"]


def test_the_route_limits_and_client_are_wired():
    (route,) = [r for r in socials_api.router.routes if isinstance(r, APIRoute) and r.path == ROUTE and "POST" in r.methods]
    assert not inspect.iscoroutinefunction(route.endpoint), "a route over a sync Session is a plain def (F105)"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "POST", "path": ROUTE} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])
    surface = json.loads(CONFIG_SURFACE.read_text(encoding="utf-8"))["settings"]
    for name in ("SOCIALS_UPLOAD_IMAGE_MAX_BYTES", "SOCIALS_UPLOAD_VIDEO_MAX_BYTES"):
        assert name in surface
        assert isinstance(getattr(upload_api.config, name), int)
    client = API_CLIENT.read_text(encoding="utf-8")
    call = re.search(r"async uploadSocialPostMedia\(postId: string, file: File\)[\s\S]*?\{([\s\S]*?)\n  \}", client)
    assert call, "apiClient.uploadSocialPostMedia"
    assert "`/api/socials/posts/${postId}/media`" in call.group(1) and "method: 'POST'" in call.group(1)
    assert "FormData" in call.group(1)


def test_sniffing_reads_only_the_bytes():
    assert upload_api.sniff(PNG[:16]) is upload_api.PNG
    assert upload_api.sniff(b"\xff\xd8\xff\xe0" + b"\x00" * 12) is upload_api.JPEG
    assert upload_api.sniff(b"RIFF\x00\x00\x00\x00WEBPVP8 ") is upload_api.WEBP
    assert upload_api.sniff(MP4[:16]) is upload_api.MP4
    for other in (HEIC[:16], QUICKTIME[:16], b"GIF89a" + b"\x00" * 10, b""):
        assert upload_api.sniff(other) is None
    assert SocialPost  # the model the route writes
