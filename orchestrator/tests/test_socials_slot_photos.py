"""PRD-251B (3 Oct 2026 pass) — the person's own picture in a template's photo spot.

Upload and Library used to replace the template. With a template that shows a photo, the
picture now fills its photo spot and the template stays, so its words sit over the picture:

* an upload with ``slot`` is stored under the post's prefix, registered as a Deliverable and
  recorded as the slot's file (done, from ``upload``); the template, the media and the
  content hash stay; a video, a spot the template does not have and a post with no template
  are refused before anything is stored;
* a Library picture (``PUT /posts/{id}/photos/{slot}``) is copied under the post's prefix and
  recorded the same way (from ``library``); another workspace's file is 404, one that is not
  a stored picture 422;
* the next render keeps the slot's file and makes nothing for it, and an editor save that
  sends the slot's prompt back keeps the record;
* no AI made the person's picture: a channel's AI label stays off for it.
"""
from __future__ import annotations

import hashlib
import os
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

import tests.test_prd251w1_render_lifecycle as render_harness  # noqa: E402
from core.models.core import DocumentTemplate  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402
from modules.socials import media_store, media_urls, render, service, slot_photos  # noqa: E402
from tests.test_prd251w1_render_lifecycle import WS, Deliverables, FakeStore, _create, _post, _template  # noqa: E402

env = render_harness.env
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"pixels " * 64
MP4 = b"\x00\x00\x00\x18ftypisom" + b"frames " * 200
PHOTO_CARD = next(s for s in social_starters() if s["slug"] == "photo-headline")["blocks"]
LIBRARY_ID = "7a3f0c2e-1b4d-4e8f-9a6b-2c5d8e1f0a3b"


class Store(FakeStore):
    """The documents bucket in memory, with the copy a Library picture takes."""

    def copy(self, source_key, key):
        self.objects[key] = self.objects[source_key]


@pytest.fixture
def photos(env, monkeypatch):
    store = Store()
    monkeypatch.setattr(media_store, "MediaStore", lambda *args, **kwargs: store)
    env.store = store
    env.template = _template(env, blocks=PHOTO_CARD, fmt="social_image")
    env.post = _create(env, format="image", template_id=str(env.template), length_seconds=None)
    return env


def _upload(env, data, *, slot="photo", name="visual.png", content_type="image/png"):
    form = {"slot": slot} if slot is not None else {}
    return env.client.post(f"/api/socials/posts/{env.post['id']}/media", files={"file": (name, data, content_type)}, data=form)


def _library(env, *, slot="photo", deliverable_id=LIBRARY_ID):
    return env.client.put(f"/api/socials/posts/{env.post['id']}/photos/{slot}", json={"deliverable_id": deliverable_id})


def _picture(monkeypatch, env, found):
    monkeypatch.setattr(media_urls, "deliverable_file", lambda db, workspace_id, deliverable_id, aspect="": found)
    if found is not None and found.key:
        env.store.objects[found.key] = (PNG, found.content_type)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def test_an_upload_fills_the_photo_spot_and_the_template_stays(photos):
    resp = _upload(photos, PNG)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    sha = hashlib.sha256(PNG).hexdigest()
    name = f"upload-{sha[:16]}.png"
    key = f"social-media/{WS}/{photos.post['id']}/{name}"
    assert list(photos.store.objects) == [key]
    (registered,) = Deliverables.calls
    assert registered["file_path"] == key and registered["artifact_type"] == "image"
    record = body["footage"]["photo"]
    assert record == {
        "prompt": f"Photo: your own picture ({name})", "status": "done", "toolkit": "upload", "deliverable_id": "d-video-1",
        "name": name, "content_type": "image/png", "bytes": len(PNG), "sha256": sha,
    }
    # The template, its words and the post's media are as they were: a render setting changed.
    assert body["template_id"] == str(photos.template) and body["media"] == photos.post["media"]
    assert body["content_hash"] == photos.post["content_hash"] and body["status"] == "draft"


@pytest.mark.parametrize(
    "slot, data, name, content_type, status",
    [
        ("hook", PNG, "visual.png", "image/png", 422),  # the template has no such spot
        ("photo", MP4, "clip.mp4", "video/mp4", 415),  # a photo spot takes a picture
    ],
)
def test_a_wrong_spot_or_a_video_is_refused_before_anything_is_stored(photos, slot, data, name, content_type, status):
    resp = _upload(photos, data, slot=slot, name=name, content_type=content_type)
    assert resp.status_code == status, resp.text
    assert photos.store.objects == {} and Deliverables.calls == []
    assert _post(photos, photos.post["id"]).footage in (None, {})


def test_a_post_with_no_template_has_no_photo_spot(photos):
    bare = _create(photos, format="image", length_seconds=None)
    resp = photos.client.post(f"/api/socials/posts/{bare['id']}/media", files={"file": ("v.png", PNG, "image/png")}, data={"slot": "photo"})
    assert resp.status_code == 422 and "Pick a template marked Photo" in resp.json()["detail"]
    assert photos.store.objects == {}


def test_without_a_slot_an_upload_is_still_the_whole_post(photos):
    resp = _upload(photos, PNG, slot=None)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["media"] == {"original": ["d-video-1"]} and body["template_id"] is None


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


def test_a_library_picture_is_copied_under_the_post_and_fills_the_spot(photos, monkeypatch):
    source = f"social-media/{WS}/elsewhere/cover.png"
    _picture(monkeypatch, photos, media_urls.MediaFile(aspect="", deliverable_id=LIBRARY_ID, name="cover.png", key=source,
                                                       content_type="image/png", bytes=len(PNG)))
    resp = _library(photos)

    assert resp.status_code == 200, resp.text
    name = f"library-{LIBRARY_ID.replace('-', '')[:16]}.png"
    copy = f"social-media/{WS}/{photos.post['id']}/{name}"
    assert photos.store.objects[copy] == photos.store.objects[source]
    record = resp.json()["footage"]["photo"]
    assert (record["toolkit"], record["status"], record["name"], record["deliverable_id"]) == ("library", "done", name, LIBRARY_ID)
    assert resp.json()["template_id"] == str(photos.template)


def test_a_library_file_that_is_not_a_stored_picture_or_not_the_workspaces_is_refused(photos, monkeypatch):
    _picture(monkeypatch, photos, None)
    assert _library(photos).status_code == 404
    _picture(monkeypatch, photos, media_urls.MediaFile(aspect="", deliverable_id=LIBRARY_ID, name="clip.mp4",
                                                       key=f"social-media/{WS}/x/clip.mp4", content_type="video/mp4"))
    assert _library(photos).status_code == 422
    _picture(monkeypatch, photos, media_urls.MediaFile(aspect="", deliverable_id=LIBRARY_ID, name="old.png", key=None,
                                                       error=media_urls.NOT_IN_STORAGE))
    resp = _library(photos)
    assert resp.status_code == 422 and resp.json()["detail"] == media_urls.NOT_IN_STORAGE
    assert _post(photos, photos.post["id"]).footage in (None, {})


# ---------------------------------------------------------------------------
# The render, the editor's saves, the AI label
# ---------------------------------------------------------------------------


def test_the_next_render_keeps_the_picture_and_an_editor_save_keeps_the_record(photos):
    record = _upload(photos, PNG).json()["footage"]["photo"]
    row = _post(photos, photos.post["id"])
    template = photos.session.query(DocumentTemplate.id, DocumentTemplate.format, DocumentTemplate.blocks).filter(
        DocumentTemplate.id == photos.template).one()

    plan = render.footage_plan_for(row, template, None)

    assert [(kept.slot, kept.name) for kept in plan.kept] == [("photo", record["name"])]
    assert plan.shots == () and plan.shown == ("photo",)

    # The editor sends each image slot's prompt back on save: an unchanged prompt keeps the file.
    resp = photos.client.patch(f"/api/socials/posts/{photos.post['id']}", json={"footage": {"photo": {"prompt": record["prompt"]}}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["footage"]["photo"] == record


def test_the_persons_own_picture_is_not_ai_made():
    upload = slot_photos.own_file_record({"label": "Photo"}, "photo", source=service.OWN_FILE_UPLOAD, deliverable_id="d1",
                                         name="upload-1.png", content_type="image/png")
    library = {**upload, "toolkit": service.OWN_FILE_LIBRARY}
    ai = {**upload, "toolkit": "fal_ai"}
    assert service.footage_generated({"photo": upload}) is False
    assert service.footage_generated({"photo": library}) is False
    assert service.footage_generated({"photo": upload, "hook": ai}) is True


def test_only_an_image_slot_is_a_photo_spot():
    with pytest.raises(service.InvalidPost, match="not a photo spot"):
        slot_photos.photo_spot({"slots": {"hook": {"kind": "video", "path": "assets/slots/hook.mp4"}}}, "hook")
    with pytest.raises(service.InvalidPost):
        slot_photos.photo_spot(None, "photo")
    assert slot_photos.photo_spot(PHOTO_CARD, "photo")["kind"] == "image"
