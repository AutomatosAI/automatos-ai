"""PRD-251B Wave 1, US-B102 — templates for the gallery: ``GET /api/socials/templates``
with each template's lengths and thumbnail, and the thumbnail backfill.

On the S1.1c render harness (SQLite, media-render over an httpx mock transport,
storage faked). Pinned:

* the list carries id, name, format, kind, sizes, ``durations`` (``blocks.durations``
  when declared, else the root ``data-duration`` as the one length; an image has none)
  and ``thumbnail_url`` (presigned from the stored key, null without one); another
  workspace's templates never appear; a soft-deleted template is hidden;
* ``?format=`` narrows to one post format's kind; ``text`` lists none; a bogus format is 422;
  a Socials-off workspace gets 404; the route is a plain ``def`` in the committed manifest;
* the backfill renders each template lacking a thumbnail exactly once (a video as a
  still at its first second, half resolution), stores it under the template's id,
  writes the KEY, registers no Deliverable, books nothing; a second run renders nothing;
* without a renderer the list answers with nulls and starts nothing;
* a save that changes the composition clears the thumbnail; a rename keeps it.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import uuid
from pathlib import Path

import httpx
import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402

import api.socials_templates as templates_api  # noqa: E402
import tests.test_prd251w1_render_lifecycle as render_harness  # noqa: E402
from api.socials import _render_brand_kit  # noqa: E402
from core.media_render_client import MediaRenderClient  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.documents import template_summary  # noqa: E402
from modules.documents.template_service import DocumentTemplateService  # noqa: E402
from modules.socials import template_gallery, template_thumbnails  # noqa: E402
from modules.socials.media_store import MediaStore  # noqa: E402
from tests.test_prd251w1_render_lifecycle import COMPOSITION, CREATED, TOKEN, WS, WS_OTHER, FakeStore, Renderer, _ctx  # noqa: E402

env = render_harness.env
MANIFEST = _ORCH / "reports" / "route-manifest.json"
WS_OFF = uuid.UUID("00000000-0000-0000-0000-00000000b102")
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
THUMB_OUTPUT = {"name": "thumb.png", "aspect": "9:16", "width": 540, "height": 960, "bytes": len(PNG)}
IMAGE_COMPOSITION = {
    "html": '<!doctype html><html><head></head><body><div id="root" data-composition-id="main" data-width="1080" '
            'data-height="1080"><h1>{{ headline }}</h1></div></body></html>',
    "css": "h1 { color: var(--brand-text); }",
    "variables_schema": {"headline": {"type": "text"}},
    "sizes": ["1080x1080"],
}


def _template(env, *, name, fmt="social_video", blocks=COMPOSITION, workspace_id=WS, thumbnail=None,
              created_by=None, active=True, sample=None, description=None):
    template_id = uuid.uuid4()
    env.session.execute(
        sa.text(
            "INSERT INTO document_templates (id, workspace_id, name, description, format, data_schema, sample_data, "
            "blocks, thumbnail_url, created_by, is_active) VALUES (:id, :ws, :name, :description, :fmt, '{}', "
            ":sample, :blocks, :thumbnail, :created_by, :active)"
        ),
        {
            "id": template_id.hex, "ws": workspace_id.hex, "name": name, "description": description, "fmt": fmt,
            "sample": json.dumps(sample or {"headline": "Three weeks to go"}), "blocks": json.dumps(blocks),
            "thumbnail": thumbnail, "created_by": created_by, "active": 1 if active else 0,
        },
    )
    env.session.commit()
    return template_id


def _thumbnail_of(env, template_id):
    return env.session.execute(
        sa.text("SELECT thumbnail_url FROM document_templates WHERE id = :id"), {"id": template_id.hex}
    ).scalar_one()


@pytest.fixture
def gallery(env, monkeypatch):
    """Storage configured and presigning deterministic; the backfill recorded, never started."""
    monkeypatch.setattr(MediaStore, "configured", staticmethod(lambda: True))
    monkeypatch.setattr(MediaStore, "presigned_view", lambda self, key, ttl: f"https://signed/{key}?ttl={ttl}")
    env.backfills = []
    monkeypatch.setattr(template_thumbnails, "start_backfill", lambda ws, ids, *, brand_kit_of: env.backfills.append((ws, list(ids))))
    return env


def _list(env, **params):
    return env.client.get("/api/socials/templates", params=params)


# ---------------------------------------------------------------------------
# The list
# ---------------------------------------------------------------------------


def test_the_gallery_lists_the_workspaces_social_templates_with_lengths_and_thumbnails(gallery):
    key = f"social-media/{WS}/thumb/preview-video-9x16.png"
    a = _template(gallery, name="A video", thumbnail=key, created_by="system", description="The UI story")
    b = _template(gallery, name="B video", blocks={**COMPOSITION, "durations": [45, 15, 30, 15, 0, True]})
    c = _template(gallery, name="C image", fmt="social_image", blocks=IMAGE_COMPOSITION)
    _template(gallery, name="D hidden", active=False)
    _template(gallery, name="E elsewhere", workspace_id=WS_OTHER)
    resp = _list(gallery)
    assert resp.status_code == 200, resp.text
    rows = resp.json()
    assert [row["name"] for row in rows] == ["A video", "B video", "C image"]
    first, second, third = rows
    assert first["id"] == str(a) and first["kind"] == "video" and first["format"] == "social_video"
    assert first["sizes"] == ["1080x1920"] and first["durations"] == [40]  # the root data-duration, 39.5 s
    assert first["thumbnail_url"].startswith(f"https://signed/{key}?ttl=") and first["is_starter"] is True
    assert first["description"] == "The UI story"
    assert second["id"] == str(b) and second["durations"] == [15, 30, 45] and second["thumbnail_url"] is None
    assert second["is_starter"] is False
    assert third["id"] == str(c) and third["kind"] == "image" and third["durations"] == [] and third["sizes"] == ["1080x1080"]
    # The fields' examples: the template's sample text, which the editor greys into each empty field.
    assert third["sample_data"] == {"headline": "Three weeks to go"}


def test_a_fields_examples_are_the_sample_values_of_the_templates_own_fields():
    schema = {"headline": {"type": "text"}, "members": {"type": "number"}, "open": {"type": "boolean"}}
    sample = {"headline": "Open late", "members": 1200, "open": True, "gone": "not a field", "rows": [{"a": 1}]}
    assert template_gallery.examples_of(sample, schema) == {"headline": "Open late", "members": 1200, "open": True}
    assert template_gallery.examples_of(None, schema) == {} and template_gallery.examples_of(sample, None) == {}


def test_format_narrows_the_gallery_and_text_lists_none(gallery):
    _template(gallery, name="A video")
    _template(gallery, name="C image", fmt="social_image", blocks=IMAGE_COMPOSITION)
    assert [r["name"] for r in _list(gallery, format="video").json()] == ["A video"]
    assert [r["name"] for r in _list(gallery, format="image").json()] == ["C image"]
    assert [r["name"] for r in _list(gallery, format="carousel").json()] == ["C image"]
    assert _list(gallery, format="text").json() == []
    assert _list(gallery, format="bogus").status_code == 422


def test_a_socials_off_workspace_gets_404_and_the_route_is_a_plain_def_in_the_manifest(gallery):
    gallery.session.add(Workspace(
        id=WS_OFF, name="ws-off", plan="basic", plan_limits={}, settings={}, onboarding={},
        created_at=CREATED, updated_at=CREATED,
    ))
    gallery.session.commit()
    gallery.ctx = _ctx(WS_OFF)
    assert _list(gallery).status_code == 404
    assert not inspect.iscoroutinefunction(templates_api.list_social_templates)
    routes = json.loads(MANIFEST.read_text())["routes"]
    assert {"method": "GET", "path": "/api/socials/templates"} in routes


def test_the_list_starts_the_backfill_for_the_templates_without_a_thumbnail_only_with_a_renderer(gallery, monkeypatch):
    a = _template(gallery, name="A video")
    _template(gallery, name="B video", thumbnail="social-media/x/y/preview-video-9x16.png")
    c = _template(gallery, name="C image", fmt="social_image", blocks=IMAGE_COMPOSITION)
    assert _list(gallery).status_code == 200
    assert gallery.backfills == [(WS, [a, c])]
    gallery.backfills.clear()
    render_harness._set_config(monkeypatch, SOCIALS_RENDER_URL="")
    assert _list(gallery).status_code == 200 and gallery.backfills == []


# ---------------------------------------------------------------------------
# The backfill
# ---------------------------------------------------------------------------


def _run_backfill(env, ids, renderer, store):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(renderer.handler), headers={"X-Internal-Token": TOKEN}) as http:
            return await template_thumbnails.ensure_thumbnails(
                WS, ids, brand_kit_of=_render_brand_kit, client=MediaRenderClient(http), store=store,
                session_factory=env.factory,
            )

    return asyncio.run(go())


def test_the_backfill_renders_each_missing_template_once_and_writes_its_key(env):
    video = _template(env, name="A video")
    image = _template(env, name="C image", fmt="social_image", blocks=IMAGE_COMPOSITION)
    done = _template(env, name="B done", thumbnail="social-media/x/y/preview-video-9x16.png")
    renderer, store = Renderer(outputs=(THUMB_OUTPUT,), files={"thumb.png": PNG}), FakeStore()

    made = _run_backfill(env, [video, image, done], renderer, store)

    video_key = f"social-media/{WS}/{video}/preview-video-9x16.png"
    image_key = f"social-media/{WS}/{image}/preview-image-9x16.png"
    assert sorted(made) == sorted([video_key, image_key])
    assert len(renderer.bundles) == 2
    by_reference = {bundle["reference"]: bundle for bundle in renderer.bundles}
    video_bundle = by_reference[f"{render_harness.render.EXECUTION_PREFIX}{video}"]
    image_bundle = by_reference[f"{render_harness.render.EXECUTION_PREFIX}{image}"]
    # A video is one still at its still moment; an image is its own still, at 0.
    assert video_bundle["still"] == {"at": [1.0]} and image_bundle["still"] == {"at": [0.0]}
    assert video_bundle["variables"]["size.width"] == 540 and video_bundle["variables"]["size.height"] == 960
    assert image_bundle["variables"]["size.width"] == 540 and image_bundle["variables"]["size.height"] == 540
    assert sorted(store.objects) == sorted([video_key, image_key])
    assert _thumbnail_of(env, video) == video_key and _thumbnail_of(env, image) == image_key
    assert _thumbnail_of(env, done) == "social-media/x/y/preview-video-9x16.png"
    assert render_harness.Deliverables.calls == [] and env.booked == []  # a thumbnail is no Deliverable and books nothing

    assert _run_backfill(env, [video, image, done], renderer, store) == []
    assert len(renderer.bundles) == 2  # nothing was missing: no render


def test_a_failed_render_leaves_the_template_without_a_thumbnail(env):
    video = _template(env, name="A video")
    renderer = Renderer(status="failed", error={"code": "render_timed_out", "message": "too long"})
    assert _run_backfill(env, [video], renderer, FakeStore()) == []
    assert _thumbnail_of(env, video) is None


def test_a_save_that_changes_the_composition_clears_the_thumbnail_and_a_rename_keeps_it(env):
    key = f"social-media/{WS}/thumb/preview-video-9x16.png"
    template_id = _template(env, name="A video", thumbnail=key)
    service = DocumentTemplateService(env.session)
    renamed = service.update_template(template_id, WS, name="A video, renamed")
    assert renamed is not None and renamed.thumbnail_url == key
    changed = service.update_template(template_id, WS, blocks={**COMPOSITION, "sizes": ["1080x1920", "1080x1080"]})
    assert changed is not None and changed.thumbnail_url is None


def test_footage_slots_are_the_generatable_ones():
    blocks = {"slots": {"hook": {"kind": "video"}, "broll": {"kind": "video", "generate": True},
                        "logo_reel": {"kind": "video", "generate": False}, "bad": "not a slot"}}
    assert template_gallery.footage_slots(blocks) == ["broll", "hook"]
    assert template_gallery.footage_slots({}) == [] and template_gallery.footage_slots(None) == []


def test_durations_of_and_the_starter_marker():
    assert template_gallery.durations_of({"durations": [30, 15]}, "social_video") == [15, 30]
    assert template_gallery.durations_of(COMPOSITION, "social_video") == [40]
    assert template_gallery.durations_of(COMPOSITION, "social_image") == []
    assert template_gallery.durations_of({"durations": [0, True, "15"], "html": COMPOSITION["html"]}, "social_video") == [40]
    assert template_gallery.STARTER_CREATOR == template_summary.STARTER_CREATOR
    assert template_thumbnails.still_moment(COMPOSITION) == 1.0
    assert template_thumbnails.still_moment({"html": COMPOSITION["html"].replace("39.5", "1.2")}) == 0.6
