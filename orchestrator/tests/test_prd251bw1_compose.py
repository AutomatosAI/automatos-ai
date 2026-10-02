"""PRD-251B Wave 1, US-B103 — the composer takes the editor's choices (B5), and text posts.

On the US-207 compose harness (SQLite, the real gate, the model faked). Pinned:

* a chosen ``template_id`` is the only template the model sees and is used whatever
  the model answers; another workspace's template, or one of another format, is 422
  ``template_not_allowed`` with no model call;
* a chosen ``length_seconds`` must be one the chosen template declares (or, with the
  template left to Auto, one some video template declares): 422 ``length_not_declared``
  otherwise; the proposal carries it and the prompt carries the spoken-word budget;
* ``format: text`` has no template, no variables and no claims; a named channel without
  a text post kind is 422 ``channel_not_text``, an unnamed one is left out with a warning;
* the render follows the chosen length: the composition's root ``data-duration`` is the
  post's ``length_seconds`` (the template's own value stays the default);
* the save path refuses a length the template does not declare and a template on a text
  post; a text post with targets and no media is publishable once approved.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace


os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402

import api.socials_compose as compose_api  # noqa: E402
from core.social_templates import root_duration, with_root_duration  # noqa: E402
from modules.socials import render, service  # noqa: E402
import tests.test_prd251w2_compose as compose_harness  # noqa: E402
from tests.test_prd251w2_compose import BRIEF, SCHEMA, WS_A, WS_B, _answer, _compose  # noqa: E402,F401

api = compose_harness.api
composer = compose_harness.composer

ROOT_HTML = (
    '<!doctype html><html><head></head><body><div id="root" data-composition-id="main" data-width="1080" '
    'data-height="1920" data-duration="39.5"><h1>{{ headline }}</h1></div></body></html>'
)
VIDEO_BLOCKS = {"html": ROOT_HTML, "css": "", "variables_schema": SCHEMA, "sizes": ["1080x1920"], "durations": [15, 30]}
TEXT_KIND = SimpleNamespace(kind="text", available=True)
IMAGE_KIND = SimpleNamespace(kind="image", available=True)


def _video_template(state, workspace_id=WS_A, name="UI story promo", blocks=VIDEO_BLOCKS):
    template_id = uuid.uuid4()
    state.api.session.execute(
        sa.text(
            "INSERT INTO document_templates (id, workspace_id, name, format, data_schema, blocks) "
            "VALUES (:id, :ws, :name, 'social_video', '{}', :blocks)"
        ),
        {"id": template_id.hex, "ws": workspace_id.hex, "name": name, "blocks": json.dumps(blocks)},
    )
    state.api.session.commit()
    return str(template_id)


def _with_text_kinds(state, monkeypatch):
    channels = {
        "twitter": SimpleNamespace(toolkit="twitter", label="Twitter", post_kinds=(TEXT_KIND, IMAGE_KIND)),
        "linkedin": SimpleNamespace(toolkit="linkedin", label="LinkedIn", post_kinds=(TEXT_KIND, IMAGE_KIND)),
        "instagram": SimpleNamespace(toolkit="instagram", label="Instagram", post_kinds=(IMAGE_KIND,)),
    }
    monkeypatch.setattr(compose_api, "social_channels", lambda db, ws: [channels[t] for t in state.channels])


# ---------------------------------------------------------------------------
# A chosen template
# ---------------------------------------------------------------------------


def test_a_chosen_template_is_the_only_one_the_model_sees_and_is_used_whatever_it_answers(composer):
    video = _video_template(composer)
    resp = _compose(composer, [_answer(composer, format="video", template_id=composer.foreign)],
                    {"brief": BRIEF, "format": "video", "template_id": video})
    assert resp.status_code == 200, resp.text
    proposal = resp.json()
    assert proposal["template_id"] == video and proposal["format"] == "video"
    assert not any("is used instead" in w for w in proposal["warnings"])
    (messages,) = composer.model.asked
    material = json.loads(messages[1]["content"])
    assert [t["id"] for t in material["templates"]] == [video] and material["template_id"] == video
    assert material["templates"][0]["durations"] == [15, 30]
    assert "use the one template listed, and no other" in messages[0]["content"]


def test_another_workspaces_template_or_one_of_another_format_is_422_with_no_model_call(composer):
    for body in (
        {"brief": BRIEF, "format": "image", "template_id": composer.foreign},
        {"brief": BRIEF, "format": "video", "template_id": composer.template},  # an image template for a video
        {"brief": BRIEF, "format": "image", "template_id": str(uuid.uuid4())},
    ):
        resp = _compose(composer, [_answer(composer)], body)
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"]["code"] == "template_not_allowed"
    assert composer.built == [] and composer.model.asked == []


# ---------------------------------------------------------------------------
# A chosen length
# ---------------------------------------------------------------------------


def test_a_length_the_template_declares_rides_along_with_its_word_budget_and_another_is_422(composer):
    video = _video_template(composer)
    resp = _compose(composer, [_answer(composer, format="video", template_id=video)],
                    {"brief": BRIEF, "format": "video", "template_id": video, "length_seconds": 15})
    assert resp.status_code == 200, resp.text
    assert resp.json()["length_seconds"] == 15
    (messages,) = composer.model.asked
    material = json.loads(messages[1]["content"])
    assert (material["length_seconds"], material["spoken_words_budget"]) == (15, 38)
    assert "15 seconds long" in messages[0]["content"]

    resp = _compose(composer, [_answer(composer)], {"brief": BRIEF, "format": "video", "template_id": video, "length_seconds": 45})
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "length_not_declared"
    assert composer.model.asked == []


def test_with_the_template_left_to_auto_a_length_some_video_template_declares_passes(composer):
    _video_template(composer)
    assert _compose(composer, [_answer(composer, format="video")], {"brief": BRIEF, "format": "video", "length_seconds": 30}).status_code == 200
    resp = _compose(composer, [_answer(composer)], {"brief": BRIEF, "format": "video", "length_seconds": 60})
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "length_not_declared"
    assert _compose(composer, [_answer(composer)], {"brief": BRIEF, "length_seconds": 0}).status_code == 422  # ge=1


# ---------------------------------------------------------------------------
# A text post
# ---------------------------------------------------------------------------


def test_a_text_post_has_copy_only_and_goes_to_channels_with_a_text_kind(composer, monkeypatch):
    _with_text_kinds(composer, monkeypatch)
    answer = _answer(composer, format="text", template_id=None, variables={}, sources={})
    resp = _compose(composer, [answer], {"brief": BRIEF, "format": "text", "channels": ["twitter", "linkedin"]})
    assert resp.status_code == 200, resp.text
    proposal = resp.json()
    assert proposal["format"] == "text" and proposal["template_id"] is None and proposal["template"] is None
    assert proposal["variables"] == {} and proposal["sources"] == {}
    assert set(proposal["copy"]["per_channel"]) == {"twitter", "linkedin"} and proposal["channels"] == ["twitter", "linkedin"]
    (messages,) = composer.model.asked
    assert json.loads(messages[1]["content"])["templates"] == [] and "text-only post" in messages[0]["content"]

    # Unnamed channels: the one without a text kind is left out with a warning.
    resp = _compose(composer, [answer], {"brief": BRIEF, "format": "text"})
    assert resp.status_code == 200 and resp.json()["channels"] == ["twitter", "linkedin"]
    assert "instagram takes no text-only post; it was left out" in resp.json()["warnings"]
    # A named channel without one is refused; a template on a text post too.
    resp = _compose(composer, [answer], {"brief": BRIEF, "format": "text", "channels": ["instagram"]})
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "channel_not_text"
    resp = _compose(composer, [answer], {"brief": BRIEF, "format": "text", "template_id": composer.template})
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "template_not_allowed"


# ---------------------------------------------------------------------------
# The render follows the chosen length
# ---------------------------------------------------------------------------


def test_with_root_duration_sets_the_roots_data_duration():
    assert root_duration(with_root_duration(ROOT_HTML, 15)) == 15.0
    assert root_duration(with_root_duration(ROOT_HTML.replace(' data-duration="39.5"', ""), 30)) == 30.0
    assert with_root_duration("<html><body></body></html>", 15) == "<html><body></body></html>"
    assert "data-duration=\"15\"" in with_root_duration(ROOT_HTML, 15) and "39.5" not in with_root_duration(ROOT_HTML, 15)


def test_the_bundle_carries_the_posts_length_and_the_templates_own_without_one():
    template = SimpleNamespace(id=uuid.uuid4(), format="social_video", blocks={**VIDEO_BLOCKS, "variables_schema": {"headline": {"type": "text"}}})
    post = SimpleNamespace(id=uuid.uuid4(), workspace_id=WS_A, variables={"headline": {"value": "Three weeks"}}, length_seconds=15)
    assert root_duration(render.bundle_for(post, template, {})["composition"]["html"]) == 15.0
    post.length_seconds = None
    assert root_duration(render.bundle_for(post, template, {})["composition"]["html"]) == 39.5


# ---------------------------------------------------------------------------
# The save path
# ---------------------------------------------------------------------------


def test_saving_refuses_an_undeclared_length_and_a_template_on_a_text_post(composer):
    video = _video_template(composer)
    client = composer.api.client
    base = {"title": "Countdown", "format": "video", "template_id": video, "variables": {"headline": {"value": "x", "claim": False}}}
    assert client.post("/api/socials/posts", json={**base, "length_seconds": 45}).status_code == 422
    created = client.post("/api/socials/posts", json={**base, "length_seconds": 15})
    assert created.status_code == 201, created.text
    post = created.json()
    assert post["length_seconds"] == 15
    assert client.patch(f"/api/socials/posts/{post['id']}", json={"length_seconds": 45}).status_code == 422
    assert client.patch(f"/api/socials/posts/{post['id']}", json={"length_seconds": 30}).status_code == 200
    assert client.post("/api/socials/posts", json={"title": "Plain", "format": "text", "template_id": video}).status_code == 422
    plain = client.post("/api/socials/posts", json={"title": "Plain", "format": "text", "copy": {"base": "Hello"}})
    assert plain.status_code == 201 and plain.json()["format"] == "text" and plain.json()["template_id"] is None


def test_an_approved_text_post_with_targets_and_no_media_is_publishable():
    class _Db:
        def add(self, obj):
            pass

    post = service.create_draft(_Db(), workspace_id=WS_A, created_by="user-1", title="Plain", format="text", copy={"base": "Hello"})
    service.submit(post, "user-1")
    service.approve(post, "reviewer-1", content_hash=post.content_hash)
    assert post.media == {}
    service.assert_publishable(post)
