"""PRD-251B (US-B109) — a post's music.

Pinned:

* ``music`` is null (the template's track), ``{"track": null}`` (none) or a library track
  id; anything else is refused;
* the render bundle plays it: another track with the template's fades and no start (the
  library picks one), no music at all, or the template's own cue untouched; a still has no
  audio to change;
* it is a render setting: changing it on an approved post keeps the approval and the hash;
* ``GET /api/socials/music`` lists media-render's library, says when there is no renderer,
  and is 503 while the renderer does not answer.
"""
from __future__ import annotations

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

import api.socials_templates as templates_api  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
from config import config  # noqa: E402
from core.media_render_client import MediaRenderUnavailable  # noqa: E402
from modules.documents import social_starters  # noqa: E402
from modules.socials import render, service  # noqa: E402
from modules.socials.music import validate_music, with_music  # noqa: E402
from tests.test_prd251_api import _approved  # noqa: E402

api = api_harness.api


@pytest.mark.parametrize("value, clean", [
    (None, None), ({"track": None}, {"track": None}), ({"track": "spring-of-2026"}, {"track": "spring-of-2026"}),
])
def test_a_music_choice_is_one_of_three(value, clean):
    assert validate_music(value) == clean


@pytest.mark.parametrize("value", [{"track": "Spring 2026"}, {"track": 7}, {}, {"track": "x", "start": 3}, "spring-of-2026"])
def test_anything_else_is_refused(value):
    with pytest.raises(service.InvalidPost):
        validate_music(value)


def test_the_audio_plan_takes_the_choice():
    blocks = {"html": "<x/>", "audio_plan": {"voice": {"lines": []}, "music": {"track": "where-the-night-begins", "start": 190.1, "fade_in": 0.03, "fade_out": 2.1}}}
    assert with_music(blocks, None) == blocks
    assert with_music(blocks, {"track": "where-the-night-begins"}) == blocks
    other = with_music(blocks, {"track": "spring-of-2026"})["audio_plan"]["music"]
    assert other == {"track": "spring-of-2026", "fade_in": 0.03, "fade_out": 2.1}  # the library picks the start
    assert "music" not in with_music(blocks, {"track": None})["audio_plan"]
    assert blocks["audio_plan"]["music"]["start"] == 190.1  # the template itself is untouched


def _bundle(music):
    starter = next(s for s in social_starters.social_starters() if s["slug"] == "ui-story-promo")
    template = SimpleNamespace(id=uuid.uuid4(), format="social_video", blocks=starter["blocks"])
    values = {name: {"value": value} for name, value in starter["sample_data"].items()}
    post = SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.uuid4(), variables=values, length_seconds=None, music=music)
    return render.bundle_for(post, template, {})


def test_the_render_plays_the_posts_music():
    assert _bundle(None)["audio"]["music"]["track"] == "where-the-night-begins"
    assert _bundle({"track": "spring-of-2026"})["audio"]["music"] == {"track": "spring-of-2026", "fade_in": 0.03, "fade_out": 2.1}
    assert "music" not in _bundle({"track": None})["audio"]
    social_starters._starters.cache_clear()


def test_music_is_a_render_setting_an_approval_survives(api):
    post = _approved(api)
    changed = api.client.patch(f"/api/socials/posts/{post['id']}", json={"music": {"track": "spring-of-2026"}})
    assert changed.status_code == 200, changed.text
    body = changed.json()
    assert body["music"] == {"track": "spring-of-2026"}
    assert (body["status"], body["content_hash"], body["approved_hash"]) == ("approved", post["content_hash"], post["content_hash"])
    assert api.client.patch(f"/api/socials/posts/{post['id']}", json={"music": {"track": "Not An Id"}}).status_code == 422


def test_the_music_library_route(api, monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_RENDER_URL", "")
    assert api.client.get("/api/socials/music").json() == {"tracks": [], "available": False}
    monkeypatch.setattr(config, "SOCIALS_RENDER_URL", "http://media-render:8090")
    tracks = [{"id": "spring-of-2026", "title": "Spring of 2026", "style": "tropical house"}]

    async def listed(self):
        return {"tracks": tracks}

    monkeypatch.setattr(templates_api.MediaRenderClient, "music", listed)
    assert api.client.get("/api/socials/music").json() == {"tracks": tracks, "available": True}

    async def down(self):
        raise MediaRenderUnavailable("unreachable", "media-render did not answer")

    monkeypatch.setattr(templates_api.MediaRenderClient, "music", down)
    assert api.client.get("/api/socials/music").status_code == 503
