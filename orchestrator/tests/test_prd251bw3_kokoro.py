"""PRD-251B Wave 3, US-B306 — Kokoro's voices can be chosen.

Pinned:

* the catalogue: the English voices media-render's ``voices-v1.0.bin`` holds, ``af_heart``
  first (the default every seeded starter speaks with), each id once, filterable;
* ``GET /api/socials/voices`` offers Kokoro with ``lists_voices: true``, and
  ``GET /api/socials/voices/kokoro`` lists the catalogue with no toolkit called;
* a post's voice: none (or Kokoro alone) is the template's own voice; a catalogue id is
  stored with its name; a name without an id, or an id Kokoro does not have, is refused;
  choosing one is a render setting, so an approval stands, and no toolkit is consulted;
* the render bundle speaks with it: the voice and its language (a British voice en-gb),
  the template's speed and lines untouched; a still has no voice to change.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.socials as socials_api  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
from modules.documents import social_starters  # noqa: E402
from modules.socials import kokoro_voices, render, service  # noqa: E402
from tests.test_prd251_api import _approved  # noqa: E402
from tests.test_prd251bw3_media_tools import _caps  # noqa: E402

api = api_harness.api


def test_the_catalogue_is_the_english_voices_default_first():
    ids = [voice_id for voice_id, _name, _description in kokoro_voices.KOKORO_VOICES]
    assert ids[0] == kokoro_voices.DEFAULT_KOKORO_VOICE == "af_heart"
    assert len(ids) == len(set(ids)) == 27
    assert all(voice_id[:2] in ("af", "am", "bf", "bm") for voice_id in ids)
    british = kokoro_voices.kokoro_listing("british")
    assert british and {voice["id"][0] for voice in british} == {"b"}
    assert kokoro_voices.kokoro_listing(limit=3) == kokoro_voices.kokoro_listing()[:3]
    assert kokoro_voices.kokoro_listing("George") == [{"id": "bm_george", "name": "George", "description": "British English, male"}]


@pytest.mark.parametrize("voice, stored", [
    (None, None),
    ({}, None),
    ({"toolkit": "kokoro"}, None),
    ({"toolkit": "kokoro", "voice_id": "bm_george"}, {"toolkit": "kokoro", "voice_id": "bm_george", "name": "George"}),
    ({"toolkit": "Kokoro", "voice_id": "af_bella", "name": "ignored"}, {"toolkit": "kokoro", "voice_id": "af_bella", "name": "Bella"}),
])
def test_a_kokoro_voice_is_the_templates_own_or_one_of_the_catalogue(voice, stored):
    assert service.validate_voice(voice) == stored


@pytest.mark.parametrize("voice", [
    {"toolkit": "kokoro", "name": "George"},
    {"toolkit": "kokoro", "voice_id": "zf_xiaoxiao"},
    {"toolkit": "kokoro", "voice_id": "../af_heart"},
])
def test_a_name_without_an_id_or_a_voice_kokoro_lacks_is_refused(voice):
    with pytest.raises(service.InvalidPost):
        service.validate_voice(voice)


def _bundle(voice, slug="ui-story-promo", fmt="social_video"):
    starter = next(s for s in social_starters.social_starters() if s["slug"] == slug)
    template = SimpleNamespace(id=uuid.uuid4(), format=fmt, blocks=starter["blocks"])
    values = {name: {"value": value} for name, value in starter["sample_data"].items()}
    post = SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.uuid4(), variables=values, length_seconds=None, music=None, voice=voice)
    return render.bundle_for(post, template, {})


def test_the_render_speaks_with_the_chosen_voice_in_its_language():
    own = _bundle(None)["audio"]["voice"]
    assert own["voice"] == "af_heart" and "lang" not in own
    british = _bundle({"toolkit": "kokoro", "voice_id": "bm_george", "name": "George"})["audio"]["voice"]
    assert (british["voice"], british["lang"], british["speed"]) == ("bm_george", "en-gb", own["speed"])
    assert british["lines"] == own["lines"]
    american = _bundle({"toolkit": "kokoro", "voice_id": "am_adam", "name": "Adam"})["audio"]["voice"]
    assert (american["voice"], american["lang"]) == ("am_adam", "en-us")
    toolkit = _bundle({"toolkit": "elevenlabs", "voice_id": "v1"})["audio"]["voice"]
    assert toolkit["voice"] == "af_heart"  # a toolkit's voice is spoken before the render, never by Kokoro
    social_starters._starters.cache_clear()


def test_the_blocks_are_never_changed_in_place():
    blocks = {"audio_plan": {"voice": {"voice": "af_heart", "lines": []}}}
    spoken = kokoro_voices.with_kokoro_voice(blocks, {"toolkit": "kokoro", "voice_id": "bf_emma"})
    assert spoken["audio_plan"]["voice"] == {"voice": "bf_emma", "lines": [], "lang": "en-gb"}
    assert blocks["audio_plan"]["voice"] == {"voice": "af_heart", "lines": []}
    assert kokoro_voices.with_kokoro_voice({"html": "<x/>"}, {"toolkit": "kokoro", "voice_id": "bf_emma"}) == {"html": "<x/>"}


def test_the_voice_routes_list_kokoro_with_its_catalogue(api, monkeypatch):
    monkeypatch.setattr(socials_api, "media_capabilities", lambda db, ws: _caps())
    sources = api.client.get("/api/socials/voices").json()["sources"]
    assert sources[0] == {"toolkit": "kokoro", "label": sources[0]["label"], "status": "available", "builtin": True, "lists_voices": True}
    listed = api.client.get("/api/socials/voices/kokoro", params={"q": "british"}).json()
    assert listed["toolkit"] == "kokoro" and {voice["id"][0] for voice in listed["voices"]} == {"b"}


def test_choosing_a_kokoro_voice_is_a_render_setting_and_consults_no_toolkit(api, monkeypatch):
    def no_toolkits(db, ws):
        raise AssertionError("a Kokoro voice needs no toolkit")

    monkeypatch.setattr(socials_api, "media_capabilities", no_toolkits)
    post = _approved(api)
    changed = api.client.patch(f"/api/socials/posts/{post['id']}", json={"voice": {"toolkit": "kokoro", "voice_id": "bf_emma"}})
    assert changed.status_code == 200, changed.text
    body = changed.json()
    assert body["voice"] == {"toolkit": "kokoro", "voice_id": "bf_emma", "name": "Emma"}
    assert (body["status"], body["content_hash"], body["approved_hash"]) == ("approved", post["content_hash"], post["content_hash"])
    refused = api.client.patch(f"/api/socials/posts/{post['id']}", json={"voice": {"toolkit": "kokoro", "voice_id": "xx_nobody"}})
    assert refused.status_code == 422
