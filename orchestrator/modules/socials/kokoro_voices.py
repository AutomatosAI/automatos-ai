"""PRD-251B Wave 3 (B13; US-B306): Kokoro's voices can be chosen.

Kokoro, the voice built into media-render, is free. A post speaks with its template's own
Kokoro voice (``af_heart`` in every seeded starter) unless it picks another: the post's
``voice`` is then ``{"toolkit": "kokoro", "voice_id": <one of KOKORO_VOICES>, "name"}``.
The catalogue is the English voices of the ``voices-v1.0.bin`` media-render ships
(``services/media-render/kokoro/SHA256SUMS`` pins the file); a British voice is spoken as
``en-gb``, an American one as ``en-us``. ``GET /api/socials/voices`` offers Kokoro with
``lists_voices: true``, and ``GET /api/socials/voices/kokoro`` lists this catalogue.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

KOKORO = "kokoro"
DEFAULT_KOKORO_VOICE = "af_heart"
AMERICAN, BRITISH = "en-us", "en-gb"
# (voice id, name, description): the English voices of voices-v1.0.bin.
KOKORO_VOICES = (
    ("af_heart", "Heart", "American English, female (the default)"),
    ("af_bella", "Bella", "American English, female"),
    ("af_nicole", "Nicole", "American English, female, soft"),
    ("af_sarah", "Sarah", "American English, female"),
    ("af_sky", "Sky", "American English, female"),
    ("af_nova", "Nova", "American English, female"),
    ("af_river", "River", "American English, female"),
    ("af_jessica", "Jessica", "American English, female"),
    ("af_kore", "Kore", "American English, female"),
    ("af_alloy", "Alloy", "American English, female"),
    ("af_aoede", "Aoede", "American English, female"),
    ("am_adam", "Adam", "American English, male"),
    ("am_michael", "Michael", "American English, male"),
    ("am_eric", "Eric", "American English, male"),
    ("am_liam", "Liam", "American English, male"),
    ("am_onyx", "Onyx", "American English, male, deep"),
    ("am_echo", "Echo", "American English, male"),
    ("am_fenrir", "Fenrir", "American English, male"),
    ("am_puck", "Puck", "American English, male"),
    ("bf_emma", "Emma", "British English, female"),
    ("bf_isabella", "Isabella", "British English, female"),
    ("bf_alice", "Alice", "British English, female"),
    ("bf_lily", "Lily", "British English, female"),
    ("bm_george", "George", "British English, male"),
    ("bm_lewis", "Lewis", "British English, male"),
    ("bm_daniel", "Daniel", "British English, male"),
    ("bm_fable", "Fable", "British English, male"),
)
_BY_ID = {voice_id: (name, description) for voice_id, name, description in KOKORO_VOICES}


def is_kokoro_voice(voice_id: Any) -> bool:
    return isinstance(voice_id, str) and voice_id in _BY_ID


def language_of(voice_id: str) -> str:
    return BRITISH if voice_id.startswith("b") else AMERICAN


def validate_kokoro(voice: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """A Kokoro choice: ``None`` (the template's own voice) without a voice id, else the
    catalogue voice as ``{"toolkit", "voice_id", "name"}``; an id not in it is refused."""
    from modules.socials.service import InvalidPost  # the service imports this module

    voice_id = voice.get("voice_id")
    if voice_id in (None, ""):
        if voice.get("name"):
            raise InvalidPost("Choose a Kokoro voice by its voice_id (GET /api/socials/voices/kokoro lists them)")
        return None
    if not is_kokoro_voice(voice_id):
        raise InvalidPost(f"voice.voice_id must be one of Kokoro's voices ({', '.join(_BY_ID)})")
    return {"toolkit": KOKORO, "voice_id": voice_id, "name": _BY_ID[voice_id][0]}


def kokoro_listing(query: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """The catalogue as a voice listing (``id``, ``name``, ``description``), filtered by ``query``."""
    wanted = (query or "").strip().lower()
    voices = [
        {"id": voice_id, "name": name, "description": description}
        for voice_id, name, description in KOKORO_VOICES
        if not wanted or wanted in f"{voice_id} {name} {description}".lower()
    ]
    return voices[:limit] if limit else voices


def with_kokoro_voice(blocks: Mapping[str, Any], voice: Any) -> Dict[str, Any]:
    """A video template's ``blocks`` speaking with the post's Kokoro voice, when it picked one."""
    if not isinstance(voice, Mapping) or voice.get("toolkit") != KOKORO or not is_kokoro_voice(voice.get("voice_id")):
        return dict(blocks)
    plan = dict(blocks.get("audio_plan") or {})
    spoken = plan.get("voice")
    if not isinstance(spoken, Mapping):
        return dict(blocks)
    plan["voice"] = {**spoken, "voice": voice["voice_id"], "lang": language_of(voice["voice_id"])}
    return {**blocks, "audio_plan": plan}
