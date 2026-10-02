"""PRD-251B (US-B109): a post's music, a render setting like its voice.

``social_posts.music``:

* NULL plays the template's own track (its ``audio_plan.music``, as authored);
* ``{"track": "<id>"}`` plays a track of media-render's music library instead (``GET
  /api/socials/music`` lists them), keeping the template's fades; the cue names no start,
  so the library picks one (the track's first groove with room for the video);
* ``{"track": null}`` plays no music.

Outside the content hash, as the voice is: the rendered files' digests carry what it
changed, and a post's copy carries the credit its media's music asks for (S1.6).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from core.social_templates import MUSIC_TRACK_ID, MUSIC_TRACK_ID_MAX_CHARS

TRACK = "track"
MUSIC_SHAPE = 'music must be null (the template\'s track), {"track": null} (no music) or {"track": "<library track id>"}'
FADE_KEYS = ("fade_in", "fade_out")


def validate_music(value: Any) -> Optional[Dict[str, Any]]:
    """The post's music, checked: ``None``, ``{"track": None}`` or ``{"track": "<id>"}``."""
    from modules.socials.service import InvalidPost  # the service imports this module

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {TRACK}:
        raise InvalidPost(MUSIC_SHAPE)
    track = value[TRACK]
    if track is None:
        return {TRACK: None}
    if not isinstance(track, str) or len(track) > MUSIC_TRACK_ID_MAX_CHARS or not MUSIC_TRACK_ID.match(track):
        raise InvalidPost("music.track must be a music library track id, e.g. deep-house-003")
    return {TRACK: track}


def with_music(blocks: Mapping[str, Any], music: Any) -> Dict[str, Any]:
    """A video template's ``blocks`` with the post's music in its audio plan: the template's
    own cue when the post names none (or the same track), another library track with the
    template's fades and the library's start, or no music."""
    if not isinstance(music, Mapping) or TRACK not in music:
        return dict(blocks)
    plan = dict(blocks.get("audio_plan") or {})
    own = plan.get("music") if isinstance(plan.get("music"), Mapping) else {}
    track = music[TRACK]
    if track is None:
        plan.pop("music", None)
    elif track != own.get(TRACK):
        plan["music"] = {TRACK: track, **{key: own[key] for key in FADE_KEYS if key in own}}
    else:
        return dict(blocks)
    return {**blocks, "audio_plan": plan}
