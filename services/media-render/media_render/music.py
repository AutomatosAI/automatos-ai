"""The music library a render's audio plan names by track id (PRD-251 S1.6).

The library lives in the image under ``MEDIA_RENDER_MUSIC_DIR``: the tracks and
the ``manifest.json`` the image build wrote (``music_build.py``: every track
fetched from its source, checked against its sha256 and analysed). Each track
carries its ``id``, its ``file`` (a name inside the directory), its
``duration``, its title and artist, its licence and its attribution line, and
the cue windows the analysis found. An absent manifest is an empty library: a
bundle that names a track is then refused, never rendered silent. A track the
service could not credit (no licence it knows, no attribution) stops the boot.

A render that mixes a track reports it (``Track.report``): the orchestrator
appends a CC BY track's attribution to the copy of the post it lands in.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .music_build import LICENCES, MANIFEST_NAME


# A start the library picks lands this long before a drop, so the drop is heard.
DROP_LEAD_SECONDS = 2.0

class MusicLibraryError(ValueError):
    """The manifest cannot be used; the service refuses to start on it."""


@dataclass(frozen=True)
class Track:
    id: str
    path: Path
    duration: Optional[float]
    title: str = ""
    artist: str = ""
    licence: str = ""
    attribution: str = ""
    cues: Mapping[str, Any] = field(default_factory=dict)
    style: str = ""  # e.g. "deep house": how a picker names the track (PRD-251B, a post's music)

    @property
    def credit_required(self) -> bool:
        return bool(LICENCES[self.licence]["credit"])

    def start_for(self, duration: float) -> float:
        """Where a ``duration``-second video starts in the track when its cue names no start
        (PRD-251B: a post that picked this track over its template's): the first groove that
        leaves room for the whole video, where a voice-over sits well; else just before the
        first drop that does; else the opening."""
        room = None if self.duration is None else self.duration - duration
        grooves = [g.get("start") for g in self.cues.get("grooves") or [] if isinstance(g, Mapping)]
        drops = [max(0.0, d - DROP_LEAD_SECONDS) for d in self.cues.get("drops") or [] if isinstance(d, (int, float))]
        for start in (*grooves, *drops):
            if isinstance(start, (int, float)) and start >= 0 and (room is None or start <= room):
                return float(start)
        return 0.0

    def listing(self) -> Dict[str, Any]:
        """What ``GET /music`` lists for the track: who made it, its style, length and credit."""
        licence = LICENCES[self.licence]
        return {
            "id": self.id, "title": self.title, "artist": self.artist, "style": self.style,
            "duration": self.duration, "licence": licence["name"], "credit_required": bool(licence["credit"]),
        }

    def report(self) -> Dict[str, Any]:
        """What a render that mixes this track reports about it."""
        licence = LICENCES[self.licence]
        return {
            "track": self.id,
            "title": self.title,
            "artist": self.artist,
            "licence": licence["name"],
            "licence_url": licence["url"],
            "attribution": self.attribution,
            "credit_required": bool(licence["credit"]),
        }


def _text(entry: Mapping[str, Any], key: str, track_id: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MusicLibraryError(f"track {track_id}: {key} is required")
    return value.strip()


def _track(entry: Any, root: Path) -> Track:
    if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not isinstance(entry.get("file"), str):
        raise MusicLibraryError(f"every track needs an id and a file: {entry!r:.120}")
    track_id = entry["id"]
    path = (root / entry["file"]).resolve()
    if root not in path.parents:
        raise MusicLibraryError(f"track {track_id}: {entry['file']!r} is outside the library")
    if not path.is_file():
        raise MusicLibraryError(f"track {track_id}: {path} is missing")
    duration = entry.get("duration")
    if duration is not None and (
        isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0
    ):
        raise MusicLibraryError(f"track {track_id}: duration must be a positive number of seconds")
    licence = entry.get("licence")
    if licence not in LICENCES:
        raise MusicLibraryError(f"track {track_id}: licence must be one of {', '.join(LICENCES)}, not {licence!r}")
    cues = entry.get("cues") or {}
    if not isinstance(cues, dict):
        raise MusicLibraryError(f"track {track_id}: cues must be an object")
    return Track(
        id=track_id,
        path=path,
        duration=None if duration is None else float(duration),
        title=_text(entry, "title", track_id),
        artist=_text(entry, "artist", track_id),
        licence=licence,
        attribution=_text(entry, "attribution", track_id),
        cues=cues,
        style=entry.get("style").strip() if isinstance(entry.get("style"), str) else "",
    )


def load_library(music_dir: str) -> Mapping[str, Track]:
    root = Path(music_dir).resolve()
    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        return {}
    try:
        data = json.loads(manifest.read_text())
    except ValueError as exc:
        raise MusicLibraryError(f"{manifest} is not JSON: {exc}") from None
    tracks = data.get("tracks") if isinstance(data, dict) else None
    if not isinstance(tracks, list):
        raise MusicLibraryError(f"{manifest} needs a tracks list")
    library: Dict[str, Track] = {}
    for entry in tracks:
        track = _track(entry, root)
        if track.id in library:
            raise MusicLibraryError(f"track id {track.id} appears twice")
        library[track.id] = track
    return library
