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

    @property
    def credit_required(self) -> bool:
        return bool(LICENCES[self.licence]["credit"])

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
