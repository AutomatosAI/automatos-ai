"""The music library a render's audio plan names by track id (PRD-251 S1.6).

The library lives in the image under ``MEDIA_RENDER_MUSIC_DIR``: a
``manifest.json`` whose ``tracks`` each carry at least ``id`` and ``file`` (a
name inside the directory) and, when known, ``duration`` in seconds. US-112
fills it with the curated, licensed tracks (and their licence, attribution and
cue fields, which this loader passes over). An absent manifest is an empty
library: a bundle that names a track is then refused, never rendered silent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

MANIFEST_NAME = "manifest.json"


class MusicLibraryError(ValueError):
    """The manifest cannot be used; the service refuses to start on it."""


@dataclass(frozen=True)
class Track:
    id: str
    path: Path
    duration: Optional[float]


def _track(entry: Any, root: Path) -> Track:
    if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not isinstance(entry.get("file"), str):
        raise MusicLibraryError(f"every track needs an id and a file: {entry!r:.120}")
    path = (root / entry["file"]).resolve()
    if root not in path.parents:
        raise MusicLibraryError(f"track {entry['id']}: {entry['file']!r} is outside the library")
    if not path.is_file():
        raise MusicLibraryError(f"track {entry['id']}: {path} is missing")
    duration = entry.get("duration")
    if duration is not None and (
        isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0
    ):
        raise MusicLibraryError(f"track {entry['id']}: duration must be a positive number of seconds")
    return Track(id=entry["id"], path=path, duration=None if duration is None else float(duration))


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
