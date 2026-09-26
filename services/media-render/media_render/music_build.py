"""The music library's build step (PRD-251 S1.6): fetch, verify, analyse.

    python media_render/music_build.py music/manifest.json /opt/media-render/music

The Dockerfile runs it once, while the image builds:

1. The source manifest (``music/manifest.json``, committed) is checked: every
   track has an id, a title, an artist, a licence this library takes (CC0 1.0
   or CC BY 4.0), an attribution line naming the title and the artist (and the
   licence, when the licence asks for credit), an https source URL and a sha256.
2. Every track is fetched from its URL and its sha256 compared with the
   manifest's. A mismatch fails the build: no audio file is committed to git,
   so the hash is what pins the bytes, as ``kokoro/SHA256SUMS`` pins Kokoro's.
3. Every track is decoded by ffmpeg (mono, 22.05 kHz) and analysed with numpy
   in 0.1 s frames: the loudness (RMS, dBFS), the energy below 150 Hz (the kick
   and the bass) and the 300-3,000 Hz share of the energy (vocals, chops). The
   cue windows come from them (PRD-251A stage 3):
   - a break: the bass drops out for 1 s or more, and comes back;
   - a breakdown: the same, for 4 s or more;
   - a drop: the moment the bass comes back after either, or after an intro
     without it: where a template lands a reveal;
   - a groove: 4 s or more with the bass in and the mids no busier than the
     track's own middle: where a voice-over sits well.
   Every start and end is at 0.1 s. A 2-second map of the three measures goes
   alongside, for choosing a window by eye.
4. The library directory receives each track as ``<id>.<ext>``, and
   ``manifest.json``: the source fields, the licence's name, URL and whether it
   asks for credit, the file, the duration, the cues and the map. The service
   loads it at boot (``music.py``).

This file runs on its own, before the package is copied into the image, so it
imports nothing from media_render; numpy is imported only by the analysis. The
licence table is defined here and read by the service's loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

MANIFEST_NAME = "manifest.json"

# The licences the library takes. A CC BY track's attribution is the credit
# line a post that uses it must carry; CC0 asks for none.
LICENCES: Dict[str, Dict[str, Any]] = {
    "CC-BY-4.0": {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/", "credit": True},
    "CC0-1.0": {"name": "CC0 1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/", "credit": False},
}

SOURCE_REQUIRED = ("id", "title", "artist", "licence", "attribution", "url", "sha256")
SOURCE_OPTIONAL = ("style", "reference")
TRACK_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
TRACK_ID_MAX_CHARS = 64
TEXT_MAX_CHARS = 200
ATTRIBUTION_MAX_CHARS = 300
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TRACK_EXTENSIONS = frozenset({"mp3", "wav", "ogg", "flac", "m4a"})

# Fetching: a track is a few megabytes; anything past the cap is not a track.
FETCH_TIMEOUT_SECONDS = 120
FETCH_ATTEMPTS = 3
FETCH_BACKOFF_SECONDS = 2.0
FETCH_CHUNK_BYTES = 1 << 16
MAX_TRACK_BYTES = 64 * 1024 * 1024
USER_AGENT = "automatos-media-render-build/1 (+https://automatos.app)"

# The analysis.
ANALYSIS_RATE = 22050
DECODE_TIMEOUT_SECONDS = 300
HOP_SECONDS = 0.1
BASS_FROM_HZ = 20.0
BASS_BELOW_HZ = 150.0
MID_BAND_HZ = (300.0, 3000.0)
FFT_BATCH_FRAMES = 512
POWER_FLOOR = 1e-12
# A frame quieter than this (RMS, dBFS) is silence: it sets no reference.
SILENCE_DB = -50.0
# The bass level: the power mean over 0.5 s, about one beat, so the gaps between kicks do not count.
BASS_SMOOTH_FRAMES = 5
# The mid share: the mean over 1.1 s.
MID_SMOOTH_FRAMES = 11
# "The bass when it is in": this percentile of the bass level over the frames that are not silent.
BASS_REFERENCE_PERCENTILE = 0.75
# A break: the bass level this far under the reference, for at least BREAK_MIN_SECONDS.
BREAK_DEPTH_DB = 6.0
BREAK_MERGE_SECONDS = 0.3
BREAK_MIN_SECONDS = 1.0
BREAKDOWN_MIN_SECONDS = 4.0
# The smoothing moves a break's edges by up to this many frames; the raw frames put them back.
EDGE_FRAMES = BASS_SMOOTH_FRAMES // 2
# A drop: the first frame, this soon after the bass-out stretch, whose bass is back within DROP_LEVEL_DB.
DROP_WITHIN_SECONDS = 0.5
DROP_LEVEL_DB = 3.0
# A groove: the bass level within GROOVE_BASS_DB of the reference, and the mid share
# no higher than the track's median over those frames (never below the floor).
GROOVE_BASS_DB = 4.0
GROOVE_MID_SHARE_FLOOR = 0.25
GROOVE_MERGE_SECONDS = 0.5
GROOVE_MIN_SECONDS = 4.0
MAP_STEP_SECONDS = 2.0

ANALYSIS = {
    "resolution": HOP_SECONDS,
    "rate": ANALYSIS_RATE,
    "bass_below_hz": BASS_BELOW_HZ,
    "mid_band_hz": list(MID_BAND_HZ),
    "break_depth_db": BREAK_DEPTH_DB,
    "break_min_seconds": BREAK_MIN_SECONDS,
    "breakdown_min_seconds": BREAKDOWN_MIN_SECONDS,
    "groove_min_seconds": GROOVE_MIN_SECONDS,
    "map_step": MAP_STEP_SECONDS,
}

ABOUT = (
    "Built by media_render/music_build.py from music/manifest.json: every track fetched from its url and "
    "checked against its sha256, then analysed. Cue times are seconds into the track, at 0.1 s: a template "
    "names a track and a start (audio_plan.music), and a cue at t lands at t - start in the video."
)


class MusicBuildError(RuntimeError):
    """The library cannot be built as the manifest says; the image build stops."""


@dataclass(frozen=True)
class Frame:
    """One 0.1 s frame of a track."""

    at: float
    loudness_db: float
    bass_db: float
    mid_share: float


# ── the source manifest ─────────────────────────────────────────────────────
def _text_problems(entry: Mapping[str, Any], key: str, where: str, limit: int) -> List[str]:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        return [f"{where}: {key} is required"]
    if len(value) > limit or "\n" in value or "\r" in value:
        return [f"{where}: {key} must be one line of at most {limit} characters"]
    return []


def track_extension(url: str) -> str:
    """The audio format the URL's path names (``mp3``, ...); '' when it names none."""
    suffix = PurePosixPath(urlsplit(url).path).suffix.lower().lstrip(".")
    return suffix if suffix in TRACK_EXTENSIONS else ""


def track_file(entry: Mapping[str, Any]) -> str:
    """The track's file in the library: its id, with its source's extension."""
    return f"{entry['id']}.{track_extension(entry['url'])}"


def _track_problems(entry: Any, index: int) -> List[str]:
    where = f"tracks[{index}]"
    if not isinstance(entry, dict):
        return [f"{where} must be an object"]
    track_id = entry.get("id")
    if isinstance(track_id, str):
        where = f"{where} ({track_id})"
    problems = [f"{where}: {key} is required" for key in SOURCE_REQUIRED if key not in entry]
    problems += [
        f"{where}: {key} is not a track field ({', '.join(SOURCE_REQUIRED + SOURCE_OPTIONAL)})"
        for key in entry
        if key not in SOURCE_REQUIRED + SOURCE_OPTIONAL
    ]
    if problems:
        return problems
    if not isinstance(track_id, str) or len(track_id) > TRACK_ID_MAX_CHARS or not TRACK_ID.match(track_id):
        problems.append(f"{where}: id must be lowercase words joined by '-', at most {TRACK_ID_MAX_CHARS} characters")
    for key in ("title", "artist") + tuple(k for k in SOURCE_OPTIONAL if k in entry):
        problems += _text_problems(entry, key, where, TEXT_MAX_CHARS)
    problems += _text_problems(entry, "attribution", where, ATTRIBUTION_MAX_CHARS)
    licence = LICENCES.get(entry["licence"]) if isinstance(entry["licence"], str) else None
    if licence is None:
        problems.append(f"{where}: licence must be one of {', '.join(LICENCES)}")
    attribution = entry["attribution"] if isinstance(entry["attribution"], str) else ""
    for key in ("title", "artist"):
        if isinstance(entry.get(key), str) and entry[key] not in attribution:
            problems.append(f"{where}: the attribution must name the {key}, {entry[key]!r}")
    if licence is not None and licence["credit"] and licence["name"] not in attribution:
        problems.append(f"{where}: a {licence['name']} track's attribution must name its licence, {licence['name']!r}")
    url = entry["url"]
    if not isinstance(url, str) or urlsplit(url).scheme != "https" or not urlsplit(url).netloc:
        problems.append(f"{where}: url must be an https address")
    elif not track_extension(url):
        problems.append(f"{where}: url must name an audio file ({', '.join(sorted(TRACK_EXTENSIONS))})")
    if not isinstance(entry["sha256"], str) or not SHA256.match(entry["sha256"]):
        problems.append(f"{where}: sha256 must be 64 lowercase hex characters")
    return problems


def validate_source(data: Any) -> List[Dict[str, Any]]:
    """The source manifest's tracks, or :class:`MusicBuildError` listing every problem."""
    tracks = data.get("tracks") if isinstance(data, dict) else None
    if not isinstance(tracks, list) or not tracks:
        raise MusicBuildError("the music manifest needs a non-empty tracks list")
    problems: List[str] = []
    seen: Dict[str, int] = {}
    for index, entry in enumerate(tracks):
        problems += _track_problems(entry, index)
        track_id = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(track_id, str):
            if track_id in seen:
                problems.append(f"tracks[{index}]: id {track_id} is used by tracks[{seen[track_id]}] too")
            seen.setdefault(track_id, index)
    if problems:
        raise MusicBuildError("the music manifest cannot be used:\n  " + "\n  ".join(problems))
    return tracks


def load_source(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MusicBuildError(f"{path} cannot be read as JSON: {exc}") from None
    return validate_source(data)


# ── fetching and verifying ──────────────────────────────────────────────────
Opener = Callable[..., Any]


def _download(url: str, target: Path, opener: Opener, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    size = 0
    with opener(request, timeout=timeout) as response, target.open("wb") as handle:
        while True:
            chunk = response.read(FETCH_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_TRACK_BYTES:
                raise MusicBuildError(f"{url}: larger than {MAX_TRACK_BYTES} bytes, which no track is")
            digest.update(chunk)
            handle.write(chunk)
    if size == 0:
        raise OSError("the response was empty")
    return digest.hexdigest()


def fetch(
    url: str,
    target: Path,
    *,
    opener: Optional[Opener] = None,
    attempts: int = FETCH_ATTEMPTS,
    timeout: float = FETCH_TIMEOUT_SECONDS,
    backoff: float = FETCH_BACKOFF_SECONDS,
) -> str:
    """Download ``url`` to ``target``, retrying a failed transfer: the sha256 of the bytes."""
    open_url = opener or urllib.request.urlopen
    last: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return _download(url, target, open_url, timeout)
        except (OSError, urllib.error.URLError) as exc:
            last = exc
            if attempt < attempts:
                time.sleep(backoff * attempt)
    raise MusicBuildError(f"{url}: could not be fetched ({attempts} attempts): {last}")


def verify(track_id: str, expected: str, actual: str) -> None:
    """:class:`MusicBuildError` unless the download is the file the manifest pins."""
    if actual != expected:
        raise MusicBuildError(
            f"{track_id}: sha256 mismatch: the manifest pins {expected}, the download is {actual}. "
            "Either the source changed the file or the manifest is wrong; nothing is bundled unverified."
        )


# ── the analysis ────────────────────────────────────────────────────────────
def decode(path: Path, *, ffmpeg: str = "ffmpeg", rate: int = ANALYSIS_RATE, timeout: float = DECODE_TIMEOUT_SECONDS):
    """The track as mono float samples at ``rate`` (a numpy array), decoded by ffmpeg."""
    import numpy as np

    argv = [ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"]
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise MusicBuildError(f"{path.name}: decoding ran past {timeout:g} s") from None
    if proc.returncode != 0:
        raise MusicBuildError(f"{path.name}: ffmpeg could not decode it: {proc.stderr.decode(errors='replace')[-500:]}")
    samples = np.frombuffer(proc.stdout, dtype="<f4")
    if samples.shape[0] < rate:
        raise MusicBuildError(f"{path.name}: decodes to less than a second of audio")
    return samples


def _db(power: float) -> float:
    return 10.0 * math.log10(max(power, POWER_FLOOR))


def frame_features(samples: Any, rate: int, hop: float = HOP_SECONDS) -> List[Frame]:
    """Loudness, bass energy and mid share of every ``hop``-second frame (Hann window, one FFT each)."""
    import numpy as np

    size = int(round(rate * hop))
    count = int(samples.shape[0]) // size
    if count == 0:
        return []
    n_fft = 1 << (size - 1).bit_length()
    freqs = np.fft.rfftfreq(n_fft, 1.0 / rate)
    heard = freqs > 0
    bass_band = (freqs >= BASS_FROM_HZ) & (freqs < BASS_BELOW_HZ)
    mid_band = (freqs >= MID_BAND_HZ[0]) & (freqs < MID_BAND_HZ[1])
    window = np.hanning(size)
    frames: List[Frame] = []
    for first in range(0, count, FFT_BATCH_FRAMES):
        last = min(count, first + FFT_BATCH_FRAMES)
        block = np.asarray(samples[first * size : last * size], dtype=np.float64).reshape(last - first, size)
        power = np.abs(np.fft.rfft(block * window, n=n_fft, axis=1)) ** 2
        total = power[:, heard].sum(axis=1)
        bass = power[:, bass_band].sum(axis=1)
        mids = power[:, mid_band].sum(axis=1)
        mean_square = np.mean(block * block, axis=1)
        for j in range(last - first):
            whole = float(total[j])
            frames.append(
                Frame(
                    at=round((first + j) * hop, 3),
                    loudness_db=_db(float(mean_square[j])),
                    bass_db=_db(float(bass[j])),
                    mid_share=float(mids[j]) / whole if whole > POWER_FLOOR else 0.0,
                )
            )
    return frames


def _prefix(values: Sequence[float]) -> List[float]:
    sums = [0.0]
    for value in values:
        sums.append(sums[-1] + value)
    return sums


def _window_means(values: Sequence[float], width: int) -> List[float]:
    """The mean over a centred window of ``width`` frames (narrower at the ends)."""
    sums, half, n = _prefix(values), width // 2, len(values)
    return [(sums[min(n, i + half + 1)] - sums[max(0, i - half)]) / (min(n, i + half + 1) - max(0, i - half)) for i in range(n)]


def power_mean_db(values_db: Sequence[float], width: int) -> List[float]:
    """Each frame's level as the power mean (in dB) over a centred window of ``width`` frames."""
    return [_db(mean) for mean in _window_means([10.0 ** (v / 10.0) for v in values_db], width)]


def _db_mean(values_db: Sequence[float]) -> float:
    return _db(sum(10.0 ** (v / 10.0) for v in values_db) / len(values_db))


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def _runs(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    """``[start, end)`` frame ranges where ``flags`` holds."""
    runs, start = [], None
    for index, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    return runs


def _merge(runs: Sequence[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
    """Runs whose gap is at most ``gap`` frames, joined."""
    merged: List[Tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _widen(start: int, end: int, bass: Sequence[float], floor: float) -> Tuple[int, int]:
    """Put back the edges the smoothing moved: frames beside the run whose own bass is out too."""
    for _ in range(EDGE_FRAMES):
        if start > 0 and bass[start - 1] < floor:
            start -= 1
    for _ in range(EDGE_FRAMES):
        if end < len(bass) and bass[end] < floor:
            end += 1
    return start, end


def _drop_after(end: int, bass: Sequence[float], back: float, within: int) -> Optional[int]:
    """The first frame from ``end`` whose bass is ``back``, looking ``within`` frames."""
    for index in range(end, min(len(bass), end + within + 1)):
        if bass[index] >= back:
            return index
    return None


def _seconds(frames: int, hop: float) -> float:
    return round(frames * hop, 1)


def _grooves(level: Sequence[float], mids: Sequence[float], playing: Sequence[bool], reference: float, hop: float) -> List[Dict[str, float]]:
    bass_in = [playing[i] and level[i] >= reference - GROOVE_BASS_DB for i in range(len(level))]
    busy = [mids[i] for i in range(len(level)) if bass_in[i]]
    ceiling = max(GROOVE_MID_SHARE_FLOOR, _percentile(busy, 0.5)) if busy else GROOVE_MID_SHARE_FLOOR
    groove = [bass_in[i] and mids[i] <= ceiling for i in range(len(level))]
    runs = _merge(_runs(groove), int(round(GROOVE_MERGE_SECONDS / hop)))
    return [
        {"start": _seconds(start, hop), "end": _seconds(end, hop)}
        for start, end in runs
        if (end - start) * hop >= GROOVE_MIN_SECONDS - hop / 2
    ]


def find_cues(frames: Sequence[Frame], hop: float = HOP_SECONDS) -> Dict[str, Any]:
    """Breaks, breakdowns, drops and grooves, in seconds at 0.1 s (see the module's docstring)."""
    cues: Dict[str, Any] = {"breaks": [], "breakdowns": [], "drops": [], "grooves": []}
    playing = [frame.loudness_db >= SILENCE_DB for frame in frames]
    if not any(playing):
        return cues
    bass = [frame.bass_db for frame in frames]
    level = power_mean_db(bass, BASS_SMOOTH_FRAMES)
    mids = _window_means([frame.mid_share for frame in frames], MID_SMOOTH_FRAMES)
    reference = _percentile([level[i] for i in range(len(frames)) if playing[i]], BASS_REFERENCE_PERCENTILE)
    floor = reference - BREAK_DEPTH_DB
    stretches = _merge(_runs([value < floor for value in level]), int(round(BREAK_MERGE_SECONDS / hop)))
    for start, end in stretches:
        start, end = _widen(start, end, bass, floor)
        if end >= len(frames) or (end - start) * hop < BREAK_MIN_SECONDS - hop / 2:
            continue  # an outro (the bass never comes back), or too short to cut on
        drop = _drop_after(end, bass, reference - DROP_LEVEL_DB, int(round(DROP_WITHIN_SECONDS / hop)))
        if drop is not None:
            cues["drops"].append(_seconds(drop, hop))
        if start == 0:
            continue  # an intro: the beat coming in is a drop, the intro is no break
        window = {
            "start": _seconds(start, hop),
            "end": _seconds(end, hop),
            "depth_db": round(reference - _db_mean(bass[start:end]), 1),
            "mid_share": round(sum(mids[start:end]) / (end - start), 2),
        }
        cues["breakdowns" if (end - start) * hop >= BREAKDOWN_MIN_SECONDS - hop / 2 else "breaks"].append(window)
    cues["grooves"] = _grooves(level, mids, playing, reference, hop)
    return cues


def cue_windows(cues: Mapping[str, Any]) -> int:
    """How many windows (breaks, breakdowns, grooves) the cues hold."""
    return sum(len(cues.get(kind) or []) for kind in ("breaks", "breakdowns", "grooves"))


def loudness_map(frames: Sequence[Frame], hop: float = HOP_SECONDS, step: float = MAP_STEP_SECONDS) -> Dict[str, Any]:
    """The three measures per ``step`` seconds: the coarse map for choosing a window by eye."""
    per = max(1, int(round(step / hop)))
    blocks = [frames[i : i + per] for i in range(0, len(frames), per)]
    return {
        "step": step,
        "loudness_db": [round(_db_mean([f.loudness_db for f in block]), 1) for block in blocks],
        "bass_db": [round(_db_mean([f.bass_db for f in block]), 1) for block in blocks],
        "mid_share": [round(sum(f.mid_share for f in block) / len(block), 2) for block in blocks],
    }


def analyse(path: Path, *, ffmpeg: str = "ffmpeg") -> Dict[str, Any]:
    samples = decode(path, ffmpeg=ffmpeg)
    frames = frame_features(samples, ANALYSIS_RATE)
    return {
        "duration": round(samples.shape[0] / ANALYSIS_RATE, 2),
        "cues": find_cues(frames),
        "map": loudness_map(frames),
    }


# ── the library ─────────────────────────────────────────────────────────────
def built_track(entry: Mapping[str, Any], file_name: str, analysed: Mapping[str, Any]) -> Dict[str, Any]:
    """A track as the service's manifest lists it: the source fields, the licence, the file and the analysis."""
    licence = LICENCES[entry["licence"]]
    track = {key: entry[key] for key in SOURCE_REQUIRED + SOURCE_OPTIONAL if key in entry}
    track.update(
        licence_name=licence["name"],
        licence_url=licence["url"],
        credit_required=licence["credit"],
        file=file_name,
        duration=analysed["duration"],
        cues=analysed["cues"],
        map=analysed["map"],
    )
    return track


def build(source: Path, library: Path, *, ffmpeg: str = "ffmpeg", opener: Optional[Opener] = None) -> Dict[str, Any]:
    """Fetch, verify and analyse every track of ``source`` into ``library``; the manifest written there."""
    entries = load_source(source)
    library.mkdir(parents=True, exist_ok=True)
    tracks = []
    for entry in entries:
        target = library / track_file(entry)
        verify(entry["id"], entry["sha256"], fetch(entry["url"], target, opener=opener))
        analysed = analyse(target, ffmpeg=ffmpeg)
        cues = analysed["cues"]
        print(
            f"{entry['id']}: sha256 verified; {analysed['duration']:g} s; {len(cues['breaks'])} breaks, "
            f"{len(cues['breakdowns'])} breakdowns, {len(cues['drops'])} drops, {len(cues['grooves'])} grooves",
            flush=True,
        )
        tracks.append(built_track(entry, target.name, analysed))
    manifest = {"about": ABOUT, "analysis": ANALYSIS, "tracks": tracks}
    (library / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch, verify and analyse the music library (PRD-251 S1.6).")
    parser.add_argument("source", type=Path, help="the committed manifest, music/manifest.json")
    parser.add_argument("library", type=Path, help="where the tracks and the built manifest go")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args(argv)
    try:
        manifest = build(args.source, args.library, ffmpeg=args.ffmpeg)
    except MusicBuildError as exc:
        print(f"music library: {exc}", file=sys.stderr)
        return 1
    print(f"music library: {len(manifest['tracks'])} tracks in {args.library}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
