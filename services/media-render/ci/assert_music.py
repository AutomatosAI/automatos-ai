#!/usr/bin/env python3
"""Assert the music library and a render that mixes it (US-112, the media-render CI job).

Stdlib only: it runs on the runner, not in the image. Two checks, either or both:

``--source`` and ``--built``: the committed manifest (music/manifest.json) and
the one the image build wrote (/opt/media-render/music/manifest.json, read out
of the image):
- every committed track is in the image with the same licence, attribution, url
  and sha256, so the build fetched and verified exactly those;
- every track has a duration and cue windows (breaks, breakdowns or grooves);
- each ``--break TRACK:START-END`` names a track whose analysis lists a break
  overlapping that window: the Markets reference cut on Deep House 003's
  bass-out break at 34.0-35.8 s.

``--job``: a render job (GET /render/{id}) whose report must name ``--track`` as
its music, with ``--credit`` as the credit line a post must carry.

One PASS/FAIL line per expectation, so the job log is the evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

KEPT = ("licence", "attribution", "url", "sha256")
WINDOW_KINDS = ("breaks", "breakdowns", "grooves")


def _window(spec: str) -> Tuple[str, float, float]:
    track, _, span = spec.partition(":")
    low, _, high = span.partition("-")
    return track, float(low), float(high)


def check_library(source: dict, built: dict, breaks: List[str], expect: Callable[[bool, str], None]) -> None:
    tracks: Dict[str, dict] = {t.get("id"): t for t in built.get("tracks", [])}
    committed = [t["id"] for t in source.get("tracks", [])]
    expect(sorted(tracks) == sorted(committed), f"the image carries the committed tracks: {', '.join(committed)}")
    for entry in source.get("tracks", []):
        track = tracks.get(entry["id"], {})
        same = all(track.get(key) == entry[key] for key in KEPT)
        expect(same, f"{entry['id']}: licence, attribution, url and sha256 as committed ({entry['sha256'][:12]}…)")
        cues = track.get("cues") or {}
        counts = {kind: len(cues.get(kind) or []) for kind in WINDOW_KINDS + ("drops",)}
        duration = track.get("duration") or 0
        expect(duration > 0 and sum(counts[k] for k in WINDOW_KINDS) > 0,
               f"{entry['id']}: {duration} s, cue windows {json.dumps(counts)}")
        print(f"      breaks {json.dumps(cues.get('breaks'))}")
        print(f"      breakdowns {json.dumps(cues.get('breakdowns'))}; drops {json.dumps(cues.get('drops'))}")
    for spec in breaks:
        track_id, low, high = _window(spec)
        listed = (tracks.get(track_id, {}).get("cues") or {}).get("breaks") or []
        hits = [w for w in listed if w.get("start", 0) < high and w.get("end", 0) > low]
        expect(bool(hits), f"{track_id}: a break overlapping {low:g}-{high:g} s (found {json.dumps(hits or listed)})")


def check_job(job: dict, track: str, credit: Optional[str], expect: Callable[[bool, str], None]) -> None:
    music = (job.get("report") or {}).get("music") or {}
    expect(job.get("status") == "done", f"the render job is done (found {job.get('status')})")
    expect(music.get("track") == track, f"the report names the music: {json.dumps(music)}")
    if credit is not None:
        expect(music.get("credit_required") is True and music.get("attribution") == credit,
               f"the report carries the credit line a post must carry: {credit}")
    print(f"mix: {json.dumps((job.get('report') or {}).get('audio', {}))}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="the committed manifest")
    parser.add_argument("--built", type=Path, help="the manifest the image build wrote")
    parser.add_argument("--break", dest="breaks", action="append", default=[], metavar="TRACK:START-END")
    parser.add_argument("--job", type=Path, help="a render job, as GET /render/{id} returned it")
    parser.add_argument("--track", help="the track the job's report must name")
    parser.add_argument("--credit", help="the credit line the job's report must carry")
    args = parser.parse_args(argv)

    failures: List[str] = []

    def expect(ok: bool, message: str) -> None:
        print(("PASS  " if ok else "FAIL  ") + message)
        if not ok:
            failures.append(message)

    if args.source and args.built:
        check_library(json.loads(args.source.read_text()), json.loads(args.built.read_text()), args.breaks, expect)
    if args.job:
        check_job(json.loads(args.job.read_text()), args.track, args.credit, expect)

    print(f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
