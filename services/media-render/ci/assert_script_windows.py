#!/usr/bin/env python3
"""Assert speech in every script window of a rendered MP4 (US-111, the media-render CI job).

Stdlib only: it runs on the runner. It reads the render job (GET /render/{id})
and the MP4's audio, decoded by the image's ffmpeg to raw 16-bit mono PCM, and
checks, for every voice line the report lists:

- the line has a script window (``window_end``), and its speech ends inside it;
- the audio is SPEECH where the line speaks, [at, at + seconds]: loud enough
  (``--min-speech-dbfs``) and well above the quiet between the lines
  (``--min-margin-db``). The fixture script carries no music, so between the
  lines the mix is silent;
- the lines named by ``--expect-fitted`` were sped up into their windows
  (a tempo above 1 and at most ``--max-tempo``).

One PASS/FAIL line per expectation, so the job log is the evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from array import array
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

# The quiet between the lines is read this far from any speech, clear of the
# AAC codec's spread at a line's edges.
EDGE_SECONDS = 0.08
# Stretches of quiet shorter than this are too short to measure.
MIN_QUIET_SECONDS = 0.15
# RMS of true digital silence, in dBFS, instead of minus infinity.
FLOOR_DBFS = -120.0
# How far a line's speech may run past its window before it counts.
TOLERANCE_SECONDS = 0.05


def _samples(pcm: Path) -> array:
    samples = array("h")
    samples.frombytes(pcm.read_bytes())
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def _dbfs(samples: array, rate: int, start: float, end: float) -> float:
    first, last = max(0, int(start * rate)), min(len(samples), int(end * rate))
    if last <= first:
        return FLOOR_DBFS
    power = sum(s * s for s in samples[first:last]) / (last - first)
    rms = math.sqrt(power) / 32768.0
    return 20 * math.log10(rms) if rms > 0 else FLOOR_DBFS


def _quiet_spans(spoken: Sequence[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
    """The stretches no line speaks in, each trimmed clear of the speech around it."""
    spans, cursor = [], 0.0
    for start, end in sorted(spoken):
        spans.append((cursor, start))
        cursor = max(cursor, end)
    spans.append((cursor, duration))
    trimmed = [(start + EDGE_SECONDS, end - EDGE_SECONDS) for start, end in spans]
    return [(start, end) for start, end in trimmed if end - start >= MIN_QUIET_SECONDS]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True, help="the render job, as GET /render/{id} returned it")
    parser.add_argument("--pcm", type=Path, required=True, help="the MP4's audio as raw s16le mono PCM")
    parser.add_argument("--rate", type=int, required=True, help="the PCM's sample rate")
    parser.add_argument("--duration", type=float, required=True, help="the composition's length, seconds")
    parser.add_argument("--min-speech-dbfs", type=float, default=-40.0)
    parser.add_argument("--min-margin-db", type=float, default=20.0)
    parser.add_argument("--expect-fitted", default="", help="comma-separated line ids that must have been fitted")
    parser.add_argument("--max-tempo", type=float, default=1.25)
    args = parser.parse_args(argv)

    failures: List[str] = []

    def expect(ok: bool, message: str) -> None:
        print(("PASS  " if ok else "FAIL  ") + message)
        if not ok:
            failures.append(message)

    job = json.loads(args.job.read_text())
    lines = job.get("report", {}).get("voice", [])
    samples = _samples(args.pcm)
    expect(bool(lines), f"the report lists the script's lines ({len(lines)})")
    expect(len(samples) >= args.rate * (args.duration - 0.1), f"the audio lasts the composition ({len(samples) / args.rate:.2f} s)")

    spoken = [(float(line["at"]), float(line["at"]) + float(line.get("seconds") or 0)) for line in lines]
    quiet = _quiet_spans(spoken, args.duration)
    quiet_db = max((_dbfs(samples, args.rate, start, end) for start, end in quiet), default=FLOOR_DBFS)
    print(f"between the lines: {len(quiet)} stretch(es), the loudest at {quiet_db:.1f} dBFS")

    for line in lines:
        line_id, at, seconds = line.get("id"), float(line["at"]), float(line.get("seconds") or 0)
        window_end = line.get("window_end")
        expect(window_end is not None, f"line {line_id} has a script window")
        if window_end is None:
            continue
        end = at + seconds
        expect(end <= float(window_end) + TOLERANCE_SECONDS,
               f"line {line_id} ends inside its window [{at:g}, {float(window_end):g}] s (speech ends at {end:.2f} s)")
        level = _dbfs(samples, args.rate, at, end)
        expect(level >= args.min_speech_dbfs,
               f"line {line_id} is speech in its window: {level:.1f} dBFS over [{at:g}, {end:.2f}] s (at least {args.min_speech_dbfs:g})")
        expect(level - quiet_db >= args.min_margin_db,
               f"line {line_id} stands {level - quiet_db:.1f} dB above the quiet between lines (at least {args.min_margin_db:g})")

    fitted = {line.get("id"): line.get("tempo") for line in lines if line.get("tempo") is not None}
    print(f"fitted lines: {json.dumps(fitted)}")
    for line_id in filter(None, (part.strip() for part in args.expect_fitted.split(","))):
        tempo = fitted.get(line_id)
        ok = tempo is not None and 1.0 < float(tempo) <= args.max_tempo
        expect(ok, f"line {line_id} was sped up into its window (tempo {tempo}, at most {args.max_tempo:g})")

    print(f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
