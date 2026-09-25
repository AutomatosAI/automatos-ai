#!/usr/bin/env python3
"""Assert a rendered MP4's streams and its fixture report (the media-render CI job).

Stdlib only: it runs on the runner, not in the image. It reads ffprobe's JSON
(``-show_streams -show_format -of json``) and, optionally, the fixture report,
and prints one PASS/FAIL line per expectation so the job log is the evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import List, Optional


def _streams(probe: dict, kind: str) -> List[dict]:
    return [s for s in probe.get("streams", []) if s.get("codec_type") == kind]


def _fps(stream: dict) -> float:
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return 0.0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path, help="ffprobe JSON for the rendered file")
    parser.add_argument("--video-codec", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--audio-codec", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--tolerance", type=float, required=True, help="allowed duration error, seconds")
    parser.add_argument("--report", type=Path, help="the fixture report (fixture.json)")
    parser.add_argument("--min-line-seconds", type=float, help="every voice line must be longer than this")
    args = parser.parse_args(argv)

    probe = json.loads(args.probe.read_text())
    failures: List[str] = []

    def expect(ok: bool, message: str) -> None:
        print(("PASS  " if ok else "FAIL  ") + message)
        if not ok:
            failures.append(message)

    video, audio = _streams(probe, "video"), _streams(probe, "audio")
    codecs = [s.get("codec_name") for s in video]
    expect(codecs == [args.video_codec], f"one {args.video_codec} video stream (found {codecs})")
    first = video[0] if video else {}
    size = (first.get("width"), first.get("height"))
    expect(size == (args.width, args.height), f"{args.width}x{args.height} (found {size[0]}x{size[1]})")
    fps = _fps(first)
    expect(abs(fps - args.fps) < 0.01, f"{args.fps:g} fps (found {fps:.3f}, {first.get('avg_frame_rate')})")
    audio_codecs = [s.get("codec_name") for s in audio]
    expect(audio_codecs == [args.audio_codec], f"one {args.audio_codec} audio stream (found {audio_codecs})")
    duration = float(probe.get("format", {}).get("duration") or 0)
    expect(
        abs(duration - args.duration) <= args.tolerance,
        f"duration {args.duration:g} ± {args.tolerance:g} s (found {duration:.3f} s)",
    )

    if args.report is not None:
        report = json.loads(args.report.read_text())
        lines = report.get("voice", [])
        expect(bool(lines), f"the report lists the spoken lines ({len(lines)})")
        if args.min_line_seconds is not None:
            for line in lines:
                seconds = float(line.get("seconds") or 0)
                expect(
                    seconds > args.min_line_seconds,
                    f"Kokoro line {line.get('id')} lasts more than {args.min_line_seconds:g} s (found {seconds:.3f} s)",
                )
        print(f"timings: {json.dumps(report.get('timings', {}))}")
        print(f"hyperframes check (recorded, not yet a gate): {json.dumps(report.get('check', {}))}")

    print(f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
