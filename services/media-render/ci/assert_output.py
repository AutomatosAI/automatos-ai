#!/usr/bin/env python3
"""Assert a rendered MP4 (the media-render CI job).

Stdlib only: it runs on the runner, not in the image. It reads ffprobe's JSON
(``-show_streams -show_format -of json``), and optionally the render job as
GET /render/{id} returned it and ffmpeg's ebur128 log, and prints one
PASS/FAIL line per expectation so the job log is the evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Callable, List, Optional

_INTEGRATED = re.compile(r"I:\s+(-?(?:\d+(?:\.\d+)?|inf))\s+LUFS")
_TRUE_PEAK = re.compile(r"Peak:\s+(-?(?:\d+(?:\.\d+)?|inf))\s+dBFS")


def _streams(probe: dict, kind: str) -> List[dict]:
    return [s for s in probe.get("streams", []) if s.get("codec_type") == kind]


def _fps(stream: dict) -> float:
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _summary_value(log: str, regex: re.Pattern) -> Optional[float]:
    match = regex.search(log[log.rfind("Summary:") :])
    return float(match.group(1)) if match else None


def _check_streams(probe: dict, args: argparse.Namespace, expect: Callable[[bool, str], None]) -> None:
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


def _check_job(job: dict, args: argparse.Namespace, expect: Callable[[bool, str], None]) -> None:
    report = job.get("report", {})
    expect(job.get("status") == "done", f"the render job is done (found {job.get('status')})")
    expect(report.get("check", {}).get("ok") is True, f"hyperframes check passed: {json.dumps(report.get('check', {}))}")
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
    print(f"mix: {json.dumps(report.get('audio', {}))}")


def _check_loudness(log: str, args: argparse.Namespace, expect: Callable[[bool, str], None]) -> None:
    integrated = _summary_value(log, _INTEGRATED)
    shown = "none" if integrated is None else f"{integrated:g} LUFS"
    ok = integrated is not None and abs(integrated - args.lufs) <= args.lufs_tolerance
    expect(ok, f"integrated loudness {args.lufs:g} ± {args.lufs_tolerance:g} LUFS (ebur128 found {shown})")
    peak = _summary_value(log, _TRUE_PEAK)
    print(f"true peak: {'none' if peak is None else f'{peak:g} dBFS'}")


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
    parser.add_argument("--job", type=Path, help="the render job, as GET /render/{id} returned it")
    parser.add_argument("--min-line-seconds", type=float, help="every voice line must be longer than this")
    parser.add_argument("--ebur128", type=Path, help="ffmpeg's ebur128 log for the rendered file")
    parser.add_argument("--lufs", type=float, help="the integrated loudness target")
    parser.add_argument("--lufs-tolerance", type=float, default=1.0)
    args = parser.parse_args(argv)

    failures: List[str] = []

    def expect(ok: bool, message: str) -> None:
        print(("PASS  " if ok else "FAIL  ") + message)
        if not ok:
            failures.append(message)

    _check_streams(json.loads(args.probe.read_text()), args, expect)
    if args.job is not None:
        _check_job(json.loads(args.job.read_text()), args, expect)
    if args.ebur128 is not None and args.lufs is not None:
        _check_loudness(args.ebur128.read_text(), args, expect)

    print(f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
