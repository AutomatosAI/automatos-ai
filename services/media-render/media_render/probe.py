"""ffprobe: how long a voice file lasts, and what a finished render contains."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ProbeError(RuntimeError):
    """ffprobe could not read the file."""


def probe(path: Path, *, ffprobe_bin: str, timeout_seconds: int) -> Dict[str, Any]:
    argv = [ffprobe_bin, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        raise ProbeError(f"ffprobe timed out on {path.name}") from None
    if proc.returncode != 0:
        raise ProbeError(f"ffprobe could not read {path.name}: {proc.stderr.strip()[-500:]}")
    try:
        return json.loads(proc.stdout)
    except ValueError:
        raise ProbeError(f"ffprobe printed unreadable JSON for {path.name}") from None


def _streams(info: Dict[str, Any], kind: str) -> List[Dict[str, Any]]:
    return [stream for stream in info.get("streams", []) if stream.get("codec_type") == kind]


def duration_seconds(info: Dict[str, Any]) -> Optional[float]:
    try:
        return round(float(info.get("format", {}).get("duration")), 3)
    except (TypeError, ValueError):
        return None


def _fps(stream: Dict[str, Any]) -> Optional[float]:
    try:
        return round(float(Fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "")), 3)
    except (ValueError, ZeroDivisionError):
        return None


def image_size(info: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """``(width, height)`` of an image's one picture stream."""
    video = _streams(info, "video")
    first = video[0] if video else {}
    return first.get("width"), first.get("height")


def output_facts(info: Dict[str, Any]) -> Dict[str, Any]:
    """The facts of a rendered MP4 the job report carries."""
    video, audio = _streams(info, "video"), _streams(info, "audio")
    first = video[0] if video else {}
    return {
        "duration": duration_seconds(info),
        "video_codec": first.get("codec_name"),
        "width": first.get("width"),
        "height": first.get("height"),
        "fps": _fps(first) if first else None,
        "audio_codec": audio[0].get("codec_name") if audio else None,
    }
