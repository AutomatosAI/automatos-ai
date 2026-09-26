"""Script windows: every voice line fits the time its script gives it (PRD-251 D11).

The script is split into lines, one audio file per line, whatever the source:
Kokoro speaks a line here; a workspace's voice toolkit (Fish Audio, ElevenLabs)
returns it as a file. A line's WINDOW runs from its start to the next line's
start, and the last line's to the end of the composition. The template times
the scenes, but a voice's own pace decides how long a line lasts, so the timing
flexes to the audio:

- a line that ends inside its window plays as it is;
- a line that runs past its window is sped up just enough to end
  ``gap`` seconds before the window closes, with ffmpeg's ``atempo`` (which
  keeps the pitch), up to ``max_tempo`` times as fast;
- a line that would need more is left as it is, and the voice check refuses the
  render, naming the line, its length and its window (pipeline.voice_findings).

A fitted line's voiced segments (the words, timing.py) are scaled with it, so
the report says where its words land now.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

# A tempo is rounded UP to this many decimals, so a fitted line never ends past its target.
TEMPO_DECIMALS = 4
# Float noise below this (in units of the last decimal) is not a reason to round up:
# 2.0 s into 2.0 - 0.1 - 0.3 s is 1.25x, not 1.2501x.
ROUNDING_NOISE_DECIMALS = 6
LOG_TAIL_CHARS = 2000


class FitError(RuntimeError):
    """ffmpeg could not change a line's tempo."""


@dataclass(frozen=True)
class Timed:
    """What the fit reads of a line: where it starts and how long it lasts."""

    id: str
    at: float
    seconds: Optional[float]


def window_ends(lines: Sequence[Timed], duration: float) -> Dict[str, float]:
    """Each line's window end: the next line's start, or the end of the composition for the last."""
    ordered = sorted(lines, key=lambda line: line.at)
    starts = [line.at for line in ordered[1:]] + [duration]
    return {line.id: end for line, end in zip(ordered, starts)}


def tempo_to_fit(at: float, seconds: Optional[float], window_end: float, gap: float) -> Optional[float]:
    """The tempo that ends the line ``gap`` seconds before its window closes.

    ``None`` when the line already ends inside its window (or its length is
    unknown); ``math.inf`` when no tempo can fit it (no room at all).
    """
    if seconds is None or at + seconds <= window_end:
        return None
    room = window_end - gap - at
    if room <= 0:
        return math.inf
    scale = 10**TEMPO_DECIMALS
    return math.ceil(round(seconds / room * scale, ROUNDING_NOISE_DECIMALS)) / scale


def plan_fit(lines: Sequence[Timed], duration: float, *, max_tempo: float, gap: float) -> Dict[str, float]:
    """``{line id: tempo}`` for every line that runs past its window and fits at ``max_tempo`` or less."""
    ends = window_ends(lines, duration)
    plan: Dict[str, float] = {}
    for line in lines:
        tempo = tempo_to_fit(line.at, line.seconds, ends[line.id], gap)
        if tempo is not None and tempo <= max_tempo:
            plan[line.id] = tempo
    return plan


def scaled_segments(segments: Sequence[Tuple[float, float]], tempo: float) -> Tuple[Tuple[float, float], ...]:
    """A line's voiced segments once it plays ``tempo`` times as fast."""
    return tuple((round(start / tempo, 3), round(end / tempo, 3)) for start, end in segments)


def stretch(source: Path, target: Path, tempo: float, *, ffmpeg_bin: str, timeout_seconds: int) -> None:
    """Write ``source`` played ``tempo`` times as fast, pitch kept, to ``target`` (a 16-bit WAV)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        ffmpeg_bin, "-v", "error", "-y", "-i", str(source),
        "-af", f"atempo={tempo:.{TEMPO_DECIMALS}f}", "-c:a", "pcm_s16le", str(target),
    ]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        raise FitError(f"fitting {source.name} ran past {timeout_seconds} s") from None
    if proc.returncode != 0 or not target.is_file():
        raise FitError(f"fitting {source.name} failed: {proc.stderr.strip()[-LOG_TAIL_CHARS:]}")
