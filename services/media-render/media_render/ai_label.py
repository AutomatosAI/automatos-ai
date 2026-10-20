"""The visible "AI-generated" label on a video (PRD-264 EUAI-08a, EU AI Act Art. 50(4)).

A bundle asks for it with ``ai_label``:

    "ai_label": {"text": "AI-generated", "seconds": 3, "interval": 30}

The label shows for the first ``seconds`` of the video, and again for ``seconds`` from
the start of every ``interval`` after (0-3 s, 30-33 s, 60-63 s, ...). An interval no
longer than ``seconds`` shows it the whole video. The orchestrator sends it for a post
whose label is on, its text in the post's language
(``orchestrator/modules/socials/deepfake_label.py``).

Only the video render, ``render.mp4``, carries it: a still and a preview's frames and
reel are snapshots of the composition as it is, and their post carries the label in its
caption. The pipeline burns it in after ``hyperframes render`` and before the file is
marked (``provenance.py``) and signed (``content_credentials.py``), so the IPTC mark and the
C2PA manifest still come after the last encode: the signed file, and its recorded SHA-256,
are the labelled one.

ffmpeg's ``drawtext`` draws it top left, white on a dark box, sized to the frame's shorter
side, in the image's DejaVu Sans Bold (``MEDIA_RENDER_AI_LABEL_FONT``). The text reaches
ffmpeg through a file, with expansion off, and ffmpeg runs in the label's own directory,
so no character of the text or of the job's path is ever read as filter syntax.

Standard library only: the orchestrator's tests import the parser to hold the contract.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from . import validate
from .validate import BundleError

FIELD = "ai_label"
MAX_TEXT_CHARS = 40
LINE_BREAKS = ("\n", "\r", "\t")
# The label's look, as shares of the frame's shorter side and of the font size.
FONT_SHARE = 0.045
MIN_FONT_PX = 12
MARGIN_SHARE = 0.04
BOX_SHARE = 0.4
FONT_COLOUR = "white"
BOX_COLOUR = "black@0.6"
# Where the label's text and the labelled copy are written, inside the job's directory.
WORK_DIR = "label"
TEXT_FILE = "label.txt"
LABELLED_NAME = "labelled.mp4"
# The re-encode: the hyperframes render's codec and pixel format, at near-transparent quality.
ENCODE = (
    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
    "-c:a", "copy", "-movflags", "+faststart",
)
TAIL_CHARS = 4000


class LabelError(RuntimeError):
    """ffmpeg could not burn the label in."""


@dataclass(frozen=True)
class AiLabel:
    text: str
    seconds: float
    interval: float


def _positive(value: Any, where: str) -> float:
    number = validate.number(value, where, minimum=0)
    if number <= 0:
        raise BundleError(f"{where} must be more than 0")
    return number


def parse_ai_label(raw: Any) -> Optional[AiLabel]:
    """The bundle's ``ai_label``, or ``None`` when it asks for none. Raises BundleError.

    One line of text of at most MAX_TEXT_CHARS characters; ``seconds`` and ``interval``
    more than 0 (a video shorter than ``seconds`` shows the label all the way through)."""
    if raw is None:
        return None
    label = validate.mapping(raw, FIELD)
    validate.keys(label, FIELD, required=("text", "seconds", "interval"))
    text = validate.text(label["text"], f"{FIELD}.text", max_chars=MAX_TEXT_CHARS)
    if any(mark in text for mark in LINE_BREAKS):
        raise BundleError(f"{FIELD}.text must be one line")
    seconds = _positive(label["seconds"], f"{FIELD}.seconds")
    return AiLabel(text=text.strip(), seconds=seconds, interval=_positive(label["interval"], f"{FIELD}.interval"))


def shows_at(label: AiLabel, t: float) -> bool:
    """Whether the label is on screen at ``t`` seconds: the rule ``enable`` gives ffmpeg."""
    return math.fmod(t, label.interval) < label.seconds


def drawtext_filter(label: AiLabel, *, width: int, height: int, font_path: str) -> str:
    """The ``-vf`` that draws the label: ``font_path`` is the config's (a plain absolute
    path), and the text is :data:`TEXT_FILE` in the working directory."""
    short = min(width, height)
    size = max(MIN_FONT_PX, round(short * FONT_SHARE))
    margin = round(short * MARGIN_SHARE)
    enable = f"lt(mod(t,{label.interval:g}),{label.seconds:g})"
    return (
        f"drawtext=fontfile={font_path}:textfile={TEXT_FILE}:expansion=none:fontsize={size}"
        f":fontcolor={FONT_COLOUR}:box=1:boxcolor={BOX_COLOUR}:boxborderw={max(1, round(size * BOX_SHARE))}"
        f":x={margin}:y={margin}:enable='{enable}'"
    )


def burn_argv(ffmpeg_bin: str, video: Path, label: AiLabel, *, width: int, height: int, font_path: str) -> List[str]:
    """The ffmpeg command that writes :data:`LABELLED_NAME` (relative to its working directory)."""
    vf = drawtext_filter(label, width=width, height=height, font_path=font_path)
    return [ffmpeg_bin, "-v", "error", "-y", "-i", str(video), "-vf", vf, *ENCODE, LABELLED_NAME]


def burn(
    video: Path, label: AiLabel, *, work_dir: Path, width: int, height: int, ffmpeg_bin: str, font_path: str,
    timeout_seconds: int,
) -> float:
    """Burn ``label`` into ``video`` in place (a labelled copy replaces it); the seconds it took."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / TEXT_FILE).write_text(label.text, encoding="utf-8")
    argv = burn_argv(ffmpeg_bin, video.resolve(), label, width=width, height=height, font_path=font_path)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv, cwd=str(work_dir), capture_output=True, text=True, errors="replace", timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired:
        raise LabelError(f"ffmpeg ran past {timeout_seconds} s") from None
    labelled = work_dir / LABELLED_NAME
    if proc.returncode != 0 or not labelled.is_file():
        raise LabelError(proc.stderr.strip()[-TAIL_CHARS:] or f"ffmpeg exited {proc.returncode}")
    os.replace(labelled, video)
    return round(time.monotonic() - started, 3)
