"""The mix: the voice lines, the music bed and the SFX in one ffmpeg graph (PRD-251A stage 4).

The graph is the reference's (docs/PRDS/prd251-reference/mix-reference.py), with
any part allowed to be absent:
- the music window, faded in and out;
- each voice line placed with adelay, the lines mixed together;
- the music ducked under the voice by sidechaincompress (threshold 0.015,
  ratio 10, attack 10 ms, release 420 ms), then held at x0.6;
- each SFX placed with adelay at its own volume;
- everything summed with amix, normalised by loudnorm I=-14 TP=-1.5 LRA=11,
  and written as 48 kHz stereo 16-bit WAV: the composition's one <audio>.

Three changes from the reference script, all for the -14 LUFS target:
1. loudnorm runs in two passes: measure, then normalise with the measured
   values. That is how loudnorm normalises a file rather than a live stream;
   the reference's single pass landed 0.5-1.0 LU under the target on the
   40 s videos (PRD-251A section 1).
2. One second of silence is padded before loudnorm and trimmed after it.
   loudnorm gives any input under 3 s a linear gain capped by the sample peak,
   which leaves speech several LU short, so every mix takes its dynamic path.
3. The music window is read straight from the library track (-ss/-t), not
   from a pre-cut excerpt file.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Every stream is brought to the composition's single track format first.
TRACK_FORMAT = "aformat=sample_rates=48000:channel_layouts=stereo"
SAMPLE_RATE = 48000
MIX_CODEC = "pcm_s16le"
MUSIC_BED_LEVEL = 0.6
DUCKING = "sidechaincompress=threshold=0.015:ratio=10:attack=10:release=420:makeup=1"
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"
LOUDNESS_TARGET_LUFS = -14.0
LOUDNORM_PAD_SECONDS = 1.0
MUSIC_FADE_IN_SECONDS = 0.02
MUSIC_FADE_OUT_SECONDS = 1.7
SFX_DEFAULT_VOLUME = 0.5

# loudnorm option <- the first pass's JSON key, and the range loudnorm accepts.
_SECOND_PASS = (
    ("measured_I", "input_i", (-99.0, 0.0)),
    ("measured_TP", "input_tp", (-99.0, 99.0)),
    ("measured_LRA", "input_lra", (0.0, 99.0)),
    ("measured_thresh", "input_thresh", (-99.0, 0.0)),
    ("offset", "target_offset", (-99.0, 99.0)),
)
_LOUDNORM_JSON = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}")
_EBUR128_I = re.compile(r"I:\s+(-?(?:\d+(?:\.\d+)?|inf))\s+LUFS")
_EBUR128_LRA = re.compile(r"LRA:\s+(-?(?:\d+(?:\.\d+)?|inf))\s+LU\b")
_EBUR128_PEAK = re.compile(r"Peak:\s+(-?(?:\d+(?:\.\d+)?|inf))\s+dBFS")


class MixError(RuntimeError):
    """ffmpeg could not make the mix."""


@dataclass(frozen=True)
class VoiceTrack:
    path: Path
    at: float


@dataclass(frozen=True)
class MusicTrack:
    path: Path
    start: float
    fade_in: float = MUSIC_FADE_IN_SECONDS
    fade_out: float = MUSIC_FADE_OUT_SECONDS


@dataclass(frozen=True)
class SfxTrack:
    path: Path
    at: float
    volume: float = SFX_DEFAULT_VOLUME


@dataclass(frozen=True)
class MixPlan:
    duration: float
    voice: Tuple[VoiceTrack, ...] = ()
    music: Optional[MusicTrack] = None
    sfx: Tuple[SfxTrack, ...] = ()

    @property
    def silent(self) -> bool:
        return not (self.voice or self.music or self.sfx)


def _n(value: float) -> str:
    """A number as ffmpeg reads it: milliseconds precision, no trailing zeros."""
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


def _delay(at: float) -> str:
    ms = int(round(at * 1000))
    return f"adelay={ms}|{ms}"


def _placed(kind: str, tracks: Sequence[Any], first_input: int, duration: str, level: bool) -> Tuple[List[str], str]:
    """Filters placing each track at its start time, then summing them over the bed."""
    parts, labels = [], []
    for i, track in enumerate(tracks):
        volume = f"volume={_n(track.volume)}," if level else ""
        parts.append(f"[{first_input + i}:a]{TRACK_FORMAT},{volume}{_delay(track.at)}[{kind}{i}]")
        labels.append(f"[{kind}{i}]")
    summed = f"amix=inputs={len(labels)}:normalize=0:duration=longest,apad=whole_dur={duration},atrim=0:{duration}"
    return parts, "".join(labels) + summed


def filter_graph(plan: MixPlan, loudnorm: str) -> Tuple[List[str], str]:
    """The input arguments and the filter graph of the mix, ending in ``loudnorm``."""
    if plan.silent:
        raise ValueError("a silent plan has no mix graph")
    d = _n(plan.duration)
    inputs: List[str] = []
    graph: List[str] = []
    stems: List[str] = []
    # Input order: the music (if any), then the voice lines, then the SFX.
    first_voice = 1 if plan.music else 0
    first_sfx = first_voice + len(plan.voice)
    if plan.music:
        music = plan.music
        inputs += ["-ss", _n(music.start), "-t", d, "-i", str(music.path)]
        chain = [TRACK_FORMAT]
        if music.fade_in > 0:
            chain.append(f"afade=t=in:st=0:d={_n(music.fade_in)}")
        if music.fade_out > 0:
            chain.append(f"afade=t=out:st={_n(plan.duration - music.fade_out)}:d={_n(music.fade_out)}")
        graph.append("[0:a]" + ",".join(chain + [f"atrim=0:{d}"]) + "[m]")
    if plan.voice:
        parts, summed = _placed("v", plan.voice, first_voice, d, level=False)
        inputs += [arg for line in plan.voice for arg in ("-i", str(line.path))]
        graph += parts + [summed + "[vo]"]
        if plan.music:
            graph += ["[vo]asplit=2[vomix][vokey]", f"[m][vokey]{DUCKING}[mduck]", f"[mduck]volume={MUSIC_BED_LEVEL}[mq]"]
            stems += ["[mq]", "[vomix]"]
        else:
            stems.append("[vo]")
    elif plan.music:
        graph.append(f"[m]volume={MUSIC_BED_LEVEL}[mq]")
        stems.append("[mq]")
    if plan.sfx:
        parts, summed = _placed("s", plan.sfx, first_sfx, d, level=True)
        inputs += [arg for cue in plan.sfx for arg in ("-i", str(cue.path))]
        graph += parts + [summed + "[sx]"]
        stems.append("[sx]")
    mixed = f"amix=inputs={len(stems)}:normalize=0:duration=longest," if len(stems) > 1 else ""
    graph.append(
        "".join(stems)
        + mixed
        + f"atrim=0:{d},apad=pad_dur={_n(LOUDNORM_PAD_SECONDS)},{loudnorm},"
        + f"aresample={SAMPLE_RATE},atrim=0:{d},{TRACK_FORMAT}[out]"
    )
    return inputs, ";".join(graph)


def _graph_argv(ffmpeg_bin: str, plan: MixPlan, loudnorm: str) -> List[str]:
    inputs, graph = filter_graph(plan, loudnorm)
    return [ffmpeg_bin, "-hide_banner", "-nostats", "-y", *inputs, "-filter_complex", graph, "-map", "[out]"]


def measure_argv(ffmpeg_bin: str, plan: MixPlan) -> List[str]:
    """Pass one: run the graph with loudnorm measuring (its JSON goes to stderr)."""
    return _graph_argv(ffmpeg_bin, plan, f"{LOUDNORM}:print_format=json") + ["-f", "null", "-"]


def mix_argv(ffmpeg_bin: str, plan: MixPlan, loudnorm: str, output: Path) -> List[str]:
    """Pass two: the same graph, normalising with the measured values, into the mix file."""
    return _graph_argv(ffmpeg_bin, plan, loudnorm) + ["-c:a", MIX_CODEC, str(output)]


def silence_argv(ffmpeg_bin: str, duration: float, output: Path) -> List[str]:
    """A plan with no voice, music or SFX still gets its track: silence, at the mix format."""
    source = f"anullsrc=r={SAMPLE_RATE}:cl=stereo"
    return [ffmpeg_bin, "-hide_banner", "-nostats", "-v", "error", "-y", "-f", "lavfi", "-i", source] + [
        "-t", _n(duration), "-c:a", MIX_CODEC, str(output)
    ]


def loudnorm_stats(stderr: str) -> Dict[str, Any]:
    """The JSON loudnorm prints with print_format=json (the last one in the log)."""
    blocks = _LOUDNORM_JSON.findall(stderr)
    if not blocks:
        raise MixError("loudnorm printed no measurement")
    return json.loads(blocks[-1])


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 2) if math.isfinite(number) else None


def second_pass_loudnorm(stats: Mapping[str, Any]) -> Optional[str]:
    """loudnorm fed the first pass's measurements; None when the mix measured silent."""
    values = []
    for option, key, (low, high) in _SECOND_PASS:
        value = _finite(stats.get(key))
        if value is None:
            return None
        values.append(f"{option}={min(max(value, low), high):.2f}")
    return f"{LOUDNORM}:{':'.join(values)}:linear=true:print_format=json"


def _run(argv: Sequence[str], timeout_seconds: int) -> subprocess.CompletedProcess:
    # errors="replace": ffmpeg echoes input tags as they are, and a music file's
    # Latin-1 title must not crash the mix.
    try:
        proc = subprocess.run(
            list(argv), capture_output=True, text=True, errors="replace", timeout=timeout_seconds, check=False
        )
    except subprocess.TimeoutExpired:
        raise MixError(f"ffmpeg timed out after {timeout_seconds} s") from None
    if proc.returncode != 0:
        raise MixError(f"ffmpeg exit {proc.returncode}: {proc.stderr[-2000:]}")
    return proc


def run_mix(plan: MixPlan, output: Path, *, ffmpeg_bin: str, timeout_seconds: int) -> Dict[str, Any]:
    """Make the mix file; returns the loudness facts for the job report."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if plan.silent:
        _run(silence_argv(ffmpeg_bin, plan.duration, output), timeout_seconds)
        return {"silent": True}
    measured = loudnorm_stats(_run(measure_argv(ffmpeg_bin, plan), timeout_seconds).stderr)
    second = second_pass_loudnorm(measured)
    final_log = _run(mix_argv(ffmpeg_bin, plan, second or LOUDNORM, output), timeout_seconds).stderr
    final = loudnorm_stats(final_log) if second else {}
    return {
        "silent": False,
        "input_lufs": _finite(measured.get("input_i")),
        "integrated_lufs": _finite(final.get("output_i")),
        "true_peak_dbtp": _finite(final.get("output_tp")),
        "lra": _finite(final.get("output_lra")),
        "normalization": final.get("normalization_type"),
    }


def ebur128_argv(ffmpeg_bin: str, path: Path) -> List[str]:
    measure = ["-map", "0:a:0", "-af", "ebur128=peak=true", "-f", "null", "-"]
    return [ffmpeg_bin, "-hide_banner", "-nostats", "-i", str(path)] + measure


def ebur128_summary(stderr: str) -> Dict[str, Optional[float]]:
    """Integrated loudness, loudness range and true peak from ebur128's summary."""
    summary = stderr[stderr.rfind("Summary:") :]

    def first(regex: re.Pattern) -> Optional[float]:
        match = regex.search(summary)
        return _finite(match.group(1)) if match else None

    return {"integrated_lufs": first(_EBUR128_I), "lra": first(_EBUR128_LRA), "true_peak_dbfs": first(_EBUR128_PEAK)}


def measure_loudness(path: Path, *, ffmpeg_bin: str, timeout_seconds: int) -> Dict[str, Optional[float]]:
    return ebur128_summary(_run(ebur128_argv(ffmpeg_bin, path), timeout_seconds).stderr)
