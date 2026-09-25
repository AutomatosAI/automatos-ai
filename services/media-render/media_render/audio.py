"""The ffmpeg audio assembly.

US-101 places the voice lines, one WAV per line at its start time, on a bed
the length of the composition: the voice half of the reference mix
(docs/PRDS/prd251-reference/mix-reference.py). US-102 adds the music bed, its
ducking, the SFX and loudness normalisation to -14 LUFS.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

# Every stream is brought to the composition's single track format first.
TRACK_FORMAT = "aformat=sample_rates=48000:channel_layouts=stereo"


def voice_placement_argv(
    ffmpeg_bin: str,
    lines: Sequence[Tuple[Path, float]],
    duration: float,
    output: Path,
) -> List[str]:
    """The ffmpeg command that places each (wav, start seconds) line on the bed."""
    if duration <= 0:
        raise ValueError("the bed needs a positive duration")
    if not lines:
        raise ValueError("at least one voice line is needed")
    for path, start in lines:
        if not 0 <= start < duration:
            raise ValueError(f"{path.name} starts at {start}s, outside the {duration}s bed")

    argv = [ffmpeg_bin, "-v", "error", "-y"]
    for path, _ in lines:
        argv += ["-i", str(path)]
    graph = []
    for index, (_, start) in enumerate(lines):
        ms = int(round(start * 1000))
        graph.append(f"[{index}:a]{TRACK_FORMAT},adelay={ms}|{ms}[v{index}]")
    inputs = "".join(f"[v{index}]" for index in range(len(lines)))
    graph.append(
        f"{inputs}amix=inputs={len(lines)}:normalize=0:duration=longest,"
        f"apad=whole_dur={duration},atrim=0:{duration}[out]"
    )
    argv += ["-filter_complex", ";".join(graph), "-map", "[out]", "-c:a", "pcm_s16le", str(output)]
    return argv
