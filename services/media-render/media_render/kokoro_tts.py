"""Kokoro text-to-speech, the default voice (PRD-251 D11).

The GPL boundary: kokoro-onnx phonemizes through phonemizer and espeak-ng, both
GPL-3.0. They are imported here, inside the media-render image, and nowhere in
the orchestrator (PRD-251 D3 and Traps).

One WAV per script line, so every line can be placed exactly in the mix, with
the line's voiced segments (timing.py) so on-screen text can land on its words.
Callers serialise synthesis through tts.Speaker: one line is spoken at a time.
The service loads the model in the background once it is listening
(PRD-262 US-053, server.py), so the first line spoken does not wait for it.

A line's WAV is written unmarked and synthesis runs no exiftool, so the render's
timed voice step pays for no mark (PRD-264 EUAI-06a). Where a
line's WAV leaves the service, POST /tts marks it before it answers
(tts.mark_audio); a render's lines reach only the mix and the video, which are
marked after their last encode (pipeline.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple

from .config import Settings
from .timing import voiced_segments


@dataclass(frozen=True)
class SpokenLine:
    path: Path
    seconds: float
    sample_rate: int
    segments: Tuple[Tuple[float, float], ...] = ()


@lru_cache(maxsize=1)
def _model(model_path: str, voices_path: str, espeak_data_path: str) -> Any:
    # Imported here so the service can boot, and its tests can run, without
    # loading the 310 MB model until it is warmed (load_model) or a line is spoken.
    import espeakng_loader
    from kokoro_onnx import Kokoro
    from kokoro_onnx.config import EspeakConfig

    espeak = EspeakConfig(lib_path=espeakng_loader.get_library_path(), data_path=espeak_data_path)
    return Kokoro(model_path, voices_path, espeak_config=espeak)


def load_model(settings: Settings) -> float:
    """Load the model into the process (once; a later call finds it loaded): the seconds it took."""
    started = time.monotonic()
    _model(settings.kokoro_model_path, settings.kokoro_voices_path, settings.espeak_data_path)
    return round(time.monotonic() - started, 3)


def synthesize_line(
    text: str,
    output: Path,
    settings: Settings,
    *,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
    lang: Optional[str] = None,
) -> SpokenLine:
    """Speak one script line into ``output`` (a WAV, not yet marked) and report its length and segments.

    No exiftool runs here: the caller that hands the WAV out marks it (POST /tts, tts.mark_audio).
    """
    import soundfile

    line = text.strip()
    if not line:
        raise ValueError("a script line needs text to speak")
    model = _model(settings.kokoro_model_path, settings.kokoro_voices_path, settings.espeak_data_path)
    samples, sample_rate = model.create(
        line,
        voice=voice or settings.kokoro_voice,
        speed=settings.kokoro_speed if speed is None else speed,
        lang=lang or settings.kokoro_lang,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(output), samples, sample_rate)
    return SpokenLine(
        path=output,
        seconds=round(len(samples) / sample_rate, 3),
        sample_rate=sample_rate,
        segments=tuple(voiced_segments(samples, sample_rate)),
    )
