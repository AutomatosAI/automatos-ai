"""Kokoro, loaded once, speaking one sentence at a time as 16-bit PCM.

The GPL boundary: kokoro-onnx phonemizes through phonemizer and espeak-ng, both
GPL-3.0. They are imported here, inside this image (and media-render's), and
nowhere in the orchestrator.

The pattern is media-render's ``kokoro_tts.py`` (the import deferred until the
model loads, espeak-ng's data path from config, never from espeakng_loader), but
not its module: that one writes a WAV file per script line with its voiced
segments for a render's timing, and reads media-render's own settings. Here a
sentence becomes PCM in memory for the MP3 encoder, and nothing touches disk.

One onnxruntime session serves every request. kokoro-onnx serialises espeak-ng
(process-wide state) behind its own lock and runs inference concurrently, so
several syntheses share the session; the server bounds how many run at once.
The session's thread pool does not spin while idle: with several syntheses at
once, spinning threads would take the CPU the others need.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from .config import Settings

# Kokoro's output: mono float32 at 24 kHz.
SAMPLE_RATE = 24000
PCM_MAX = 32767
# Loggers that would write a request's text, or its phonemes, at DEBUG.
QUIET_LOGGERS = ("kokoro_onnx", "phonemizer")


class Engine:
    """The Kokoro model, loaded by ``load`` (once, in the background at boot)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Optional[Any] = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> float:
        """Load the model and the voices: the seconds it took. Blocking: run it in a thread."""
        started = time.monotonic()
        if self._model is None:
            self._model = _load_model(self._settings)
        return round(time.monotonic() - started, 3)

    def speak(self, text: str, voice: str, speed: float, language: str) -> bytes:
        """One sentence as little-endian 16-bit mono PCM at 24 kHz. Blocking: run it in a thread.

        Raises ValueError when Kokoro has nothing to say for it (no phonemes, an unknown voice).
        """
        if self._model is None:
            raise RuntimeError("the Kokoro model is not loaded yet")
        samples, sample_rate = self._model.create(text, voice=voice, speed=speed, lang=language)
        if sample_rate != SAMPLE_RATE:
            raise RuntimeError(f"Kokoro spoke at {sample_rate} Hz, not {SAMPLE_RATE}")
        return to_pcm16(samples)


def to_pcm16(samples: Any) -> bytes:
    """Float samples in [-1, 1] as little-endian 16-bit PCM, clipped."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * PCM_MAX).astype("<i2").tobytes()


def silence(seconds: float) -> bytes:
    """``seconds`` of 16-bit PCM silence at Kokoro's rate."""
    return bytes(2 * int(round(seconds * SAMPLE_RATE)))


def _load_model(settings: Settings) -> Any:
    # Imported here so the service boots, and its tests run, without loading the 310 MB model.
    import espeakng_loader
    import onnxruntime
    from kokoro_onnx import Kokoro
    from kokoro_onnx.config import EspeakConfig

    # kokoro-onnx logs each text's phonemes at DEBUG (its level follows LOG_LEVEL): the text never reaches a log.
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = settings.threads
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = onnxruntime.InferenceSession(
        settings.kokoro_model_path, sess_options=options, providers=["CPUExecutionProvider"]
    )
    espeak = EspeakConfig(lib_path=espeakng_loader.get_library_path(), data_path=settings.espeak_data_path)
    return Kokoro.from_session(session, settings.kokoro_voices_path, espeak_config=espeak)
