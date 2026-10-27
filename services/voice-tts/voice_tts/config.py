"""voice-tts's environment seam: every setting the service reads, in one place.

The service is its own image and never sees the orchestrator's ``config.py``.
Every environment read happens here (the orchestrator's config discipline, and
media-render's ``config.py``), so call sites never read ``os.environ`` inline.
Defaults describe the image's own layout (Dockerfile).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, TypeVar

# The same name on both sides: the orchestrator sends it as ``Authorization: Bearer``.
TOKEN_ENV = "VOICE_TTS_TOKEN"
# The one way to run without a token, for a developer's or a test's own machine: boot
# refuses an empty VOICE_TTS_TOKEN in every environment unless this is "true".
ALLOW_NO_TOKEN_ENV = "VOICE_TTS_ALLOW_NO_TOKEN"

# The contract (README "API"): text of 1 to 500 characters after trimming; speed clamped to 0.8-1.2.
DEFAULT_MAX_CHARS = 500
SPEED_RANGE = (0.8, 1.2)
DEFAULT_SPEED = 1.0
# MPEG-2 Layer III (24 kHz) bitrates LAME encodes at a constant rate.
MP3_BITRATES = frozenset({8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160})
# cgroup v2 names the container's CPU quota here: "<quota> <period>", or "max <period>" for none.
CGROUP_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")

T = TypeVar("T")


class ConfigError(ValueError):
    """A setting holds a value the service cannot run with."""


@dataclass(frozen=True)
class Settings:
    environment: str
    bind_host: str
    port: int
    # Never in a repr, so never in a log line.
    token: str = field(repr=False)
    allow_no_token: bool
    kokoro_model_path: str
    kokoro_voices_path: str
    espeak_data_path: str
    lame_bin: str
    default_voice: str
    max_chars: int
    # How many syntheses run at once; past it a request is 429 ``tts_busy``.
    max_concurrent: int
    # onnxruntime's intra-op threads for one synthesis.
    threads: int
    mp3_kbps: int
    # Silence between two sentences, which Kokoro trims from each one it speaks.
    sentence_pause_seconds: float
    encode_timeout_seconds: int

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


def available_cpus(cpu_max: Path = CGROUP_CPU_MAX, cpu_count: Optional[int] = None) -> int:
    """The CPUs this container may use: its cgroup quota when it has one, else the machine's.

    ``os.cpu_count()`` reports the host's CPUs, not a container's limit (Railway's plan or
    compose's ``cpus``), so a default taken from it alone would run far more syntheses at once
    than the container can serve.
    """
    machine = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    try:
        quota, period = cpu_max.read_text(encoding="ascii").split()[:2]
        limit = math.ceil(int(quota) / int(period))
    except (OSError, ValueError):
        return max(1, machine)
    return max(1, min(machine, limit))


def _parse(name: str, raw: str, convert: Callable[[str], T], valid: Callable[[T], bool], expected: str) -> T:
    try:
        value = convert(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not valid: expected {expected}") from None
    if not valid(value):
        raise ConfigError(f"{name}={raw!r} is not valid: expected {expected}")
    return value


@dataclass(frozen=True)
class _Env:
    """One environment mapping; every refusal names its variable."""

    source: Mapping[str, str]

    def text(self, name: str, default: str) -> str:
        return (self.source.get(name) or "").strip() or default

    def positive_int(self, name: str, default: int) -> int:
        return _parse(name, self.text(name, str(default)), int, lambda v: v > 0, "a positive whole number")


def _limits(env: _Env, cpus: int) -> dict:
    """How much the service takes on at once, and how it encodes."""
    return {
        "max_chars": env.positive_int("VOICE_TTS_MAX_CHARS", DEFAULT_MAX_CHARS),
        "max_concurrent": env.positive_int("VOICE_TTS_MAX_CONCURRENT", cpus),
        "threads": env.positive_int("VOICE_TTS_THREADS", cpus),
        "mp3_kbps": _parse(
            "VOICE_TTS_MP3_KBPS",
            env.text("VOICE_TTS_MP3_KBPS", "64"),
            int,
            lambda v: v in MP3_BITRATES,
            "one of " + ", ".join(str(rate) for rate in sorted(MP3_BITRATES)),
        ),
        "sentence_pause_seconds": _parse(
            "VOICE_TTS_SENTENCE_PAUSE_SECONDS",
            env.text("VOICE_TTS_SENTENCE_PAUSE_SECONDS", "0.25"),
            float,
            lambda v: 0.0 <= v <= 2.0,
            "seconds from 0 to 2",
        ),
        "encode_timeout_seconds": env.positive_int("VOICE_TTS_ENCODE_TIMEOUT_SECONDS", 30),
    }


def load_settings(environ: Optional[Mapping[str, str]] = None, cpus: Optional[int] = None) -> Settings:
    """The settings from the environment (``os.environ`` unless given). Raises ConfigError."""
    env = _Env(os.environ if environ is None else environ)
    return Settings(
        environment=env.text("ENVIRONMENT", "development"),
        bind_host=env.text("VOICE_TTS_BIND_HOST", "0.0.0.0"),
        port=_parse("VOICE_TTS_PORT", env.text("VOICE_TTS_PORT", "8100"), int, lambda v: 0 < v < 65536, "a TCP port"),
        token=(env.source.get(TOKEN_ENV) or "").strip(),
        allow_no_token=env.text(ALLOW_NO_TOKEN_ENV, "false").lower() == "true",
        kokoro_model_path=env.text("VOICE_TTS_KOKORO_MODEL", "/opt/kokoro/kokoro-v1.0.onnx"),
        kokoro_voices_path=env.text("VOICE_TTS_KOKORO_VOICES", "/opt/kokoro/voices-v1.0.bin"),
        espeak_data_path=env.text("VOICE_TTS_ESPEAK_DATA_PATH", "/opt/espeak"),
        lame_bin=env.text("VOICE_TTS_LAME_BIN", "lame"),
        default_voice=env.text("VOICE_TTS_DEFAULT_VOICE", "af_heart"),
        **_limits(env, available_cpus() if cpus is None else cpus),
    )
