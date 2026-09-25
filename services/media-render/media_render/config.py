"""media-render's environment seam (PRD-251 D3).

The service is its own image and never sees the orchestrator's ``config.py``.
Every environment read in the service happens here, so call sites never read
``os.environ`` inline (the orchestrator's config discipline, and the worker's
``worker_config.py``). Defaults describe the image's own layout (Dockerfile).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, TypeVar

# The image switches these off (Dockerfile ENV) and boot refuses to start
# without them: the CLI sends PostHog telemetry by default, installs skills
# globally on init, and checks npm for updates (PRD-251 D3 and Traps).
HYPERFRAMES_OFF_SWITCHES = (
    "HYPERFRAMES_NO_TELEMETRY",
    "DO_NOT_TRACK",
    "HYPERFRAMES_SKIP_SKILLS",
    "HYPERFRAMES_NO_UPDATE_CHECK",
)

# espeak-ng truncates its data path at about 160 characters, and phonemizer
# resolves symlinks first, so the resolved path must stay under this.
ESPEAK_DATA_PATH_LIMIT = 160

# The same name on both sides, like WORKER_INTERNAL_TOKEN: the orchestrator's
# client sends it as X-Internal-Token (US-104).
TOKEN_ENV = "SOCIALS_RENDER_TOKEN"

RENDER_QUALITIES = frozenset({"draft", "looks", "standard", "high", "delivery"})
RENDER_FPS = frozenset({24, 25, 30, 50, 60})

T = TypeVar("T")


class ConfigError(ValueError):
    """A setting holds a value the service cannot run with."""


@dataclass(frozen=True)
class Settings:
    environment: str
    bind_host: str
    port: int
    internal_token: str
    espeak_data_path: str
    kokoro_model_path: str
    kokoro_voices_path: str
    kokoro_voice: str
    kokoro_speed: float
    kokoro_lang: str
    gsap_path: str
    versions_path: str
    browser_path: str
    hyperframes_bin: str
    ffmpeg_bin: str
    render_quality: str
    render_fps: int
    render_workers: str
    render_timeout_seconds: int
    check_timeout_seconds: int
    mix_timeout_seconds: int

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


def _parse(name: str, raw: str, convert: Callable[[str], T], valid: Callable[[T], bool], expected: str) -> T:
    try:
        value = convert(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not valid: expected {expected}") from None
    if not valid(value):
        raise ConfigError(f"{name}={raw!r} is not valid: expected {expected}")
    return value


def _workers(raw: str) -> str:
    if raw != "auto" and not (raw.isdigit() and int(raw) > 0):
        raise ValueError(raw)
    return raw


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Read the settings from ``env`` (the process environment by default).

    Raises ConfigError naming the variable when a value cannot be used, so a
    misconfigured container fails at boot rather than on its first render.
    """
    source = os.environ if env is None else env

    def text(name: str, default: str) -> str:
        return (source.get(name) or "").strip() or default

    def positive_int(name: str, default: str) -> int:
        return _parse(name, text(name, default), int, lambda v: v > 0, "a positive whole number")

    return Settings(
        environment=text("ENVIRONMENT", "development"),
        bind_host=text("MEDIA_RENDER_BIND_HOST", "0.0.0.0"),
        port=_parse(
            "MEDIA_RENDER_PORT", text("MEDIA_RENDER_PORT", "8090"), int, lambda v: 0 < v < 65536, "a TCP port"
        ),
        internal_token=(source.get(TOKEN_ENV) or "").strip(),
        espeak_data_path=text("MEDIA_RENDER_ESPEAK_DATA_PATH", "/opt/espeak"),
        kokoro_model_path=text("MEDIA_RENDER_KOKORO_MODEL", "/opt/kokoro/kokoro-v1.0.onnx"),
        kokoro_voices_path=text("MEDIA_RENDER_KOKORO_VOICES", "/opt/kokoro/voices-v1.0.bin"),
        kokoro_voice=text("MEDIA_RENDER_KOKORO_VOICE", "af_heart"),
        kokoro_speed=_parse(
            "MEDIA_RENDER_KOKORO_SPEED",
            text("MEDIA_RENDER_KOKORO_SPEED", "0.95"),
            float,
            lambda v: 0.5 <= v <= 2.0,
            "a speed between 0.5 and 2.0",
        ),
        kokoro_lang=text("MEDIA_RENDER_KOKORO_LANG", "en-us"),
        gsap_path=text("MEDIA_RENDER_GSAP_PATH", "/opt/media-render/vendor/gsap.min.js"),
        versions_path=text("MEDIA_RENDER_VERSIONS_PATH", "/opt/media-render/versions.json"),
        browser_path=text("HYPERFRAMES_BROWSER_PATH", "/opt/chrome/chrome-headless-shell"),
        hyperframes_bin=text("MEDIA_RENDER_HYPERFRAMES_BIN", "hyperframes"),
        ffmpeg_bin=text("MEDIA_RENDER_FFMPEG_BIN", "ffmpeg"),
        render_quality=_parse(
            "MEDIA_RENDER_QUALITY",
            text("MEDIA_RENDER_QUALITY", "delivery"),
            str,
            lambda v: v in RENDER_QUALITIES,
            "one of " + ", ".join(sorted(RENDER_QUALITIES)),
        ),
        render_fps=_parse(
            "MEDIA_RENDER_FPS",
            text("MEDIA_RENDER_FPS", "30"),
            int,
            lambda v: v in RENDER_FPS,
            "one of " + ", ".join(str(f) for f in sorted(RENDER_FPS)),
        ),
        render_workers=_parse(
            "MEDIA_RENDER_WORKERS", text("MEDIA_RENDER_WORKERS", "auto"), _workers, bool, "'auto' or a positive number"
        ),
        render_timeout_seconds=positive_int("MEDIA_RENDER_RENDER_TIMEOUT_SECONDS", "900"),
        check_timeout_seconds=positive_int("MEDIA_RENDER_CHECK_TIMEOUT_SECONDS", "300"),
        mix_timeout_seconds=positive_int("MEDIA_RENDER_MIX_TIMEOUT_SECONDS", "120"),
    )
