"""media-render's environment seam (PRD-251 D3).

The service is its own image and never sees the orchestrator's ``config.py``.
Every environment read in the service happens here, so call sites never read
``os.environ`` inline (the orchestrator's config discipline, and the worker's
``worker_config.py``). Defaults describe the image's own layout (Dockerfile).
Every limit and every timeout the service applies is a setting below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple, TypeVar

from .media_urls import UrlPrefix, parse_prefixes

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
# ffmpeg's atempo keeps speech natural this far; past it a line is rewritten, not sped up.
VOICE_TEMPO_RANGE = (1.0, 2.0)
VOICE_FIT_GAP_RANGE = (0.0, 1.0)

MEBIBYTE = 1024 * 1024

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
    ffprobe_bin: str
    render_quality: str
    render_fps: int
    render_workers: str
    render_timeout_seconds: int
    check_timeout_seconds: int
    mix_timeout_seconds: int
    probe_timeout_seconds: int
    fetch_timeout_seconds: int
    # Where jobs stage their compositions and keep their outputs until they expire.
    work_dir: str
    # The music library (US-112): manifest.json plus the tracks it names.
    music_dir: str
    # Presigned media URLs must fall under one of these (our storage only).
    media_url_prefixes: Tuple[UrlPrefix, ...]
    # Concurrency (owner, 2026-09-23): two renders at once overall, one per workspace.
    max_concurrent_renders: int
    max_renders_per_workspace: int
    # Staging and `hyperframes check` run before the render queue, bounded here.
    max_concurrent_checks: int
    max_checks_per_workspace: int
    # Jobs not yet finished (checking, queued or rendering); past this, 503.
    max_active_jobs: int
    # What a 503 tells the caller to wait before it submits again.
    busy_retry_after_seconds: int
    job_ttl_seconds: int
    sweep_interval_seconds: int
    max_bundle_bytes: int
    max_asset_bytes: int
    max_media_bytes: int
    max_files: int
    max_duration_seconds: int
    max_variables: int
    max_variable_chars: int
    # Brand tokens, and brand fonts, per bundle.
    max_brand_tokens: int
    tts_max_lines: int
    tts_max_chars: int
    # Script windows (US-111, fit.py): a voice line longer than its window is
    # sped up (pitch kept) to end this many seconds before the next line, at most
    # this many times as fast; a line that needs more is refused.
    voice_max_tempo: float
    voice_fit_gap_seconds: float
    # A preview (US-106): snapshot frames of the composition instead of the full
    # render, scaled to this width, joined into a short reel at this many frames a second.
    preview_width: int
    preview_max_frames: int
    preview_reel_fps: int
    preview_timeout_seconds: int
    # A still (US-107): an image, one full-size PNG per moment; a carousel takes one per slide.
    still_max_frames: int

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

    def positive_int(name: str, default: int) -> int:
        return _parse(name, text(name, str(default)), int, lambda v: v > 0, "a positive whole number")

    def prefixes(name: str) -> Tuple[UrlPrefix, ...]:
        try:
            return parse_prefixes(source.get(name) or "")
        except ValueError as exc:
            raise ConfigError(f"{name}: {exc}") from None

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
        ffprobe_bin=text("MEDIA_RENDER_FFPROBE_BIN", "ffprobe"),
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
        render_timeout_seconds=positive_int("MEDIA_RENDER_RENDER_TIMEOUT_SECONDS", 900),
        check_timeout_seconds=positive_int("MEDIA_RENDER_CHECK_TIMEOUT_SECONDS", 300),
        mix_timeout_seconds=positive_int("MEDIA_RENDER_MIX_TIMEOUT_SECONDS", 120),
        probe_timeout_seconds=positive_int("MEDIA_RENDER_PROBE_TIMEOUT_SECONDS", 60),
        fetch_timeout_seconds=positive_int("MEDIA_RENDER_FETCH_TIMEOUT_SECONDS", 120),
        work_dir=text("MEDIA_RENDER_WORK_DIR", "/tmp/media-render"),
        music_dir=text("MEDIA_RENDER_MUSIC_DIR", "/opt/media-render/music"),
        media_url_prefixes=prefixes("MEDIA_RENDER_MEDIA_URL_PREFIXES"),
        max_concurrent_renders=positive_int("MEDIA_RENDER_MAX_CONCURRENT_RENDERS", 2),
        max_renders_per_workspace=positive_int("MEDIA_RENDER_MAX_RENDERS_PER_WORKSPACE", 1),
        max_concurrent_checks=positive_int("MEDIA_RENDER_MAX_CONCURRENT_CHECKS", 1),
        max_checks_per_workspace=positive_int("MEDIA_RENDER_MAX_CHECKS_PER_WORKSPACE", 1),
        max_active_jobs=positive_int("MEDIA_RENDER_MAX_ACTIVE_JOBS", 20),
        busy_retry_after_seconds=positive_int("MEDIA_RENDER_BUSY_RETRY_AFTER_SECONDS", 30),
        job_ttl_seconds=positive_int("MEDIA_RENDER_JOB_TTL_SECONDS", 3600),
        sweep_interval_seconds=positive_int("MEDIA_RENDER_SWEEP_INTERVAL_SECONDS", 60),
        max_bundle_bytes=positive_int("MEDIA_RENDER_MAX_BUNDLE_BYTES", 32 * MEBIBYTE),
        max_asset_bytes=positive_int("MEDIA_RENDER_MAX_ASSET_BYTES", 8 * MEBIBYTE),
        max_media_bytes=positive_int("MEDIA_RENDER_MAX_MEDIA_BYTES", 256 * MEBIBYTE),
        max_files=positive_int("MEDIA_RENDER_MAX_FILES", 32),
        max_duration_seconds=positive_int("MEDIA_RENDER_MAX_DURATION_SECONDS", 180),
        max_variables=positive_int("MEDIA_RENDER_MAX_VARIABLES", 200),
        max_variable_chars=positive_int("MEDIA_RENDER_MAX_VARIABLE_CHARS", 2000),
        max_brand_tokens=positive_int("MEDIA_RENDER_MAX_BRAND_TOKENS", 64),
        tts_max_lines=positive_int("MEDIA_RENDER_TTS_MAX_LINES", 40),
        tts_max_chars=positive_int("MEDIA_RENDER_TTS_MAX_CHARS", 500),
        voice_max_tempo=_parse(
            "MEDIA_RENDER_VOICE_MAX_TEMPO",
            text("MEDIA_RENDER_VOICE_MAX_TEMPO", "1.25"),
            float,
            lambda v: VOICE_TEMPO_RANGE[0] <= v <= VOICE_TEMPO_RANGE[1],
            f"a tempo from {VOICE_TEMPO_RANGE[0]:g} to {VOICE_TEMPO_RANGE[1]:g}",
        ),
        voice_fit_gap_seconds=_parse(
            "MEDIA_RENDER_VOICE_FIT_GAP_SECONDS",
            text("MEDIA_RENDER_VOICE_FIT_GAP_SECONDS", "0.1"),
            float,
            lambda v: VOICE_FIT_GAP_RANGE[0] <= v < VOICE_FIT_GAP_RANGE[1],
            f"seconds from {VOICE_FIT_GAP_RANGE[0]:g} up to {VOICE_FIT_GAP_RANGE[1]:g}",
        ),
        preview_width=positive_int("MEDIA_RENDER_PREVIEW_WIDTH", 540),
        preview_max_frames=positive_int("MEDIA_RENDER_PREVIEW_MAX_FRAMES", 12),
        preview_reel_fps=positive_int("MEDIA_RENDER_PREVIEW_REEL_FPS", 2),
        preview_timeout_seconds=positive_int("MEDIA_RENDER_PREVIEW_TIMEOUT_SECONDS", 120),
        still_max_frames=positive_int("MEDIA_RENDER_STILL_MAX_FRAMES", 10),
    )
