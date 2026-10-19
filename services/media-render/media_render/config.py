"""media-render's environment seam (PRD-251 D3).

The service is its own image and never sees the orchestrator's ``config.py``.
Every environment read in the service happens here, so call sites never read
``os.environ`` inline (the orchestrator's config discipline, and the worker's
``worker_config.py``). Defaults describe the image's own layout (Dockerfile).
Every limit and every timeout the service applies is a setting below.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, TypeVar

from .espeak_build import LANGUAGES
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
# PRD-264 EUAI-08a: the label's font goes into an ffmpeg filter as it is, so it is a plain
# absolute path: nothing in it can be read as filter syntax (ai_label.py).
FONT_PATH = re.compile(r"^/[A-Za-z0-9/._-]+$")

# PRD-264 EUAI-06b (D15): the algorithms c2pa signs with; the key must be of the matching type.
C2PA_ALGORITHMS = ("es256", "es384", "es512", "ps256", "ps384", "ps512", "ed25519")
C2PA_CERT_ENV = "MEDIA_RENDER_C2PA_SIGNING_CERT_PEM"
C2PA_KEY_ENV = "MEDIA_RENDER_C2PA_SIGNING_KEY_PEM"
TSA_SCHEMES = ("https://", "http://")
# A per-install certificate is made again 30 days before it ends (signing_keys.py), so it lasts longer.
C2PA_LOCAL_CERT_MIN_DAYS = 60
ESCAPED_NEWLINE = "\\n"

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
    # PRD-264 EUAI-06a (D14): the XMP writer and reader of the AI mark (provenance.py).
    exiftool_bin: str
    # PRD-264 EUAI-08a: the font of a video's visible "AI-generated" label (ai_label.py).
    ai_label_font_path: str
    render_quality: str
    render_fps: int
    render_workers: str
    render_timeout_seconds: int
    check_timeout_seconds: int
    mix_timeout_seconds: int
    probe_timeout_seconds: int
    fetch_timeout_seconds: int
    mark_timeout_seconds: int
    # How long burning the AI-generated label into a video may take (a re-encode).
    label_timeout_seconds: int
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
    # One workspace's unfinished jobs; past this, 429 for that workspace alone, so
    # one workspace's burst never holds every slot above (P251W1-RVW-4).
    max_active_jobs_per_workspace: int
    # What a 503 or a 429 tells the caller to wait before it submits again.
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
    # PRD-264 EUAI-06b (D15): the Content Credentials signer (signing_keys.py). The hosted
    # service is given a certificate chain and its key (Railway secrets; never logged, so
    # never in a repr); without them, this install makes its own in ``c2pa_data_dir``.
    c2pa_cert_pem: str = field(repr=False)
    c2pa_key_pem: str = field(repr=False)
    c2pa_alg: str
    c2pa_tsa_url: str
    c2pa_data_dir: str
    c2pa_local_cert_days: int

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


@dataclass(frozen=True)
class _Env:
    """One environment mapping; every refusal names its variable."""

    source: Mapping[str, str]

    def text(self, name: str, default: str) -> str:
        return (self.source.get(name) or "").strip() or default

    def positive_int(self, name: str, default: int) -> int:
        return _parse(name, self.text(name, str(default)), int, lambda v: v > 0, "a positive whole number")

    def prefixes(self, name: str) -> Tuple[UrlPrefix, ...]:
        try:
            return parse_prefixes(self.source.get(name) or "")
        except ValueError as exc:
            raise ConfigError(f"{name}: {exc}") from None


def _service(env: _Env) -> Dict[str, Any]:
    """Where the service listens, and the token it asks for."""
    return {
        "environment": env.text("ENVIRONMENT", "development"),
        "bind_host": env.text("MEDIA_RENDER_BIND_HOST", "0.0.0.0"),
        "port": _parse(
            "MEDIA_RENDER_PORT", env.text("MEDIA_RENDER_PORT", "8090"), int, lambda v: 0 < v < 65536, "a TCP port"
        ),
        "internal_token": (env.source.get(TOKEN_ENV) or "").strip(),
    }


def _image(env: _Env) -> Dict[str, Any]:
    """The image's own files and programs (the Dockerfile's layout), and where jobs work."""
    return {
        "espeak_data_path": env.text("MEDIA_RENDER_ESPEAK_DATA_PATH", "/opt/espeak"),
        "kokoro_model_path": env.text("MEDIA_RENDER_KOKORO_MODEL", "/opt/kokoro/kokoro-v1.0.onnx"),
        "kokoro_voices_path": env.text("MEDIA_RENDER_KOKORO_VOICES", "/opt/kokoro/voices-v1.0.bin"),
        "gsap_path": env.text("MEDIA_RENDER_GSAP_PATH", "/opt/media-render/vendor/gsap.min.js"),
        "versions_path": env.text("MEDIA_RENDER_VERSIONS_PATH", "/opt/media-render/versions.json"),
        "browser_path": env.text("HYPERFRAMES_BROWSER_PATH", "/opt/chrome/chrome-headless-shell"),
        "hyperframes_bin": env.text("MEDIA_RENDER_HYPERFRAMES_BIN", "hyperframes"),
        "ffmpeg_bin": env.text("MEDIA_RENDER_FFMPEG_BIN", "ffmpeg"),
        "ffprobe_bin": env.text("MEDIA_RENDER_FFPROBE_BIN", "ffprobe"),
        "exiftool_bin": env.text("MEDIA_RENDER_EXIFTOOL_BIN", "exiftool"),
        "ai_label_font_path": _parse(
            "MEDIA_RENDER_AI_LABEL_FONT",
            env.text("MEDIA_RENDER_AI_LABEL_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            str,
            lambda v: bool(FONT_PATH.match(v)),
            "an absolute path of letters, digits, '/', '.', '_' and '-'",
        ),
        "work_dir": env.text("MEDIA_RENDER_WORK_DIR", "/tmp/media-render"),
        "music_dir": env.text("MEDIA_RENDER_MUSIC_DIR", "/opt/media-render/music"),
    }


def _rendering(env: _Env) -> Dict[str, Any]:
    """How a render runs, and how long each of its steps may take."""
    return {
        "render_quality": _parse(
            "MEDIA_RENDER_QUALITY",
            env.text("MEDIA_RENDER_QUALITY", "delivery"),
            str,
            lambda v: v in RENDER_QUALITIES,
            "one of " + ", ".join(sorted(RENDER_QUALITIES)),
        ),
        "render_fps": _parse(
            "MEDIA_RENDER_FPS",
            env.text("MEDIA_RENDER_FPS", "30"),
            int,
            lambda v: v in RENDER_FPS,
            "one of " + ", ".join(str(f) for f in sorted(RENDER_FPS)),
        ),
        "render_workers": _parse(
            "MEDIA_RENDER_WORKERS", env.text("MEDIA_RENDER_WORKERS", "auto"), _workers, bool, "'auto' or a positive number"
        ),
        "render_timeout_seconds": env.positive_int("MEDIA_RENDER_RENDER_TIMEOUT_SECONDS", 900),
        "check_timeout_seconds": env.positive_int("MEDIA_RENDER_CHECK_TIMEOUT_SECONDS", 300),
        "mix_timeout_seconds": env.positive_int("MEDIA_RENDER_MIX_TIMEOUT_SECONDS", 120),
        "probe_timeout_seconds": env.positive_int("MEDIA_RENDER_PROBE_TIMEOUT_SECONDS", 60),
        "fetch_timeout_seconds": env.positive_int("MEDIA_RENDER_FETCH_TIMEOUT_SECONDS", 120),
        "mark_timeout_seconds": env.positive_int("MEDIA_RENDER_MARK_TIMEOUT_SECONDS", 60),
        "label_timeout_seconds": env.positive_int("MEDIA_RENDER_LABEL_TIMEOUT_SECONDS", 300),
    }


def _admission(env: _Env) -> Dict[str, Any]:
    """How many jobs run, wait and are held at once, overall and per workspace, and for how long."""
    return {
        "max_concurrent_renders": env.positive_int("MEDIA_RENDER_MAX_CONCURRENT_RENDERS", 2),
        "max_renders_per_workspace": env.positive_int("MEDIA_RENDER_MAX_RENDERS_PER_WORKSPACE", 1),
        "max_concurrent_checks": env.positive_int("MEDIA_RENDER_MAX_CONCURRENT_CHECKS", 1),
        "max_checks_per_workspace": env.positive_int("MEDIA_RENDER_MAX_CHECKS_PER_WORKSPACE", 1),
        "max_active_jobs": env.positive_int("MEDIA_RENDER_MAX_ACTIVE_JOBS", 20),
        "max_active_jobs_per_workspace": env.positive_int("MEDIA_RENDER_MAX_ACTIVE_JOBS_PER_WORKSPACE", 4),
        "busy_retry_after_seconds": env.positive_int("MEDIA_RENDER_BUSY_RETRY_AFTER_SECONDS", 30),
        "job_ttl_seconds": env.positive_int("MEDIA_RENDER_JOB_TTL_SECONDS", 3600),
        "sweep_interval_seconds": env.positive_int("MEDIA_RENDER_SWEEP_INTERVAL_SECONDS", 60),
    }


def _bundle_limits(env: _Env) -> Dict[str, Any]:
    """What one bundle, and one voice request, may carry."""
    return {
        "media_url_prefixes": env.prefixes("MEDIA_RENDER_MEDIA_URL_PREFIXES"),
        "max_bundle_bytes": env.positive_int("MEDIA_RENDER_MAX_BUNDLE_BYTES", 32 * MEBIBYTE),
        "max_asset_bytes": env.positive_int("MEDIA_RENDER_MAX_ASSET_BYTES", 8 * MEBIBYTE),
        "max_media_bytes": env.positive_int("MEDIA_RENDER_MAX_MEDIA_BYTES", 256 * MEBIBYTE),
        "max_files": env.positive_int("MEDIA_RENDER_MAX_FILES", 32),
        "max_duration_seconds": env.positive_int("MEDIA_RENDER_MAX_DURATION_SECONDS", 180),
        "max_variables": env.positive_int("MEDIA_RENDER_MAX_VARIABLES", 200),
        "max_variable_chars": env.positive_int("MEDIA_RENDER_MAX_VARIABLE_CHARS", 2000),
        "max_brand_tokens": env.positive_int("MEDIA_RENDER_MAX_BRAND_TOKENS", 64),
        "tts_max_lines": env.positive_int("MEDIA_RENDER_TTS_MAX_LINES", 40),
        "tts_max_chars": env.positive_int("MEDIA_RENDER_TTS_MAX_CHARS", 500),
    }


def _voice(env: _Env) -> Dict[str, Any]:
    """Kokoro's defaults, and how a spoken line is fitted to its script window."""
    return {
        "kokoro_voice": env.text("MEDIA_RENDER_KOKORO_VOICE", "af_heart"),
        "kokoro_speed": _parse(
            "MEDIA_RENDER_KOKORO_SPEED",
            env.text("MEDIA_RENDER_KOKORO_SPEED", "0.95"),
            float,
            lambda v: 0.5 <= v <= 2.0,
            "a speed between 0.5 and 2.0",
        ),
        # PRD-262 US-052: the image's espeak-ng data speaks these alone (espeak_build.py).
        "kokoro_lang": _parse(
            "MEDIA_RENDER_KOKORO_LANG",
            env.text("MEDIA_RENDER_KOKORO_LANG", "en-us"),
            str,
            lambda v: v in LANGUAGES,
            "one of " + ", ".join(LANGUAGES),
        ),
        "voice_max_tempo": _parse(
            "MEDIA_RENDER_VOICE_MAX_TEMPO",
            env.text("MEDIA_RENDER_VOICE_MAX_TEMPO", "1.25"),
            float,
            lambda v: VOICE_TEMPO_RANGE[0] <= v <= VOICE_TEMPO_RANGE[1],
            f"a tempo from {VOICE_TEMPO_RANGE[0]:g} to {VOICE_TEMPO_RANGE[1]:g}",
        ),
        "voice_fit_gap_seconds": _parse(
            "MEDIA_RENDER_VOICE_FIT_GAP_SECONDS",
            env.text("MEDIA_RENDER_VOICE_FIT_GAP_SECONDS", "0.1"),
            float,
            lambda v: VOICE_FIT_GAP_RANGE[0] <= v < VOICE_FIT_GAP_RANGE[1],
            f"seconds from {VOICE_FIT_GAP_RANGE[0]:g} up to {VOICE_FIT_GAP_RANGE[1]:g}",
        ),
    }


def _frames(env: _Env) -> Dict[str, Any]:
    """Previews (US-106) and stills (US-107)."""
    return {
        "preview_width": env.positive_int("MEDIA_RENDER_PREVIEW_WIDTH", 540),
        "preview_max_frames": env.positive_int("MEDIA_RENDER_PREVIEW_MAX_FRAMES", 12),
        "preview_reel_fps": env.positive_int("MEDIA_RENDER_PREVIEW_REEL_FPS", 2),
        "preview_timeout_seconds": env.positive_int("MEDIA_RENDER_PREVIEW_TIMEOUT_SECONDS", 120),
        "still_max_frames": env.positive_int("MEDIA_RENDER_STILL_MAX_FRAMES", 10),
    }


def _pem(env: _Env, name: str) -> str:
    """A PEM setting: a value pasted on one line, its newlines written ``\\n``, is unfolded."""
    raw = (env.source.get(name) or "").strip()
    if ESCAPED_NEWLINE in raw and "\n" not in raw:
        raw = raw.replace(ESCAPED_NEWLINE, "\n")
    return raw + "\n" if raw else ""


def _signing(env: _Env) -> Dict[str, Any]:
    """The Content Credentials signer (PRD-264 EUAI-06b, D15): a configured chain and key, or this install's own."""
    cert, key = _pem(env, C2PA_CERT_ENV), _pem(env, C2PA_KEY_ENV)
    if bool(cert) != bool(key):
        raise ConfigError(f"{C2PA_CERT_ENV} and {C2PA_KEY_ENV} are set together or not at all")
    return {
        "c2pa_cert_pem": cert,
        "c2pa_key_pem": key,
        "c2pa_alg": _parse(
            "MEDIA_RENDER_C2PA_SIGNING_ALG",
            env.text("MEDIA_RENDER_C2PA_SIGNING_ALG", "es256"),
            str,
            lambda v: v in C2PA_ALGORITHMS,
            "one of " + ", ".join(C2PA_ALGORITHMS),
        ),
        "c2pa_tsa_url": _parse(
            "MEDIA_RENDER_C2PA_TSA_URL",
            env.text("MEDIA_RENDER_C2PA_TSA_URL", ""),
            str,
            lambda v: v == "" or v.startswith(TSA_SCHEMES),
            "an http(s) URL of a timestamp authority, or unset",
        ),
        "c2pa_data_dir": env.text("MEDIA_RENDER_C2PA_DATA_DIR", "/var/lib/media-render/c2pa"),
        "c2pa_local_cert_days": _parse(
            "MEDIA_RENDER_C2PA_LOCAL_CERT_DAYS",
            env.text("MEDIA_RENDER_C2PA_LOCAL_CERT_DAYS", "3650"),
            int,
            lambda v: v >= C2PA_LOCAL_CERT_MIN_DAYS,
            f"a whole number of days, at least {C2PA_LOCAL_CERT_MIN_DAYS}",
        ),
    }


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Read the settings from ``env`` (the process environment by default).

    Raises ConfigError naming the variable when a value cannot be used, so a
    misconfigured container fails at boot rather than on its first render.
    """
    read = _Env(os.environ if env is None else env)
    # Unpacked one by one, so a field two sections both set is a TypeError, not a silent override.
    return Settings(
        **_service(read),
        **_image(read),
        **_rendering(read),
        **_admission(read),
        **_bundle_limits(read),
        **_voice(read),
        **_frames(read),
        **_signing(read),
    )
