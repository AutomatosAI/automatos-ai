"""The composition bundle POST /render takes (PRD-251 S1.1b).

    {
      "workspace_id": "…",                        one render at a time per workspace
      "reference": "post id, for logs",           optional
      "composition": {"html": "<!doctype html>…", "css": "…"},
      "variables": {"headline": "…", "scene2_at": 2.4},
      "brand": {"tokens": {"bg": "#1a1714", "heading-font": "\\"Geist\\", sans-serif"},
                "fonts": [{"family": "Geist", "weight": 700, "path": "assets/brand/geist-700.woff2"}]},
      "files": [{"path": "assets/brand/logo.svg", "data_uri": "data:image/svg+xml;base64,…"}],
      "media": [{"path": "assets/cine/hook.mp4", "url": "<presigned GET on our storage>"}],
      "audio": {"voice": {"voice": "af_heart", "speed": 0.95,
                          "lines": [{"id": "l01", "at": 0.3, "text": "…"},
                                    {"id": "l02", "at": 2.4, "path": "assets/vo/l02.wav"}]},
                "music": {"track": "<library id>", "start": 32.0},
                "sfx": [{"path": "assets/sfx/click.ogg", "at": 10.44, "volume": 0.5}]},
      "preview": {"at": [1.2, 10.0, 36.5]},     optional: snapshot these moments, not the full render
      "still": {"at": [0.5, 1.5]}               optional: an image, one full-size PNG per moment
    }

``files`` are inlined as data: URIs (the brand kit's logo and font files, small
generated charts). ``media`` are fetched from our storage when the job runs, and
only from the allowlisted prefixes; ``parse_bundle`` refuses any other URL
before anything is fetched. Voice lines are Kokoro text, or a file from a
workspace's voice toolkit already copied into our storage. Nothing here is
generated: the renderer assembles (D3).

A ``preview`` asks for frames instead of the video (US-106): the job is
staged, spoken, mixed and checked exactly as a render is, then snapshots the
composition at each moment and returns them as small PNGs, with a short reel
of them (pipeline.RenderPipeline.preview).

A ``still`` is an image (US-107): the job is staged, mixed and checked the same
way, then its render is a full-size PNG of the composition at each moment
(pipeline.RenderPipeline.still): one for a card, one per slide for a carousel.
A still has no sound, so it takes no audio plan, and it is its own preview.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

from . import audio, validate
from .composition import VARIABLE_NAME, Composition, FontFace, brand_stylesheet, build_document, inspect_document
from .config import Settings
from .media_urls import redact, url_allowed
from .music import Track
from .validate import BundleError

ASSETS_ROOT = "assets/"
# The renderer stages GSAP and the mix here; a bundle never writes into them.
RESERVED_DIRS = ("assets/vendor/", "assets/audio/")

MEDIA_TYPES: Dict[str, FrozenSet[str]] = {
    "svg": frozenset({"image/svg+xml"}),
    "png": frozenset({"image/png"}),
    "jpg": frozenset({"image/jpeg"}),
    "jpeg": frozenset({"image/jpeg"}),
    "webp": frozenset({"image/webp"}),
    "gif": frozenset({"image/gif"}),
    "woff2": frozenset({"font/woff2", "application/font-woff2", "application/octet-stream"}),
    "woff": frozenset({"font/woff", "application/font-woff", "application/octet-stream"}),
    "ttf": frozenset({"font/ttf", "font/sfnt", "application/x-font-ttf", "application/octet-stream"}),
    "otf": frozenset({"font/otf", "font/sfnt", "application/x-font-opentype", "application/octet-stream"}),
}
FONT_EXTENSIONS = frozenset({"woff2", "woff", "ttf", "otf"})
AUDIO_EXTENSIONS = frozenset({"wav", "mp3", "ogg", "oga", "opus", "m4a", "aac", "flac"})
MEDIA_EXTENSIONS = frozenset({"mp4", "mov", "webm", "m4v", "png", "jpg", "jpeg", "webp", "gif", "svg"}) | AUDIO_EXTENSIONS

WORKSPACE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
TOKEN_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
# A token is one CSS value: nothing that ends the declaration, opens a block,
# fetches (url(), @import) or closes the <style> element.
TOKEN_VALUE_FORBIDDEN = re.compile(r"[;{}<>\\\n\r]|/\*|url\(|@import|expression\(", re.IGNORECASE)
FONT_FAMILY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")
FONT_WEIGHTS = frozenset({"normal", "bold"} | {str(weight) for weight in range(100, 1000, 100)})
FONT_STYLES = frozenset({"normal", "italic"})
# Field lengths, part of the contract; how MANY of each a bundle carries is config.
MAX_TOKEN_CHARS = 200
MAX_REFERENCE_CHARS = 200


@dataclass(frozen=True)
class InlineFile:
    path: str
    data: bytes


@dataclass(frozen=True)
class MediaInput:
    path: str
    url: str


@dataclass(frozen=True)
class VoiceLine:
    id: str
    at: float
    text: Optional[str] = None
    path: Optional[str] = None


@dataclass(frozen=True)
class VoicePlan:
    voice: str
    speed: float
    lang: str
    lines: Tuple[VoiceLine, ...]


@dataclass(frozen=True)
class MusicCue:
    track: str
    path: Path
    start: float
    fade_in: float
    fade_out: float
    # The library's word on the track (music.Track.report): its title, licence
    # and attribution, which the job report carries for the post's credit line.
    about: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SfxCue:
    path: str
    at: float
    volume: float


@dataclass(frozen=True)
class AudioPlan:
    voice: Optional[VoicePlan] = None
    music: Optional[MusicCue] = None
    sfx: Tuple[SfxCue, ...] = ()


@dataclass(frozen=True)
class Preview:
    at: Tuple[float, ...]


@dataclass(frozen=True)
class Still:
    at: Tuple[float, ...]


@dataclass(frozen=True)
class Bundle:
    workspace_id: str
    reference: Optional[str]
    composition: Composition
    files: Tuple[InlineFile, ...]
    media: Tuple[MediaInput, ...]
    audio: AudioPlan
    preview: Optional[Preview] = None
    still: Optional[Still] = None


def _file_path(value: Any, where: str, extensions: FrozenSet[str]) -> str:
    path = validate.asset_path(value, where, root=ASSETS_ROOT, extensions=extensions)
    if path.startswith(RESERVED_DIRS):
        raise BundleError(f"{where} is inside {' or '.join(RESERVED_DIRS)}, which the renderer stages itself")
    return path


def _files(raw: Any, settings: Settings) -> Tuple[InlineFile, ...]:
    files = []
    for i, entry in enumerate(validate.items(raw or [], "files", limit=settings.max_files)):
        where = f"files[{i}]"
        validate.keys(validate.mapping(entry, where), where, required=("path", "data_uri"))
        path = _file_path(entry["path"], f"{where}.path", frozenset(MEDIA_TYPES))
        mimes = MEDIA_TYPES[path.rsplit(".", 1)[-1].lower()]
        data = validate.data_uri(entry["data_uri"], f"{where}.data_uri", mimes=mimes, max_bytes=settings.max_asset_bytes)
        files.append(InlineFile(path=path, data=data))
    return tuple(files)


def _media(raw: Any, settings: Settings) -> Tuple[MediaInput, ...]:
    """Every media URL is checked against the storage allowlist here, before any fetch."""
    media = []
    for i, entry in enumerate(validate.items(raw or [], "media", limit=settings.max_files)):
        where = f"media[{i}]"
        validate.keys(validate.mapping(entry, where), where, required=("path", "url"))
        path = _file_path(entry["path"], f"{where}.path", MEDIA_EXTENSIONS)
        url = entry["url"]
        if not isinstance(url, str) or not url_allowed(url, settings.media_url_prefixes):
            shown = redact(url) if isinstance(url, str) else repr(url)
            raise BundleError(f"{where} ({path}): {shown} is not on the storage allowlist; media come only from our storage")
        media.append(MediaInput(path=path, url=url))
    return tuple(media)


def _brand(raw: Any, file_paths: FrozenSet[str], settings: Settings) -> str:
    brand = validate.mapping(raw or {}, "brand")
    validate.keys(brand, "brand", optional=("tokens", "fonts"))
    tokens = validate.mapping(brand.get("tokens") or {}, "brand.tokens")
    if len(tokens) > settings.max_brand_tokens:
        raise BundleError(f"brand.tokens holds {len(tokens)} tokens; the limit is {settings.max_brand_tokens}")
    for name, value in tokens.items():
        validate.pattern(name, f"brand.tokens name {name!r}", TOKEN_NAME, "lowercase words joined by '-'")
        validate.text(value, f"brand.tokens.{name}", max_chars=MAX_TOKEN_CHARS)
        if TOKEN_VALUE_FORBIDDEN.search(value):
            raise BundleError(f"brand.tokens.{name} must be a single CSS value")
    fonts = []
    for i, entry in enumerate(validate.items(brand.get("fonts") or [], "brand.fonts", limit=settings.max_brand_tokens)):
        where = f"brand.fonts[{i}]"
        validate.keys(validate.mapping(entry, where), where, required=("family", "path"), optional=("weight", "style"))
        family = validate.pattern(entry["family"], f"{where}.family", FONT_FAMILY, "a font family name")
        weight, style = str(entry.get("weight", 400)), str(entry.get("style", "normal"))
        if weight not in FONT_WEIGHTS or style not in FONT_STYLES:
            raise BundleError(f"{where} needs a weight of 100-900 (or normal/bold) and a style of normal or italic")
        path = entry["path"]
        if not isinstance(path, str) or path not in file_paths or path.rsplit(".", 1)[-1].lower() not in FONT_EXTENSIONS:
            raise BundleError(f"{where}.path must name a font file (woff2, woff, ttf, otf) inlined in files")
        fonts.append(FontFace(family=family, weight=weight, style=style, path=path))
    return brand_stylesheet(tokens, fonts)


def _composition(payload: Mapping[str, Any], brand_css: str, known: FrozenSet[str], settings: Settings) -> Composition:
    composition = validate.mapping(payload["composition"], "composition")
    validate.keys(composition, "composition", required=("html",), optional=("css",))
    document = validate.text(composition["html"], "composition.html", max_chars=settings.max_bundle_bytes)
    css = validate.text(composition.get("css", ""), "composition.css", max_chars=settings.max_bundle_bytes, allow_empty=True)
    variables = validate.mapping(payload.get("variables") or {}, "variables")
    if len(variables) > settings.max_variables:
        raise BundleError(f"variables holds {len(variables)} entries; the limit is {settings.max_variables}")
    for name, value in variables.items():
        validate.pattern(name, f"variable name {name!r}", VARIABLE_NAME, "letters, digits and _ (dotted paths allowed)")
        if isinstance(value, str):
            validate.text(value, f"variables.{name}", max_chars=settings.max_variable_chars, allow_empty=True)
        elif not isinstance(value, bool):
            validate.number(value, f"variables.{name}")
    filled = build_document(document, css, variables, brand_css)
    return inspect_document(filled, known_paths=known, max_duration=settings.max_duration_seconds)


def _audio_input(value: Any, where: str, audio_paths: FrozenSet[str]) -> str:
    if not isinstance(value, str) or value not in audio_paths:
        raise BundleError(f"{where} must name an audio file listed in media")
    return value


def _voice(raw: Any, duration: float, audio_paths: FrozenSet[str], settings: Settings) -> Optional[VoicePlan]:
    if raw is None:
        return None
    voice = validate.mapping(raw, "audio.voice")
    validate.keys(voice, "audio.voice", required=("lines",), optional=("voice", "speed", "lang"))
    lines: List[VoiceLine] = []
    for i, entry in enumerate(validate.items(voice["lines"], "audio.voice.lines", limit=settings.tts_max_lines)):
        where = f"audio.voice.lines[{i}]"
        validate.keys(validate.mapping(entry, where), where, required=("id", "at"), optional=("text", "path"))
        line_id = validate.line_id(entry["id"], f"{where}.id")
        if any(line.id == line_id for line in lines):
            raise BundleError(f"{where}.id {line_id} is used twice")
        at = validate.number(entry["at"], f"{where}.at", minimum=0, below=duration)
        if ("text" in entry) == ("path" in entry):
            raise BundleError(f"{where} needs either text (spoken by Kokoro) or path (a voice file in media)")
        if "text" in entry:
            text = validate.text(entry["text"], f"{where}.text", max_chars=settings.tts_max_chars)
            lines.append(VoiceLine(id=line_id, at=at, text=text))
        else:
            lines.append(VoiceLine(id=line_id, at=at, path=_audio_input(entry["path"], f"{where}.path", audio_paths)))
    return VoicePlan(
        voice=validate.voice_name(voice.get("voice", settings.kokoro_voice), "audio.voice.voice"),
        speed=validate.speed(voice.get("speed", settings.kokoro_speed), "audio.voice.speed"),
        lang=validate.language(voice.get("lang", settings.kokoro_lang), "audio.voice.lang"),
        lines=tuple(lines),
    )


def _music(raw: Any, duration: float, library: Mapping[str, Track]) -> Optional[MusicCue]:
    if raw is None:
        return None
    music = validate.mapping(raw, "audio.music")
    validate.keys(music, "audio.music", required=("track",), optional=("start", "fade_in", "fade_out"))
    track = library.get(music["track"]) if isinstance(music["track"], str) else None
    if track is None:
        raise BundleError(f"audio.music.track {music['track']!r} is not in the music library")
    start = validate.number(music.get("start", 0), "audio.music.start", minimum=0)
    fade_in = validate.number(music.get("fade_in", audio.MUSIC_FADE_IN_SECONDS), "audio.music.fade_in", minimum=0)
    fade_out = validate.number(music.get("fade_out", audio.MUSIC_FADE_OUT_SECONDS), "audio.music.fade_out", minimum=0)
    if fade_in + fade_out > duration:
        raise BundleError("audio.music fades last longer than the composition")
    if track.duration is not None and start + duration > track.duration:
        raise BundleError(f"audio.music: {track.id} lasts {track.duration:g} s; a window from {start:g} s runs past its end")
    return MusicCue(track=track.id, path=track.path, start=start, fade_in=fade_in, fade_out=fade_out, about=track.report())


def _sfx(raw: Any, duration: float, audio_paths: FrozenSet[str], settings: Settings) -> Tuple[SfxCue, ...]:
    cues = []
    for i, entry in enumerate(validate.items(raw or [], "audio.sfx", limit=settings.max_files)):
        where = f"audio.sfx[{i}]"
        validate.keys(validate.mapping(entry, where), where, required=("path", "at"), optional=("volume",))
        path = _audio_input(entry["path"], f"{where}.path", audio_paths)
        at = validate.number(entry["at"], f"{where}.at", minimum=0, below=duration)
        volume = validate.number(entry.get("volume", audio.SFX_DEFAULT_VOLUME), f"{where}.volume", minimum=0, maximum=1)
        cues.append(SfxCue(path=path, at=at, volume=volume))
    return tuple(cues)


def _audio(raw: Any, duration: float, audio_paths: FrozenSet[str], settings: Settings, library: Mapping[str, Track]) -> AudioPlan:
    plan = validate.mapping(raw or {}, "audio")
    validate.keys(plan, "audio", optional=("voice", "music", "sfx"))
    return AudioPlan(
        voice=_voice(plan.get("voice"), duration, audio_paths, settings),
        music=_music(plan.get("music"), duration, library),
        sfx=_sfx(plan.get("sfx"), duration, audio_paths, settings),
    )


def _preview(raw: Any, duration: float, settings: Settings) -> Optional[Preview]:
    """The moments a preview snapshots: each inside the composition, each once, in order."""
    if raw is None:
        return None
    preview = validate.mapping(raw, "preview")
    validate.keys(preview, "preview", required=("at",))
    moments = validate.items(preview["at"], "preview.at", limit=settings.preview_max_frames)
    if not moments:
        raise BundleError("preview.at must name at least one moment")
    at = [validate.number(t, f"preview.at[{i}]", minimum=0, below=duration) for i, t in enumerate(moments)]
    return Preview(at=tuple(sorted(set(at))))


def _still(raw: Any, duration: float, settings: Settings) -> Optional[Still]:
    """The moments an image is taken at: each inside the composition, in time order, each once."""
    if raw is None:
        return None
    still = validate.mapping(raw, "still")
    validate.keys(still, "still", required=("at",))
    moments = validate.items(still["at"], "still.at", limit=settings.still_max_frames)
    if not moments:
        raise BundleError("still.at must name at least one moment")
    at = [validate.number(t, f"still.at[{i}]", minimum=0, below=duration) for i, t in enumerate(moments)]
    if any(later <= earlier for earlier, later in zip(at, at[1:])):
        raise BundleError("still.at must list its moments in time order, each once")
    return Still(at=tuple(at))


def parse_bundle(payload: Any, settings: Settings, library: Mapping[str, Track]) -> Bundle:
    """Validate a render bundle. Raises BundleError (HTTP 400) before anything is fetched."""
    body = validate.mapping(payload, "the bundle")
    validate.keys(
        body,
        "the bundle",
        required=("workspace_id", "composition"),
        optional=("reference", "variables", "brand", "files", "media", "audio", "preview", "still"),
    )
    if body.get("still") is not None:
        if body.get("preview") is not None:
            raise BundleError("a still is its own preview: send still or preview, not both")
        if body.get("audio"):
            raise BundleError("a still has no sound: leave audio out")
    workspace_id = validate.pattern(body["workspace_id"], "workspace_id", WORKSPACE_ID, "a workspace id")
    reference = body.get("reference")
    if reference is not None:
        reference = validate.text(reference, "reference", max_chars=MAX_REFERENCE_CHARS)
    files = _files(body.get("files"), settings)
    media = _media(body.get("media"), settings)
    paths = [item.path for item in files] + [item.path for item in media]
    duplicates = sorted({path for path in paths if paths.count(path) > 1})
    if duplicates:
        raise BundleError(f"a path is used by more than one file: {', '.join(duplicates)}")
    brand_css = _brand(body.get("brand"), frozenset(item.path for item in files), settings)
    composition = _composition(body, brand_css, frozenset(paths), settings)
    audio_paths = frozenset(item.path for item in media if item.path.rsplit(".", 1)[-1].lower() in AUDIO_EXTENSIONS)
    return Bundle(
        workspace_id=workspace_id,
        reference=reference,
        composition=composition,
        files=files,
        media=media,
        audio=_audio(body.get("audio"), composition.duration, audio_paths, settings, library),
        preview=_preview(body.get("preview"), composition.duration, settings),
        still=_still(body.get("still"), composition.duration, settings),
    )
