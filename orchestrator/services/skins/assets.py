"""PRD-259 §3 — a skin package's asset files: which paths may exist, and what their bytes must be.

A package holds ``skin.json`` and, under ``assets/``, only:

* ``assets/preview.png``: a PNG, at most ``SKIN_MAX_PREVIEW_KB``;
* ``assets/fonts/<name>.woff2``: woff2 by its bytes, at most ``SKIN_MAX_FONT_KB`` each;
* ``assets/mascot/<state>.(webm|mp4|png)`` for the 12 mascot states: sniffed by their
  bytes, at most ``SKIN_MAX_MASCOT_MB`` together.

The type is decided by the path and proven by the bytes, never taken from the
package; the checked type is what the assets route serves.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, NamedTuple, Optional, Tuple

from config import config
from modules.documents.brand_fonts import WOFF2_MIME, WOFF2_SIGNATURE, BrandFontError, check_woff2
from services.skins.fields import SkinError, skin_error
from services.skins.schema import FONT_FILE_PATTERN, MASCOT_STATES, PREVIEW_PATH

KB = 1024
MB = 1024 * 1024

PNG_MIME = "image/png"
WEBM_MIME = "video/webm"
MP4_MIME = "video/mp4"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
WEBM_SIGNATURE = b"\x1a\x45\xdf\xa3"  # the EBML header
MP4_BRAND_OFFSET = 4
MP4_BRAND_BOX = b"ftyp"

MASCOT_FORMATS: Tuple[str, ...] = ("webm", "mp4", "png")
MASCOT_FILE_PATTERN = re.compile(
    rf"assets/mascot/({'|'.join(MASCOT_STATES)})\.({'|'.join(MASCOT_FORMATS)})"
)

# Every type the assets route may serve: what the checks below produce, nothing else.
ASSET_TYPES: Tuple[str, ...] = (WOFF2_MIME, PNG_MIME, WEBM_MIME, MP4_MIME)


class AssetKind(NamedTuple):
    """One kind of asset file: how to recognise its path and its size cap.

    ``shared_cap`` means the cap holds for all the kind's files together (the mascot),
    not for each one.
    """

    name: str
    matches: Callable[[str], bool]
    cap_bytes: Callable[[], int]
    shared_cap: bool
    too_large_code: str
    too_large: Callable[[], str]


def _is_mp4(data: bytes) -> bool:
    return data[MP4_BRAND_OFFSET:MP4_BRAND_OFFSET + len(MP4_BRAND_BOX)] == MP4_BRAND_BOX


MASCOT_SNIFF: Dict[str, Tuple[Callable[[bytes], bool], str]] = {
    "webm": (lambda data: data.startswith(WEBM_SIGNATURE), WEBM_MIME),
    "mp4": (_is_mp4, MP4_MIME),
    "png": (lambda data: data.startswith(PNG_SIGNATURE), PNG_MIME),
}

PREVIEW = AssetKind(
    "preview", lambda path: path == PREVIEW_PATH, lambda: config.SKIN_MAX_PREVIEW_KB * KB, False,
    "PREVIEW_TOO_LARGE", lambda: f"The preview image must be {config.SKIN_MAX_PREVIEW_KB} KB or smaller.",
)
FONT = AssetKind(
    "font", lambda path: FONT_FILE_PATTERN.fullmatch(path) is not None, lambda: config.SKIN_MAX_FONT_KB * KB, False,
    "FONT_TOO_LARGE", lambda: f"Each font file must be {config.SKIN_MAX_FONT_KB} KB or smaller.",
)
MASCOT = AssetKind(
    "mascot", lambda path: MASCOT_FILE_PATTERN.fullmatch(path) is not None, lambda: config.SKIN_MAX_MASCOT_MB * MB,
    True, "MASCOT_TOO_LARGE", lambda: f"The mascot files together must be {config.SKIN_MAX_MASCOT_MB} MB or smaller.",
)
ASSET_KINDS: Tuple[AssetKind, ...] = (PREVIEW, FONT, MASCOT)


def asset_kind(path: str) -> Optional[AssetKind]:
    """The kind of asset ``path`` names, or None when no asset may live there."""
    return next((kind for kind in ASSET_KINDS if kind.matches(path)), None)


def too_large_error(kind: AssetKind, path: str) -> SkinError:
    """The refusal for a file over its kind's cap."""
    return skin_error(kind.too_large_code, path, kind.too_large())


def mascot_paths() -> Tuple[str, ...]:
    """Every file a 'custom' mascot needs: each state in each format."""
    return tuple(f"assets/mascot/{state}.{fmt}" for state in MASCOT_STATES for fmt in MASCOT_FORMATS)


def _font_type(path: str, data: bytes) -> Tuple[Optional[str], Optional[SkinError]]:
    if not data.startswith(WOFF2_SIGNATURE):
        return None, skin_error("NOT_WOFF2", path, f"{path} is not a woff2 font: convert it to woff2 first.")
    try:
        check_woff2(data)
    except BrandFontError as exc:
        return None, skin_error("BAD_FONT_FILE", path, f"{path}: {exc}")
    return WOFF2_MIME, None


def _preview_type(path: str, data: bytes) -> Tuple[Optional[str], Optional[SkinError]]:
    if not data.startswith(PNG_SIGNATURE):
        return None, skin_error("NOT_PNG", path, f"{path} must be a PNG image.")
    return PNG_MIME, None


def _mascot_type(path: str, data: bytes) -> Tuple[Optional[str], Optional[SkinError]]:
    fmt = path.rsplit(".", 1)[-1]
    sniff, mime = MASCOT_SNIFF[fmt]
    if not sniff(data):
        return None, skin_error("BAD_MASCOT_FILE", path, f"{path} is not a {fmt} file by its contents.")
    return mime, None


_TYPE_CHECKS: Dict[str, Callable[[str, bytes], Tuple[Optional[str], Optional[SkinError]]]] = {
    FONT.name: _font_type,
    PREVIEW.name: _preview_type,
    MASCOT.name: _mascot_type,
}


def checked_type(kind: AssetKind, path: str, data: bytes) -> Tuple[Optional[str], Optional[SkinError]]:
    """The content type the bytes prove for ``path``, or the error saying why they don't."""
    return _TYPE_CHECKS[kind.name](path, data)
