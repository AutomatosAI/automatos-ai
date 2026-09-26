"""Workspace brand fonts, and the brand kit a social render reads (PRD-251 D5, S1.3).

A font file is uploaded through ``POST /api/documents/brand-kit/fonts`` with the
face it provides (family name, weight, style) and is stored like the logo
(``modules/documents/brand_logo.py``): under ``config.DOCUMENT_STORAGE_DIR`` at
``<workspace_id>/brand/fonts/<id>.woff2``, mirrored to S3, streamed back by
``GET /api/documents/brand-kit/fonts/{id}``. The kit lists it in ``font_files``,
which only the upload and delete routes write. Uploading a face the kit already
has (same family, weight and style) replaces that file.

Only woff2 is accepted (D5), checked by its header, never by the declared type:
the signature, a complete file, one face (a collection is refused) and at least
one table. The size and count caps keep a render bundle, which carries every
brand file inline, inside media-render's limit.

At render time :func:`brand_kit_for_media_render` inlines every uploaded brand
file as a data: URI: the logo, the logo mark (into ``logo_mark_url``) and the
fonts (``font_files`` becomes ``[{family, weight, style, data_uri}]``), the
shapes ``core/media_render_bundle.py`` reads. A render reads only what its
bundle carries (D9), so the fonts never have to be fetched.

Storage settings come through ``config``; nothing here reads the environment.
"""

from __future__ import annotations

import base64
import logging
import struct
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from core.media_render_bundle import FONT_FAMILY, FONT_STYLES
from modules.documents.brand_kit import FONT_WEIGHT_VALUES, MAX_FONT_FILE_NAME_CHARS, MAX_FONT_FILES, BrandFontFile
from modules.documents.brand_logo import (
    brand_kit_for_render,
    delete_brand_file,
    load_brand_file,
    logo_data_uri,
    store_brand_file,
)

logger = logging.getLogger(__name__)

# The UI route that lists, streams and removes the stored fonts (mounted under /api/documents).
BRAND_FONTS_ROUTE = "/api/documents/brand-kit/fonts"

WOFF2_MIME = "font/woff2"
WOFF2_SIGNATURE = b"wOF2"
WOFF2_HEADER_BYTES = 48
# The sfnt a woff2 wraps: TrueType outlines (0x00010000) or CFF ("OTTO").
WOFF2_FLAVORS = frozenset({0x00010000, 0x4F54544F})
WOFF2_COLLECTION_FLAVOR = 0x74746366  # "ttcf"
MAX_FONT_BYTES = 2 * 1024 * 1024  # 2 MB: a full Latin face is 20-200 KB in woff2
MAX_FONT_FAMILY_CHARS = 64


class BrandFontError(ValueError):
    """The upload was refused (type, size, face or count). Message is user-facing."""


def check_woff2(data: bytes) -> None:
    """Refuse anything but one complete woff2 face within the size cap."""
    if not data:
        raise BrandFontError("The uploaded file is empty.")
    if len(data) > MAX_FONT_BYTES:
        raise BrandFontError(f"A font file must be {MAX_FONT_BYTES // (1024 * 1024)} MB or smaller.")
    if data[:4] != WOFF2_SIGNATURE:
        raise BrandFontError("A font file must be a woff2 font: convert a TTF or OTF file to woff2 first.")
    if len(data) < WOFF2_HEADER_BYTES:
        raise BrandFontError("The woff2 header is incomplete: is the file a complete woff2 font?")
    flavor, length, num_tables = struct.unpack(">IIH", data[4:14])
    if length != len(data):
        raise BrandFontError(f"The woff2 file is incomplete: its header says {length} bytes, the upload has {len(data)}.")
    if flavor == WOFF2_COLLECTION_FLAVOR:
        raise BrandFontError("A font collection is not supported: upload one face per file.")
    if flavor not in WOFF2_FLAVORS:
        raise BrandFontError("The woff2 file does not hold a TrueType or OpenType font.")
    if num_tables == 0:
        raise BrandFontError("The woff2 file holds no font tables.")


def check_face(family: str, weight: int, style: str) -> tuple[str, int, str]:
    """The face an upload provides, trimmed; :class:`BrandFontError` naming what is wrong."""
    family, style = family.strip(), style.strip().lower()
    if not FONT_FAMILY.match(family):
        raise BrandFontError(
            f"The font family name must be letters, digits, spaces, hyphens or underscores, "
            f"{MAX_FONT_FAMILY_CHARS} characters at most, e.g. Brand Sans."
        )
    if weight not in FONT_WEIGHT_VALUES:
        raise BrandFontError("The font weight must be 100 to 900, in hundreds (400 is regular, 700 bold).")
    if style not in FONT_STYLES:
        raise BrandFontError(f"The font style must be {' or '.join(sorted(FONT_STYLES))}.")
    return family, weight, style


def font_storage_path(workspace_id: UUID, font_id: str) -> str:
    return f"{workspace_id}/brand/fonts/{font_id}.woff2"


def _same_face(font: Dict[str, Any], family: str, weight: int, style: str) -> bool:
    return (str(font.get("family", "")).casefold(), font.get("weight"), font.get("style")) == (family.casefold(), weight, style)


def add_brand_font(
    workspace_id: UUID,
    fonts: List[Dict[str, Any]],
    data: bytes,
    *,
    family: str,
    weight: int,
    style: str,
    file_name: str = "",
) -> List[Dict[str, Any]]:
    """Store an uploaded woff2 and return the kit's new ``font_files``.

    A face the kit already has is replaced (its old file is removed once the new
    one is stored); a new face must fit under :data:`MAX_FONT_FILES`.
    """
    family, weight, style = check_face(family, weight, style)
    check_woff2(data)
    replaced = [font for font in fonts if _same_face(font, family, weight, style)]
    kept = [font for font in fonts if not _same_face(font, family, weight, style)]
    if len(kept) >= MAX_FONT_FILES:
        raise BrandFontError(f"A brand kit holds {MAX_FONT_FILES} font files at most: remove one first.")
    font_id = uuid4().hex
    entry = BrandFontFile(
        id=font_id,
        family=family,
        weight=weight,
        style=style,
        path=font_storage_path(workspace_id, font_id),
        file_name="".join(ch for ch in file_name if ch.isprintable()).strip()[:MAX_FONT_FILE_NAME_CHARS],
        bytes=len(data),
    ).model_dump()
    store_brand_file(entry["path"], data, WOFF2_MIME)
    for font in replaced:
        delete_brand_file(font.get("path") or "")
    return [*kept, entry]


def find_brand_font(fonts: List[Dict[str, Any]], font_id: str) -> Optional[Dict[str, Any]]:
    return next((font for font in fonts if font.get("id") == font_id), None)


def remove_brand_font(fonts: List[Dict[str, Any]], font_id: str) -> Optional[List[Dict[str, Any]]]:
    """The kit's ``font_files`` without ``font_id`` (its file removed), or ``None`` if the kit has no such font."""
    font = find_brand_font(fonts, font_id)
    if font is None:
        return None
    delete_brand_file(font.get("path") or "")
    return [other for other in fonts if other is not font]


def load_brand_font(font: Dict[str, Any]) -> Optional[bytes]:
    return load_brand_file(font.get("path") or "", MAX_FONT_BYTES)


def brand_kit_for_media_render(kit: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of the kit with every uploaded brand file inline, as a social render takes it.

    The logo as :func:`brand_kit_for_render` inlines it, an uploaded logo mark
    in ``logo_mark_url``, and ``font_files`` as ``[{family, weight, style,
    data_uri}]``. A stored file whose bytes are gone is left out (a font) or
    left as the kit had it (a mark); the template's fallbacks apply.
    """
    rendered = brand_kit_for_render(kit)
    mark_path = kit.get("logo_mark_path") or ""
    mark = logo_data_uri(mark_path) if mark_path else None
    fonts = []
    for font in kit.get("font_files") or []:
        data = load_brand_font(font)
        if not data:
            logger.warning("[BrandFonts] font file %s has no stored bytes; the render goes without it", font.get("id"))
            continue
        fonts.append({
            "family": font.get("family"),
            "weight": font.get("weight"),
            "style": font.get("style"),
            "data_uri": f"data:{WOFF2_MIME};base64,{base64.b64encode(data).decode('ascii')}",
        })
    return {**rendered, **({"logo_mark_url": mark} if mark else {}), "font_files": fonts}


__all__ = [
    "BRAND_FONTS_ROUTE",
    "MAX_FONT_BYTES",
    "WOFF2_MIME",
    "BrandFontError",
    "add_brand_font",
    "brand_kit_for_media_render",
    "check_face",
    "check_woff2",
    "find_brand_font",
    "font_storage_path",
    "load_brand_font",
    "remove_brand_font",
]
