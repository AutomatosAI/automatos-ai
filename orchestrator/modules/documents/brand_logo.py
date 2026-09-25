"""Workspace brand logo and logo mark — upload, storage, and render-time inlining (PRD-242 S3, PRD-251 D5).

PRD-167 S4 exposed ``brand_kit.logo_url`` as a *public https URL* only: the
WeasyPrint fetcher (PRD-156 S4) refuses every non-public host and every
non-http scheme, and the DOCX image fetch refuses redirects — so a logo living
in the platform's own object store (MinIO on ``localhost:9000`` locally, a
private bucket in SaaS) could never be rendered, and a non-technical user had
no way to upload one at all.

The logo is now a stored file, referenced by a storage-relative ``logo_path``
(``<workspace_id>/brand/logo.png``) on the brand kit:

* uploaded through ``POST /api/documents/brand-kit/logo`` (PNG/JPEG only —
  python-docx cannot embed SVG, and one accepted set keeps PDF and DOCX honest),
* written under ``config.DOCUMENT_STORAGE_DIR`` (the same root the DOCX
  renderer's path-confined reader trusts) and mirrored to S3 like generated
  documents (Railway containers are ephemeral),
* served back to the UI by ``GET /api/documents/brand-kit/logo`` (local file,
  then S3 — the ``serve_generated_file`` pattern),
* **inlined as a ``data:`` URI at render time** (:func:`brand_kit_for_render`)
  so neither renderer needs network access: the WeasyPrint fetcher already
  admits ``data:``, and the DOCX renderer decodes it to bytes.

PRD-251 D5 adds the logo MARK, a square mark separate from the wordmark: the
same upload rules plus a square shape, stored at ``<workspace_id>/brand/logo-mark.png``
(``logo_mark_path``) and served by ``/api/documents/brand-kit/logo-mark``. The
brand files share one store (:func:`store_brand_file`, :func:`load_brand_file`,
:func:`delete_brand_file`): the logo, the mark, and the font files
(``modules/documents/brand_fonts.py``).

Nothing here reads ``os.getenv`` — every setting comes through ``config``.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

from config import config
from core.storage import ensure_bucket, get_s3_client, is_storage_configured

logger = logging.getLogger(__name__)

# The UI routes that stream the stored logo and logo mark (mounted under /api/documents).
BRAND_LOGO_ROUTE = "/api/documents/brand-kit/logo"
BRAND_LOGO_MARK_ROUTE = "/api/documents/brand-kit/logo-mark"

MAX_LOGO_BYTES = 2 * 1024 * 1024  # 2 MB — a letterhead logo, not a poster
# The PDF renderer rasterises the logo in-process; a tiny file can still declare
# a huge canvas (a "pixel flood"). Dimensions are read from the header and capped.
MAX_LOGO_DIMENSION = 4096  # px, either axis
# PRD-251 D5: the mark is square; its longer side may be at most this much longer.
MAX_LOGO_MARK_ASPECT = 1.2

# The stored names under <workspace_id>/brand/.
LOGO_STEM = "logo"
LOGO_MARK_STEM = "logo-mark"

# Accepted image types, keyed by the magic bytes we sniff (never trust the
# declared Content-Type). SVG is deliberately absent: python-docx cannot embed it.
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png", ".png"),
    (b"\xff\xd8\xff", "image/jpeg", ".jpg"),
)
_MIME_BY_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


class BrandLogoError(ValueError):
    """The upload was refused (type, size, or shape). Message is user-facing."""


def sniff_image_type(data: bytes) -> Optional[tuple[str, str]]:
    """``(mime, extension)`` for a PNG/JPEG payload, else ``None``."""
    for magic, mime, ext in _MAGIC:
        if data[: len(magic)] == magic:
            return mime, ext
    return None


def _png_dimensions(data: bytes) -> Optional[tuple[int, int]]:
    # IHDR is mandatory and first: 8-byte signature, 4-byte length, "IHDR", width, height.
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return (width, height) if width and height else None


_JPEG_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _jpeg_dimensions(data: bytes) -> Optional[tuple[int, int]]:
    # Walk the marker segments to the first SOFn: length(2) precision(1) height(2) width(2).
    i = 2
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:  # standalone markers
            i += 2
            continue
        if marker == 0xD9 or marker == 0xDA:  # EOI / start of scan before any SOF
            return None
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        if length < 2:
            return None
        if marker in _JPEG_SOF_MARKERS:
            if i + 9 > n:
                return None
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            return (width, height) if width and height else None
        i += 2 + length
    return None


def image_dimensions(data: bytes) -> Optional[tuple[int, int]]:
    """``(width, height)`` from a PNG/JPEG header, or ``None`` if unreadable. Pure."""
    sniffed = sniff_image_type(data)
    if sniffed is None:
        return None
    return _png_dimensions(data) if sniffed[0] == "image/png" else _jpeg_dimensions(data)


def _storage_root() -> Path:
    return Path(config.DOCUMENT_STORAGE_DIR).resolve()


def _confined_local_path(path: str) -> Optional[Path]:
    """Absolute path under the storage root, or ``None`` if the path escapes it."""
    if not path:
        return None
    root = _storage_root()
    candidate = (root / path.lstrip("/\\")).resolve()
    if root != candidate and root not in candidate.parents:
        logger.warning("[BrandLogo] refusing path outside storage root: %r", path)
        return None
    return candidate


def s3_brand_file_key(path: str) -> str:
    return f"workspaces/{path}"


def logo_storage_path(workspace_id: UUID, ext: str, stem: str = LOGO_STEM) -> str:
    return f"{workspace_id}/brand/{stem}{ext}"


# ------------------------------------------------------------------
# The brand file store: the logo, the logo mark and the font files
# ------------------------------------------------------------------


def store_brand_file(path: str, data: bytes, mime: str) -> None:
    """Write a brand file under the storage root and mirror it to S3.

    The S3 mirror is best-effort (local serving covers the container's lifetime).
    """
    local = _confined_local_path(path)
    if local is None:  # pragma: no cover — a UUID and a fixed name can't escape the root
        raise BrandLogoError("Invalid brand file path.")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    _mirror_to_s3(path, data, mime)


def _mirror_to_s3(path: str, data: bytes, mime: str) -> bool:
    if not is_storage_configured():
        return False
    bucket = config.S3_DOCUMENTS_BUCKET or "automatos-ai"
    try:
        ensure_bucket(bucket)
        get_s3_client().put_object(Bucket=bucket, Key=s3_brand_file_key(path), Body=data, ContentType=mime)
        return True
    except Exception:  # noqa: BLE001 — persistence mirror only; local serving still works
        logger.exception("[BrandLogo] S3 mirror failed for %s", path)
        return False


def load_brand_file(path: str, max_bytes: int) -> Optional[bytes]:
    """A brand file's bytes from the local store, then S3; ``None`` when neither has it.

    A copy in object storage larger than ``max_bytes`` is not read.
    """
    local = _confined_local_path(path)
    if local is None:
        return None
    if local.is_file():
        try:
            return local.read_bytes()
        except OSError:
            logger.warning("[BrandLogo] unreadable local brand file %s", local)
    if not is_storage_configured():
        return None
    bucket = config.S3_DOCUMENTS_BUCKET or "automatos-ai"
    try:
        body = get_s3_client().get_object(Bucket=bucket, Key=s3_brand_file_key(path))["Body"].read()
    except Exception:  # noqa: BLE001 — missing object / storage hiccup → no file
        logger.info("[BrandLogo] brand file %s not in object storage", path)
        return None
    if len(body) > max_bytes:
        return None
    try:  # re-warm the local cache so the next render skips the round-trip
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(body)
    except OSError:
        pass
    return body


def delete_brand_file(path: str) -> None:
    """Remove a stored brand file everywhere (best-effort; never raises)."""
    local = _confined_local_path(path)
    if local is not None and local.exists():
        try:
            local.unlink()
        except OSError:
            logger.warning("[BrandLogo] could not delete local brand file %s", local)
    if is_storage_configured():
        try:
            bucket = config.S3_DOCUMENTS_BUCKET or "automatos-ai"
            get_s3_client().delete_object(Bucket=bucket, Key=s3_brand_file_key(path))
        except Exception:  # noqa: BLE001
            logger.info("[BrandLogo] S3 delete skipped for %s", path)


# ------------------------------------------------------------------
# The logo and the logo mark
# ------------------------------------------------------------------


def _checked_image(data: bytes, what: str) -> tuple[str, str, tuple[int, int]]:
    """``(mime, extension, (width, height))`` of an acceptable upload, else :class:`BrandLogoError`."""
    if not data:
        raise BrandLogoError("The uploaded file is empty.")
    if len(data) > MAX_LOGO_BYTES:
        raise BrandLogoError(f"{what} must be {MAX_LOGO_BYTES // (1024 * 1024)} MB or smaller.")
    sniffed = sniff_image_type(data)
    if sniffed is None:
        raise BrandLogoError(f"{what} must be a PNG or JPEG image (SVG is not supported in DOCX output).")
    mime, ext = sniffed
    dims = image_dimensions(data)
    if dims is None:
        raise BrandLogoError("The image header could not be read — is the file a complete PNG or JPEG?")
    if max(dims) > MAX_LOGO_DIMENSION:
        raise BrandLogoError(f"{what} must be at most {MAX_LOGO_DIMENSION}×{MAX_LOGO_DIMENSION} px (got {dims[0]}×{dims[1]}).")
    return mime, ext, dims


def _store_image(workspace_id: UUID, stem: str, data: bytes, mime: str, ext: str) -> str:
    path = logo_storage_path(workspace_id, ext, stem)
    store_brand_file(path, data, mime)
    # A previous upload with the other extension must not linger as a stale twin.
    local = _confined_local_path(path)
    for stale_ext in (".png", ".jpg"):
        stale = local.parent / f"{stem}{stale_ext}" if local is not None else None
        if stale is not None and stale_ext != ext and stale.exists():
            stale.unlink()
    return path


def save_brand_logo(workspace_id: UUID, data: bytes) -> str:
    """Validate and persist an uploaded logo; return its storage-relative path.

    Raises :class:`BrandLogoError` with a user-facing message on a bad upload.
    The S3 mirror is best-effort (local serving covers the container's lifetime).
    """
    mime, ext, _ = _checked_image(data, "Logo")
    return _store_image(workspace_id, LOGO_STEM, data, mime, ext)


def save_brand_logo_mark(workspace_id: UUID, data: bytes) -> str:
    """Validate and persist an uploaded logo mark (PRD-251 D5): a logo upload that is square."""
    mime, ext, (width, height) = _checked_image(data, "The logo mark")
    if max(width, height) > MAX_LOGO_MARK_ASPECT * min(width, height):
        raise BrandLogoError(
            f"The logo mark must be square: the symbol without the name (got {width}×{height} px). "
            "Upload the wide version as the logo."
        )
    return _store_image(workspace_id, LOGO_MARK_STEM, data, mime, ext)


def load_brand_logo(logo_path: str) -> Optional[bytes]:
    """Logo (or logo mark) bytes from the local store, then S3; ``None`` when neither has it."""
    return load_brand_file(logo_path, MAX_LOGO_BYTES)


def delete_brand_logo(logo_path: str) -> None:
    """Remove the stored logo (or logo mark) everywhere (best-effort; never raises)."""
    delete_brand_file(logo_path)


def logo_mime(logo_path: str) -> str:
    return _MIME_BY_EXT.get(os.path.splitext(logo_path)[1].lower(), "application/octet-stream")


def logo_data_uri(logo_path: str) -> Optional[str]:
    data = load_brand_logo(logo_path) if logo_path else None
    if not data:
        return None
    return f"data:{logo_mime(logo_path)};base64,{base64.b64encode(data).decode('ascii')}"


def brand_kit_for_render(kit: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of the kit whose ``logo_url`` the renderers can actually load.

    An uploaded logo (``logo_path``) becomes an inline ``data:`` URI — allowed by
    the WeasyPrint fetcher and decoded by the DOCX renderer — so a private object
    store never has to be reachable from the render path. An external
    ``logo_url`` is left as-is (the fetchers keep their public-host rules).
    A social render also needs the logo mark and the fonts:
    ``modules.documents.brand_fonts.brand_kit_for_media_render``.
    """
    if not isinstance(kit, dict):
        return kit
    logo_path = kit.get("logo_path") or ""
    if not logo_path:
        return dict(kit)
    inline = logo_data_uri(logo_path)
    if inline is None:
        # Stored path but no bytes anywhere: fall back to the external URL, if any.
        return dict(kit)
    return {**kit, "logo_url": inline}


__all__ = [
    "BRAND_LOGO_MARK_ROUTE",
    "BRAND_LOGO_ROUTE",
    "MAX_LOGO_BYTES",
    "MAX_LOGO_DIMENSION",
    "MAX_LOGO_MARK_ASPECT",
    "image_dimensions",
    "BrandLogoError",
    "brand_kit_for_render",
    "delete_brand_file",
    "delete_brand_logo",
    "load_brand_file",
    "load_brand_logo",
    "logo_data_uri",
    "logo_mime",
    "logo_storage_path",
    "s3_brand_file_key",
    "save_brand_logo",
    "save_brand_logo_mark",
    "sniff_image_type",
    "store_brand_file",
]
