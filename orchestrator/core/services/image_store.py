"""
Image Store Service
====================

Uploads generated images (base64) to the platform object store (S3 / MinIO
via ``core.storage``, PRD-233 S4) and serves them back by id.

S3 key pattern: generated-images/{workspace_id}/{uuid}.{ext}
Pointer:        generated-image-pointers/{uuid}  → that key (PRD-251 S0.4c)

The public route knows only the id. Every save also writes a small pointer
object, so resolving an id is ONE GET — never a listing. (A pointer object, not
a lookup row: no table, no migration, and the public route needs no database
session.) Ids saved before pointers existed fall back to a PAGINATED listing —
every page, never just the first 1000 keys — and the pointer is then written,
so the next lookup is one GET too. Bodies stream, optionally one byte range
(``get_object(Range=...)``); nothing reads a whole object into memory.
"""

import asyncio
import base64
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterator, Optional
from uuid import UUID, uuid4

from config import config
from core.storage import ensure_bucket, get_s3_client

logger = logging.getLogger(__name__)

MIME_TO_EXT = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/svg+xml": "svg",
}

IMAGE_KEY_PREFIX = "generated-images"
POINTER_KEY_PREFIX = "generated-image-pointers"
DEFAULT_WORKSPACE_SEGMENT = "default"
DEFAULT_CONTENT_TYPE = "image/png"
POINTER_CONTENT_TYPE = "text/plain; charset=utf-8"
LIST_PAGE_SIZE = 1000  # S3's per-page maximum; the listing walks every page.
WORKSPACE_LIST_PAGE_SIZE = 5

_MISSING_KEY_CODES = frozenset({"NoSuchKey", "404", "NotFound"})
_INVALID_RANGE_CODE = "InvalidRange"
# One range, RFC 9110 form: bytes=a-b, bytes=a- or bytes=-n.
_SINGLE_BYTE_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class ImageRangeNotSatisfiable(Exception):
    """The requested byte range starts past the end of the image."""

    def __init__(self, size: int):
        super().__init__(f"range not satisfiable for a {size}-byte image")
        self.size = size


@dataclass(frozen=True)
class ImageStream:
    """An image body opened for streaming — the whole object or one range."""

    body: Iterator[bytes]
    content_type: str
    content_length: int
    content_range: Optional[str] = None  # "bytes a-b/size" when a range was served


def parse_byte_range(header: Optional[str]) -> Optional[str]:
    """The ``Range`` header as S3 takes it, or ``None`` to serve the whole body.

    One range only; a malformed, reversed or multi-range header is ignored (the
    server may ignore Range — RFC 9110 §14.2), so the caller gets a 200.
    """
    if not header:
        return None
    match = _SINGLE_BYTE_RANGE.match(header.strip())
    if not match:
        return None
    start, end = match.groups()
    if not start and not end:
        return None
    if start and end and int(start) > int(end):
        return None
    return f"bytes={start}-{end}"


def _is_image_id(image_id: str) -> bool:
    """Only the canonical uuid4 strings save_image mints — anything else (a
    fragment, a path) never reaches the bucket."""
    try:
        return str(UUID(image_id)) == image_id
    except (ValueError, TypeError, AttributeError):
        return False


def _pointer_key(image_id: str) -> str:
    return f"{POINTER_KEY_PREFIX}/{image_id}"


def _key_image_id(key: str) -> str:
    """generated-images/{ws}/{id}.{ext} → id (exact, never a substring match)."""
    return key.rsplit("/", 1)[-1].split(".", 1)[0]


def _error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None) or {}
    return str(response.get("Error", {}).get("Code", ""))


def _iter_body(body: Any, chunk_size: int) -> Iterator[bytes]:
    try:
        yield from body.iter_chunks(chunk_size)
    finally:
        body.close()


class S3ImageStore:
    """Image store on the platform object store (S3 in SaaS, MinIO locally)."""

    def __init__(self):
        self.bucket = config.S3_DOCUMENTS_BUCKET
        logger.info("Image store: S3 (bucket=%s)", self.bucket)

    @property
    def client(self):
        """The process-wide S3 client — lazy, no network at construction."""
        return get_s3_client()

    async def save_image(
        self,
        base64_data: str,
        mime_type: str = "image/png",
        workspace_id: Optional[str] = None,
    ) -> str:
        ext = MIME_TO_EXT.get(mime_type, "png")
        image_id = str(uuid4())
        ws = workspace_id or DEFAULT_WORKSPACE_SEGMENT
        key = f"{IMAGE_KEY_PREFIX}/{ws}/{image_id}.{ext}"
        image_bytes = base64.b64decode(base64_data)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: ensure_bucket(self.bucket))
        await loop.run_in_executor(
            None,
            lambda: self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=image_bytes,
                ContentType=mime_type,
            ),
        )
        # After the image, so a pointer never names a missing object. A failed
        # pointer write leaves the image findable by the legacy listing, which
        # writes the pointer on first lookup.
        await loop.run_in_executor(None, lambda: self._write_pointer(image_id, key))
        logger.info("Saved image to S3: %s (%d bytes)", key, len(image_bytes))
        return image_id

    async def resolve_key(self, image_id: str, workspace_id: Optional[str] = None) -> Optional[str]:
        """The image's object key: one GET of its pointer, or the legacy listing."""
        if not _is_image_id(image_id):
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._resolve_key(image_id, workspace_id))

    async def open_image(
        self,
        image_id: str,
        byte_range: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> Optional[ImageStream]:
        """Open an image for streaming — the whole body, or ``byte_range``
        (``parse_byte_range``'s form). ``None`` when the id resolves to nothing;
        raises :class:`ImageRangeNotSatisfiable` for a range past the end."""
        key = await self.resolve_key(image_id, workspace_id)
        if key is None:
            return None
        loop = asyncio.get_running_loop()
        request = {"Bucket": self.bucket, "Key": key}
        if byte_range:
            request["Range"] = byte_range
        try:
            obj = await loop.run_in_executor(None, lambda: self.client.get_object(**request))
        except Exception as exc:
            code = _error_code(exc)
            if code == _INVALID_RANGE_CODE:
                size = await loop.run_in_executor(None, lambda: self._object_size(key))
                raise ImageRangeNotSatisfiable(size) from exc
            if code in _MISSING_KEY_CODES:
                logger.warning("Image %s resolves to %s, which is gone", image_id, key)
                return None
            raise
        return ImageStream(
            body=_iter_body(obj["Body"], config.GENERATED_IMAGE_STREAM_CHUNK_BYTES),
            content_type=obj.get("ContentType") or DEFAULT_CONTENT_TYPE,
            content_length=int(obj["ContentLength"]),
            content_range=obj.get("ContentRange") if byte_range else None,
        )

    # ── synchronous S3 steps (run in the executor) ─────────────────────────

    def _resolve_key(self, image_id: str, workspace_id: Optional[str]) -> Optional[str]:
        key = self._read_pointer(image_id)
        if key is not None:
            return key
        key = self._find_legacy_key(image_id, workspace_id)
        if key is not None:
            self._write_pointer(image_id, key)
        return key

    def _read_pointer(self, image_id: str) -> Optional[str]:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=_pointer_key(image_id))
        except Exception as exc:
            if _error_code(exc) in _MISSING_KEY_CODES:
                return None
            raise
        body = obj["Body"]
        try:
            key = body.read().decode("utf-8").strip()
        finally:
            body.close()
        if _key_image_id(key) != image_id:
            logger.warning("Ignoring pointer for image %s: it names %r", image_id, key)
            return None
        return key

    def _write_pointer(self, image_id: str, key: str) -> None:
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=_pointer_key(image_id),
                Body=key.encode("utf-8"),
                ContentType=POINTER_CONTENT_TYPE,
            )
        except Exception as exc:
            logger.warning("Could not write the pointer for image %s (%s): %s", image_id, key, exc)

    def _find_legacy_key(self, image_id: str, workspace_id: Optional[str]) -> Optional[str]:
        """An id saved before pointers existed: the caller's workspace first (one
        small listing), then every page of the whole prefix."""
        searches = []
        if workspace_id:
            searches.append((f"{IMAGE_KEY_PREFIX}/{workspace_id}/{image_id}.", WORKSPACE_LIST_PAGE_SIZE))
        searches.append((f"{IMAGE_KEY_PREFIX}/", LIST_PAGE_SIZE))
        for prefix, page_size in searches:
            for key in self._iter_keys(prefix, page_size):
                if _key_image_id(key) == image_id:
                    return key
        return None

    def _iter_keys(self, prefix: str, page_size: int) -> Iterator[str]:
        token: Optional[str] = None
        while True:
            request = {"Bucket": self.bucket, "Prefix": prefix, "MaxKeys": page_size}
            if token:
                request["ContinuationToken"] = token
            page = self.client.list_objects_v2(**request)
            for item in page.get("Contents") or []:
                yield item["Key"]
            token = page.get("NextContinuationToken")
            if not page.get("IsTruncated") or not token:
                return

    def _object_size(self, key: str) -> int:
        return int(self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"])


# ======================================================================
# Factory
# ======================================================================

_image_store = None


def get_image_store():
    """Get or create the image store singleton."""
    global _image_store
    if _image_store is None:
        _image_store = S3ImageStore()
    return _image_store
