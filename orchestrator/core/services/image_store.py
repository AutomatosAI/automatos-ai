"""
Image Store Service
====================

Uploads generated images (base64) to the platform object store (S3 / MinIO
via ``core.storage``, PRD-233 S4) and serves them back by id.

S3 key pattern: generated-images/{workspace_id}/{uuid}.{ext}
Pointer:        generated-image-pointers/{uuid}  → that key (PRD-251 S0.4c)
Legacy index:   generated-image-pointers/legacy-index.json  (P251-RVW-3)

The public route knows only the id. Every save also writes a small pointer
object, so resolving an id is ONE GET — never a listing. (A pointer object, not
a lookup row: no table, no migration, and the public route needs no database
session.) A save whose pointer cannot be written fails, so every image saved
since pointers exist has one.

Ids saved before pointers existed are a closed set. They are found through the
legacy index: {id: key} for every image in the bucket when it was built. It is
built once, by the first process that needs it, which walks EVERY page of the
prefix, and it is kept as one object that every later process loads with one
GET. The build is single-flight: concurrent misses wait for the one walk. So
the prefix is walked once, not once per unknown id. After that, an unknown id
costs one pointer GET and no listing, and a legacy id's first lookup writes its
pointer. Bodies stream, optionally one byte range (``get_object(Range=...)``);
no image is read whole into memory.
"""

import asyncio
import base64
import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional
from uuid import UUID, uuid4

from config import config
from core.storage import ensure_bucket, get_s3_client

logger = logging.getLogger(__name__)

# F179 (C): the store keeps only what its callers save, the raster images and SVG
# (a Composio output can be one), and refuses anything else. A raster image is
# served inline; anything else, SVG included, as an attachment.
MIME_TO_EXT = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/svg+xml": "svg",
}
RASTER_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
_TYPE_ALIASES = {"image/jpg": "image/jpeg"}

IMAGE_KEY_PREFIX = "generated-images"
IMAGE_ROUTE = "/api/generated-images"
POINTER_KEY_PREFIX = "generated-image-pointers"
DEFAULT_WORKSPACE_SEGMENT = "default"
DEFAULT_CONTENT_TYPE = "image/png"
POINTER_CONTENT_TYPE = "text/plain; charset=utf-8"
# Never a pointer key: pointers are named by canonical uuids only.
LEGACY_INDEX_KEY = f"{POINTER_KEY_PREFIX}/legacy-index.json"
LEGACY_INDEX_CONTENT_TYPE = "application/json"
LIST_PAGE_SIZE = 1000  # S3's per-page maximum; the listing walks every page.
WORKSPACE_LIST_PAGE_SIZE = 5

_MISSING_KEY_CODES = frozenset({"NoSuchKey", "404", "NotFound"})
_INVALID_RANGE_CODE = "InvalidRange"
# One range, RFC 9110 form: bytes=a-b, bytes=a- or bytes=-n.
_SINGLE_BYTE_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class UnstorableImageType(ValueError):
    """save_image was given a type the public store does not keep (F179)."""


def stored_type(mime_type: str) -> str:
    """The type ``mime_type`` is stored as; UnstorableImageType outside MIME_TO_EXT."""
    requested = str(mime_type or "").strip().lower()
    canonical = _TYPE_ALIASES.get(requested, requested)
    if canonical not in MIME_TO_EXT:
        raise UnstorableImageType(f"{mime_type!r} is not an image type the public store keeps")
    return canonical


def served_inline(content_type: Optional[str]) -> bool:
    """A raster image is shown inline; anything else, an older object too, downloads."""
    base = str(content_type or "").split(";", 1)[0].strip().lower()
    return _TYPE_ALIASES.get(base, base) in RASTER_IMAGE_TYPES


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


def image_extension(mime_type: str) -> str:
    """The file extension an image of ``mime_type`` is saved with."""
    return MIME_TO_EXT.get(mime_type, "png")


def image_key(image_id: str, mime_type: str, workspace_id: Optional[str] = None) -> str:
    """generated-images/{ws}/{id}.{ext}: where :meth:`S3ImageStore.save_image`
    puts an image (an image Deliverable's file_path names it too, PRD-251 US-117)."""
    ws = workspace_id or DEFAULT_WORKSPACE_SEGMENT
    return f"{IMAGE_KEY_PREFIX}/{ws}/{image_id}.{image_extension(mime_type)}"


def generated_image_path(image_id: str) -> str:
    """The app route that serves a saved image by id (api/generated_images.py)."""
    return f"{IMAGE_ROUTE}/{image_id}"


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
        # The legacy index ({id: key}), loaded or built once per process.
        self._legacy_index: Optional[Dict[str, str]] = None
        self._legacy_index_lock = threading.Lock()
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
        mime_type = stored_type(mime_type)
        image_id = str(uuid4())
        key = image_key(image_id, mime_type, workspace_id)
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
        # After the image, so a pointer never names a missing object. The pointer
        # is the only way to reach an image saved since pointers exist (the legacy
        # index is built once), so a failed pointer write fails the save.
        await loop.run_in_executor(None, lambda: self._put_pointer(image_id, key))
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

    def _put_pointer(self, image_id: str, key: str) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=_pointer_key(image_id),
            Body=key.encode("utf-8"),
            ContentType=POINTER_CONTENT_TYPE,
        )

    def _write_pointer(self, image_id: str, key: str) -> None:
        """A legacy id's pointer, best effort: the index still resolves it."""
        try:
            self._put_pointer(image_id, key)
        except Exception as exc:
            logger.warning("Could not write the pointer for image %s (%s): %s", image_id, key, exc)

    def _find_legacy_key(self, image_id: str, workspace_id: Optional[str]) -> Optional[str]:
        """An id with no pointer: saved before pointers existed, or unknown. Until
        this process holds the legacy index, the caller's workspace is tried first
        (one small listing). Then the index answers, with no listing once it is
        held."""
        if self._legacy_index is None and workspace_id:
            prefix = f"{IMAGE_KEY_PREFIX}/{workspace_id}/{image_id}."
            for key in self._iter_keys(prefix, WORKSPACE_LIST_PAGE_SIZE):
                if _key_image_id(key) == image_id:
                    return key
        return self._legacy_keys().get(image_id)

    def _legacy_keys(self) -> Dict[str, str]:
        """The legacy index. It is loaded (one GET) or, when no process has built
        it yet, built by walking the whole prefix and then stored. Single-flight:
        concurrent callers wait for the one load or walk. A failure leaves nothing
        held, so the next miss tries again."""
        index = self._legacy_index
        if index is not None:
            return index
        with self._legacy_index_lock:
            if self._legacy_index is None:
                loaded = self._load_legacy_index()
                self._legacy_index = loaded if loaded is not None else self._build_legacy_index()
            return self._legacy_index

    def _load_legacy_index(self) -> Optional[Dict[str, str]]:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=LEGACY_INDEX_KEY)
        except Exception as exc:
            if _error_code(exc) in _MISSING_KEY_CODES:
                return None
            raise
        body = obj["Body"]
        try:
            raw = body.read()
        finally:
            body.close()
        try:
            entries = json.loads(raw)
        except ValueError:
            entries = None
        if not isinstance(entries, dict):
            logger.warning("The legacy image index %s is unreadable; rebuilding it", LEGACY_INDEX_KEY)
            return None
        return {
            image_id: key for image_id, key in entries.items()
            if isinstance(key, str) and _key_image_id(key) == image_id
        }

    def _build_legacy_index(self) -> Dict[str, str]:
        index = {_key_image_id(key): key for key in self._iter_keys(f"{IMAGE_KEY_PREFIX}/", LIST_PAGE_SIZE)}
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=LEGACY_INDEX_KEY,
                Body=json.dumps(index).encode("utf-8"),
                ContentType=LEGACY_INDEX_CONTENT_TYPE,
            )
        except Exception as exc:
            logger.warning(
                "Could not store the legacy image index %s (%s); this process keeps its copy",
                LEGACY_INDEX_KEY, exc,
            )
        logger.info("Indexed %d generated images for legacy lookups", len(index))
        return index

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
