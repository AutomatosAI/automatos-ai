"""
Socials media upload (PRD-251B US-B109)
=======================================

``POST /api/socials/posts/{post_id}/media`` (multipart, one ``file``) makes the post's
visual a file the person brings: a PNG, JPEG or WebP image of at most
``SOCIALS_UPLOAD_IMAGE_MAX_BYTES``, or an MP4 video of at most
``SOCIALS_UPLOAD_VIDEO_MAX_BYTES``. The type is sniffed from the bytes, never taken from
the file's name or its declared type: anything else is 415, a file over its limit 413.

The file is stored through the one storage factory (``MediaStore``, the documents
bucket) under ``social-media/{workspace}/{post}/upload-<sha>.<ext>`` like rendered media,
registered as a Deliverable (``source_type`` upload), and becomes the post's media
(``{"original": [deliverable id]}``; a still is cropped for each channel when the post
renders, and the preview shows each crop: PRD-251C US-C303, ``modules/socials/upload_crops.py``)
with ``template_id`` null, the format the file is (image or video) and no chosen length. Media is content, so the edit goes through ``service.update_post`` and voids an
approval like any edit, committed by the same compare-and-set (409 when another writer
committed first). A post that is rendering, publishing, published or archived cannot
change (409), and nothing is stored then.

PRD-251B (3 Oct 2026 pass): with ``slot`` (a form field), the picture fills that photo spot
of the post's template instead (``modules/socials/slot_photos.py``): the template stays, and
its words sit over the picture. ``PUT /api/socials/posts/{post_id}/photos/{slot}`` does the
same with a Library picture, one of the workspace's image Deliverables, copied under the
post's prefix. A photo spot takes a picture only (422 otherwise, or when the template has no
such spot). The slot's record is a render setting: an approval stands, as a picked AI
option's does.

This router has no prefix and no gate of its own: ``api/socials.py`` includes it in the
Socials router, whose ``require_socials_enabled`` answers 404 unless both switches are
on (D1). Never mount it in the app directly. It reuses that module's post helpers,
imported when a request runs, because that module includes this one.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Tuple
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.core import DocumentTemplate
from core.models.socials import SocialPost
from modules.socials import media_store, media_urls, service, slot_photos
from modules.socials.recipes.footage import locked_post

logger = logging.getLogger(__name__)
router = APIRouter()

CAN_UPDATE = Depends(require_workspace_permission("documents:update"))
UPLOAD_SOURCE_TYPE = "upload"
UPLOAD_ASPECT = "original"
UPLOAD_FILE_PREFIX = "upload"
DIGEST_CHARS = 16
SNIFF_BYTES = 16
COPY_CHUNK_BYTES = 1 << 20
HTTP_TOO_LARGE = 413
HTTP_UNSUPPORTED = 415
NOT_SUPPORTED = "Upload a PNG, JPEG or WebP image, or an MP4 video."
NO_FILE = "Attach the file to upload as `file`."
STORAGE_UNAVAILABLE = "Object storage is not configured, so a file cannot be uploaded."
NOT_IN_LIBRARY = "That file is not in this workspace's Deliverables."
# ftyp brands of an MP4. A QuickTime movie (qt), an HEIC or AVIF still and the rest are not MP4.
MP4_BRANDS = frozenset({b"isom", b"iso2", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"avc1", b"M4V ", b"dash"})


@dataclass(frozen=True)
class UploadKind:
    extension: str
    content_type: str
    post_format: str

    @property
    def max_bytes(self) -> int:
        if self.post_format == "video":
            return config.SOCIALS_UPLOAD_VIDEO_MAX_BYTES
        return config.SOCIALS_UPLOAD_IMAGE_MAX_BYTES


PNG = UploadKind("png", "image/png", "image")
JPEG = UploadKind("jpg", "image/jpeg", "image")
WEBP = UploadKind("webp", "image/webp", "image")
MP4 = UploadKind("mp4", "video/mp4", "video")


def sniff(head: bytes) -> Optional[UploadKind]:
    """What the file's first bytes say it is, or ``None``: never its name."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return PNG
    if head.startswith(b"\xff\xd8\xff"):
        return JPEG
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return WEBP
    if head[4:8] == b"ftyp" and head[8:12] in MP4_BRANDS:
        return MP4
    return None


def _posts_api() -> Any:
    import api.socials as posts_api  # that module includes this router

    return posts_api


def _too_large(kind: UploadKind) -> HTTPException:
    megabytes = kind.max_bytes / (1 << 20)
    return HTTPException(status_code=HTTP_TOO_LARGE, detail=f"The file is over the {megabytes:g} MB limit for this kind of file.")


def _spool(source: BinaryIO, kind: UploadKind, scratch: Path, head: bytes) -> Tuple[Path, int, str]:
    """Copy the upload into ``scratch`` with its size and sha256, refusing it the moment it passes its limit."""
    digest, size = hashlib.sha256(), 0
    path = scratch / f"{UPLOAD_FILE_PREFIX}.{kind.extension}"
    with path.open("wb") as out:
        chunk = head
        while chunk:
            size += len(chunk)
            if size > kind.max_bytes:
                raise _too_large(kind)
            digest.update(chunk)
            out.write(chunk)
            chunk = source.read(COPY_CHUNK_BYTES)
    return path, size, digest.hexdigest()


def _register(db: Session, post: SocialPost, key: str, file_name: str, size: int, digest: str) -> str:
    from services.deliverable_service import DeliverableService, _infer_artifact_type

    result = DeliverableService(db, post.workspace_id).register(
        file_path=key,
        title=f"{post.title} (upload)",
        source_type=UPLOAD_SOURCE_TYPE,
        source_id=str(post.id),
        artifact_type=_infer_artifact_type(file_name),
        storage_type="s3",
        file_type=Path(file_name).suffix.lstrip("."),
        file_size_bytes=size,
        preview_url=media_store.media_route(post.id, file_name),
        preview_type="file",
        extra={"social_post_id": str(post.id), "sha256": digest, "aspect": UPLOAD_ASPECT},
    )
    if not result.get("success") or not result.get("deliverable_id"):
        logger.error("[Socials] the upload for post %s was stored but not registered: %s", post.id, result.get("error"))
        raise HTTPException(status_code=500, detail="The file could not be saved as a Deliverable.")
    return str(result["deliverable_id"])


def _store(upload: UploadFile, post: SocialPost, *, picture_only: bool = False) -> Tuple[UploadKind, str, str, int, str]:
    """Sniff, bound and store the upload: (kind, key, file name, bytes, sha256). A photo spot's
    upload (``picture_only``) is refused before anything is stored unless it is a picture."""
    head = upload.file.read(SNIFF_BYTES)
    kind = sniff(head)
    if kind is None:
        raise HTTPException(status_code=HTTP_UNSUPPORTED, detail=NOT_SUPPORTED)
    if picture_only and kind.post_format != "image":
        raise HTTPException(status_code=HTTP_UNSUPPORTED, detail=slot_photos.NOT_A_PICTURE)
    if upload.size is not None and upload.size > kind.max_bytes:
        raise _too_large(kind)
    store = media_store.MediaStore()
    if not store.configured():
        raise HTTPException(status_code=503, detail=STORAGE_UNAVAILABLE)
    with tempfile.TemporaryDirectory(prefix="socials-upload-") as scratch:
        path, size, digest = _spool(upload.file, kind, Path(scratch), head)
        file_name = f"{UPLOAD_FILE_PREFIX}-{digest[:DIGEST_CHARS]}.{kind.extension}"
        key = media_store.media_key(post.workspace_id, post.id, file_name)
        store.put_file(key, path, kind.content_type)
    return kind, key, file_name, size, digest


def _photo_spot(db: Session, post: SocialPost, slot: str) -> Dict[str, Any]:
    """The post's template's photo spot ``slot`` (the workspace's own template); 422 otherwise."""
    row = None
    if post.template_id is not None:
        row = (
            db.query(DocumentTemplate.blocks)
            .filter(DocumentTemplate.id == post.template_id, DocumentTemplate.workspace_id == post.workspace_id)
            .first()
        )
    try:
        return dict(slot_photos.photo_spot(row.blocks if row is not None else None, slot))
    except service.InvalidPost as exc:
        _posts_api()._raise_for(exc)


def _into_slot(db: Session, post: SocialPost, slot: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """The photo spot holds ``record`` now: the post read locked (footage is written whole, and
    a slot's AI options may be settling in the background), then the compare-and-set commit."""
    posts_api = _posts_api()
    locked = locked_post(db, post.workspace_id, post.id)
    if locked is None:
        raise HTTPException(status_code=404, detail=posts_api.POST_NOT_FOUND)
    if locked.status not in service.EDITABLE_STATUSES:
        posts_api._raise_for(service.IllegalTransition(locked.status, service.ACTION_EDIT))
    status, content_hash = locked.status, locked.content_hash
    slot_photos.with_own_file(locked, slot, record)
    return posts_api._commit_unchanged(db, locked, status=status, content_hash=content_hash)


@router.post("/posts/{post_id}/media", dependencies=[CAN_UPDATE])
def upload_social_post_media(
    post_id: UUID,
    file: Optional[UploadFile] = File(None),
    slot: Optional[str] = Form(None, max_length=64),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """The post's visual becomes the uploaded file, or with ``slot`` the template's photo spot
    holds it (the module docstring). A plain ``def`` (F105).

    The file is optional to FastAPI so the post is looked up first: another workspace's
    post is 404 whatever the request carries.
    """
    posts_api = _posts_api()
    post = posts_api._load(db, ctx, post_id)
    actor = posts_api._actor(ctx)
    if file is None:
        raise HTTPException(status_code=422, detail=NO_FILE)
    if post.status not in service.EDITABLE_STATUSES:
        posts_api._raise_for(service.IllegalTransition(post.status, service.ACTION_EDIT))
    spec = _photo_spot(db, post, slot) if slot is not None else None
    kind, key, file_name, size, digest = _store(file, post, picture_only=spec is not None)
    deliverable_id = _register(db, post, key, file_name, size, digest)
    logger.info("[Socials] post %s: uploaded %s (%d bytes)", post.id, key, size)
    if spec is not None:
        record = slot_photos.own_file_record(
            spec, slot, source=service.OWN_FILE_UPLOAD, deliverable_id=deliverable_id, name=file_name,
            content_type=kind.content_type, size=size, sha256=digest,
        )
        return _into_slot(db, post, slot, record)
    status, content_hash = post.status, post.content_hash
    changes = {"media": {UPLOAD_ASPECT: [deliverable_id]}, "template_id": None, "length_seconds": None, "format": kind.post_format}
    try:
        service.update_post(post, actor, changes)
        return posts_api._commit_unchanged(db, post, status=status, content_hash=content_hash)
    except service.SocialsError as exc:
        posts_api._raise_for(exc)


class PhotoRequest(BaseModel):
    """The Library picture a photo spot takes: one of the workspace's image Deliverables."""

    model_config = ConfigDict(extra="forbid")

    deliverable_id: str = Field(..., min_length=1, max_length=64)


def _library_picture(db: Session, post: SocialPost, deliverable_id: str) -> media_urls.MediaFile:
    """The workspace's Deliverable as a stored picture; 404 when it is not the workspace's, 422 when
    it is not a picture in object storage."""
    found = media_urls.deliverable_file(db, post.workspace_id, deliverable_id)
    if found is None:
        raise HTTPException(status_code=404, detail=NOT_IN_LIBRARY)
    if found.key is None:
        raise HTTPException(status_code=422, detail=found.error or media_urls.NOT_IN_STORAGE)
    if not (found.content_type or "").startswith("image/"):
        raise HTTPException(status_code=422, detail=slot_photos.NOT_A_PICTURE)
    return found


@router.put("/posts/{post_id}/photos/{slot}", dependencies=[CAN_UPDATE])
def put_social_post_photo(
    post_id: UUID,
    slot: str,
    body: PhotoRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """A Library picture fills the post's template's photo spot ``slot`` (the module docstring):
    copied under the post's prefix and recorded as the slot's file. A plain ``def`` (F105)."""
    posts_api = _posts_api()
    post = posts_api._load(db, ctx, post_id)
    if post.status not in service.EDITABLE_STATUSES:
        posts_api._raise_for(service.IllegalTransition(post.status, service.ACTION_EDIT))
    spec = _photo_spot(db, post, slot)
    picture = _library_picture(db, post, body.deliverable_id)
    store = media_store.MediaStore()
    if not store.configured():
        raise HTTPException(status_code=503, detail=STORAGE_UNAVAILABLE)
    name = slot_photos.library_name(picture.deliverable_id, picture.name)
    store.copy(picture.key, media_store.media_key(post.workspace_id, post.id, name))
    record = slot_photos.own_file_record(
        spec, slot, source=service.OWN_FILE_LIBRARY, deliverable_id=picture.deliverable_id, name=name,
        content_type=picture.content_type or "image/png", size=picture.bytes,
    )
    return _into_slot(db, post, slot, record)
