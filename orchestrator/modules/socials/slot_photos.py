"""PRD-251B (3 Oct 2026 pass): the person's own picture in a template's photo slot.

Upload and Library used to replace the template: the file became the whole post. With a
template that shows a photo (an image slot, such as the photo cards), the file now fills
that slot instead, so the template's words sit over it. The slot's record is marked done
with the file, as a picked AI option's is (``ai_options.pick``), and the next render shows
it: ``render.footage_plan_for`` keeps a done slot's file and makes nothing for it.

The file lives under the post's own prefix (``media_store.media_key``): an upload is stored
there already, and a Library file is copied there. The record names where the file came
from (``service.OWN_FILE_TOOLKITS``: no AI made it, so a channel's AI label stays off), and
its prompt names the file, so the editor's saves keep the record: they send each image
slot's prompt back, and a prompt that did not change keeps what the slot holds.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Dict, Mapping, Optional
from uuid import UUID

from core.social_templates import IMAGE_SLOT
from modules.socials import service

NOT_A_PHOTO_SPOT = "{slot} is not a photo spot of this post's template. Pick a template marked Photo first."
NOT_A_PICTURE = "A photo spot takes a picture: a PNG, JPEG or WebP image."
LIBRARY_PREFIX = "library"
LIBRARY_ID_CHARS = 16


def photo_spot(blocks: Any, slot: str) -> Mapping[str, Any]:
    """The template's image slot ``slot``; :class:`service.InvalidPost` (422) when it has none."""
    slots = blocks.get("slots") if isinstance(blocks, Mapping) else None
    spec = slots.get(slot) if isinstance(slots, Mapping) else None
    if not isinstance(spec, Mapping) or spec.get("kind") != IMAGE_SLOT:
        raise service.InvalidPost(NOT_A_PHOTO_SPOT.format(slot=slot))
    return spec


def library_name(deliverable_id: Any, source_name: Optional[str]) -> str:
    """The copy's file name under the post's prefix: ``library-<id>.<the source's extension>``."""
    extension = PurePosixPath(source_name or "").suffix.lower().lstrip(".") or "png"
    return f"{LIBRARY_PREFIX}-{UUID(str(deliverable_id)).hex[:LIBRARY_ID_CHARS]}.{extension}"


def own_file_record(
    spec: Mapping[str, Any], slot: str, *, source: str, deliverable_id: Any, name: str,
    content_type: str, size: Optional[int] = None, sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """The slot's record for the person's own picture: done, its file, and where it came from."""
    label = str(spec.get("label") or slot)
    record = {
        "prompt": f"{label}: your own picture ({name})",
        "status": service.FOOTAGE_DONE,
        "toolkit": source,
        "deliverable_id": str(deliverable_id),
        "name": name,
        "content_type": content_type,
        "bytes": size,
        "sha256": sha256,
    }
    return {key: value for key, value in record.items() if value is not None}


def with_own_file(post: Any, slot: str, record: Mapping[str, Any]) -> None:
    """``post``'s footage with ``slot`` holding ``record``; every other slot as it was."""
    footage = dict(post.footage) if isinstance(post.footage, Mapping) else {}
    post.footage = {**footage, slot: dict(record)}
