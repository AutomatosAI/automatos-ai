"""An image an image model made carries the AI mark and a provenance row (PRD-264 EUAI-09).

Two paths keep an image a model made: ``platform_generate_cover_image``
(``modules/tools/discovery/handlers_blog.py``, the one image tool: a blog cover or an
image Deliverable) and a chat answer that carries an image as a data URL
(``consumers/chatbot/marked_reply_images.py``). The image arrives as base64, and passes
here twice:

* :func:`marked_model_image` as it arrives, before anything keeps it: a PNG or JPEG
  gets the IPTC "trained algorithmic media" XMP mark with a random instance id
  (``provenance_marks.marked_image_bytes``). An image that already carries a C2PA
  manifest is kept byte for byte (the manifest states its source and signs these
  bytes). Other types (WebP, GIF) pass unmarked: no pinned library writes their XMP.
  An image that cannot be read as its type raises ValueError, so it is never kept
  unmarked.
* :func:`record_model_image` once the image store has kept it: its SHA-256 becomes an
  ``image`` row (``services.content_provenance.record_produced_file``), so the public
  check can say Automatos AI made it. Only a type the mark is written for gets a row
  (FR-16: a file with a row carries a per-file instance id, or its own C2PA manifest).

``services`` is imported when an image is recorded, never when this module loads.
"""
from __future__ import annotations

import base64
from typing import Any, Tuple

from .provenance_marks import MARKED_IMAGE_TYPES, TRAINED_ALGORITHMIC_MEDIA, marked_image_bytes, sha256_hex

# What an image's provenance row says made it: the toolkit, never a person or a prompt.
IMAGE_GENERATOR = "Automatos AI image generation"
IMAGE_KIND = "image"


def _image_type(mime: str) -> str:
    return (mime or "").lower()


def marked_model_image(mime: str, b64: str) -> Tuple[str, str]:
    """(mime, base64) of the model's image with the AI mark; the base64 as it came when the
    image is kept unchanged. ValueError when the base64 or the image cannot be read."""
    data = base64.b64decode(b64, validate=True)
    marked = marked_image_bytes(data, mime, TRAINED_ALGORITHMIC_MEDIA)
    return mime, (b64 if marked is data else base64.b64encode(marked).decode("ascii"))


def record_model_image(workspace_id: Any, mime: str, b64: str) -> bool:
    """Record the kept image's SHA-256 as an ``image`` provenance row (best effort, off the
    event loop). False when its type carries no mark, so it gets no row."""
    if _image_type(mime) not in MARKED_IMAGE_TYPES:
        return False
    from services.content_provenance import record_produced_file

    record_produced_file(workspace_id, sha256_hex(base64.b64decode(b64)), IMAGE_KIND, IMAGE_GENERATOR)
    return True


__all__ = ["IMAGE_GENERATOR", "IMAGE_KIND", "marked_model_image", "record_model_image"]
