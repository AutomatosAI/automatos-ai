"""An image a chat answer carries is saved with the AI mark (PRD-264 EUAI-09, Art. 50(2)).

A model can answer in chat with an image, as a base64 ``data:`` URL in its markdown.
``service._upload_inline_images`` is the last pass over each text the turn saves: it
puts each such image in the image store and leaves its link in the saved message.
:func:`reply_images_are_marked` wraps it. Before the upload, every PNG or JPEG data
URL is swapped for the same image with the IPTC "trained algorithmic media" XMP mark
and a random instance id (``modules.documents.marked_images``; an image with a C2PA
manifest is kept as it came). After it, each image the store kept has its SHA-256
recorded as an ``image`` provenance row, when the turn has a workspace. ``modules`` is
imported only when a reply carries an image, never when this module (and so the chat
service) loads.

An image that cannot be read as its type is left as it came and logged: it is not a
picture any viewer can show. WebP and GIF are left unmarked (no pinned library writes
their XMP without re-encoding the picture).
"""
from __future__ import annotations

import functools
import logging
import re
from typing import Any, Awaitable, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

Upload = Callable[..., Awaitable[str]]
Image = Tuple[str, str]  # (mime, base64)

# A PNG or JPEG data URL, as the upload's own pattern finds them in a reply's markdown.
_MARKABLE_DATA_URL = re.compile(r"data:(image/(?:png|jpeg|jpg));base64,([A-Za-z0-9+/=\s]+)")
_WHITESPACE = re.compile(r"\s+")


def _marked_text(text: str) -> Tuple[str, List[Image]]:
    """``text`` with each PNG/JPEG data URL marked, and the (mime, base64) of each one marked."""
    if not _MARKABLE_DATA_URL.search(text or ""):
        return text, []
    from modules.documents.marked_images import marked_model_image

    marked: List[Image] = []

    def mark(match: "re.Match[str]") -> str:
        mime, b64 = match.group(1), _WHITESPACE.sub("", match.group(2))
        try:
            mime, b64 = marked_model_image(mime, b64)
        except ValueError as e:
            logger.warning("[EUAI-09] an image in a chat answer could not be read to mark it: %s", e)
            return match.group(0)
        marked.append((mime, b64))
        return f"data:{mime};base64,{b64}"

    return _MARKABLE_DATA_URL.sub(mark, text), marked


def _kept(images: List[Image], saved: str) -> List[Image]:
    """The marked images the upload took out of the text (the ones still inline were not kept)."""
    return [image for image in images if f"base64,{image[1]}" not in saved]


def reply_images_are_marked(upload: Upload) -> Upload:
    """Wrap ``service._upload_inline_images(text, workspace_id=None)``: the images it saves
    carry the AI mark, and each one kept gets a provenance row."""
    @functools.wraps(upload)
    async def wrapped(text: str, workspace_id: Optional[Any] = None) -> str:
        marked_text, images = _marked_text(text)
        saved = await upload(marked_text, workspace_id=workspace_id)
        if workspace_id is not None and images:
            from modules.documents.marked_images import record_model_image

            for mime, b64 in _kept(images, saved):
                record_model_image(workspace_id, mime, b64)
        return saved
    return wrapped


__all__ = ["reply_images_are_marked"]
