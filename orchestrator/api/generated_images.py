"""
Generated Images API
====================

Serves generated images stored in the platform object store.
GET /api/generated-images/{image_id}

Public endpoint — image IDs are unguessable UUIDs, and <img> tags in
rendered markdown cannot send auth headers.

The body streams from S3 (never read whole into memory) and one ``Range:
bytes=a-b`` is honoured with a 206 (PRD-251 S0.4c).
"""

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response, StreamingResponse

from core.services.image_store import IMAGE_ROUTE, ImageRangeNotSatisfiable, get_image_store, parse_byte_range

logger = logging.getLogger(__name__)

router = APIRouter(prefix=IMAGE_ROUTE, tags=["Generated Images"])

CACHE_CONTROL = "public, max-age=86400, immutable"


@router.get("/{image_id}")
async def get_generated_image(image_id: str, range_header: Optional[str] = Header(None, alias="Range")):
    """Fetch a generated image by its UUID (public, unguessable ID)."""
    store = get_image_store()
    byte_range = parse_byte_range(range_header)
    try:
        # No workspace: the id resolves through its pointer (one GET).
        stream = await store.open_image(image_id, byte_range=byte_range)
    except ImageRangeNotSatisfiable as exc:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{exc.size}", "Accept-Ranges": "bytes"},
        )
    except Exception as exc:
        logger.error("Failed to open generated image %s: %s", image_id, exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Image storage unavailable") from exc
    if stream is None:
        raise HTTPException(status_code=404, detail="Image not found")

    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": CACHE_CONTROL,
        "Content-Length": str(stream.content_length),
    }
    status_code = 200
    if stream.content_range:
        status_code = 206
        headers["Content-Range"] = stream.content_range
    return StreamingResponse(
        stream.body,
        status_code=status_code,
        media_type=stream.content_type,
        headers=headers,
    )
