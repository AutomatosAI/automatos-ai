"""
Generated Images API
====================

Serves generated images stored in S3 or local filesystem.
GET /api/generated-images/{image_id}
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from core.services.image_store import get_image_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/generated-images", tags=["Generated Images"])


@router.get("/{image_id}")
async def get_generated_image(image_id: str):
    """Fetch a generated image by its UUID."""
    store = get_image_store()
    result = await store.get_image(image_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Image not found")
    image_bytes, content_type = result
    return Response(
        content=image_bytes,
        media_type=content_type,
        headers={
            "Cache-Control": "public, max-age=86400, immutable",
        },
    )
