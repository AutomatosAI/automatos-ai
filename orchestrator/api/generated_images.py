"""
Generated Images API
====================

Serves generated images stored in S3 or local filesystem.
GET /api/generated-images/{image_id}
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.services.image_store import get_image_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/generated-images", tags=["Generated Images"])


@router.get("/{image_id}")
async def get_generated_image(
    image_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Fetch a generated image by its UUID, scoped to caller's workspace."""
    store = get_image_store()
    result = await store.get_image(image_id, workspace_id=str(ctx.workspace_id))
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
