"""
Image Store Service
====================

Uploads generated images (base64) to the platform object store (S3 / MinIO
via ``core.storage``, PRD-233 S4) and serves them back by id.

S3 key pattern: generated-images/{workspace_id}/{uuid}.{ext}
"""

import asyncio
import base64
import logging
from typing import Optional, Tuple
from uuid import uuid4

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
        ws = workspace_id or "default"
        key = f"generated-images/{ws}/{image_id}.{ext}"
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
        logger.info("Saved image to S3: %s (%d bytes)", key, len(image_bytes))
        return image_id

    async def get_image(self, image_id: str, workspace_id: Optional[str] = None) -> Optional[Tuple[bytes, str]]:
        ws = workspace_id or "default"
        prefix = f"generated-images/{ws}/{image_id}"
        loop = asyncio.get_running_loop()
        try:
            # List objects matching the prefix to find the exact extension
            resp = await loop.run_in_executor(
                None,
                lambda: self.client.list_objects_v2(
                    Bucket=self.bucket, Prefix=prefix, MaxKeys=5
                ),
            )
            contents = resp.get("Contents", [])
            if not contents:
                # Try without workspace scope (search all)
                prefix_any = f"generated-images/"
                resp = await loop.run_in_executor(
                    None,
                    lambda: self.client.list_objects_v2(
                        Bucket=self.bucket, Prefix=prefix_any, MaxKeys=1000
                    ),
                )
                contents = [
                    c for c in resp.get("Contents", [])
                    if image_id in c["Key"]
                ]
            if not contents:
                return None
            key = contents[0]["Key"]
            obj = await loop.run_in_executor(
                None,
                lambda: self.client.get_object(Bucket=self.bucket, Key=key),
            )
            body = await loop.run_in_executor(None, lambda: obj["Body"].read())
            content_type = obj.get("ContentType", "image/png")
            return body, content_type
        except Exception as e:
            logger.error("Failed to retrieve image %s: %s", image_id, e)
            return None


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
