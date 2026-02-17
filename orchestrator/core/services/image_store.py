"""
Image Store Service
====================

Uploads generated images (base64) to S3 and returns servable URLs.
Falls back to local filesystem when AWS credentials are not configured.

S3 key pattern: generated-images/{workspace_id}/{uuid}.{ext}
"""

import asyncio
import base64
import io
import logging
import os
import pathlib
from typing import Optional, Tuple
from uuid import uuid4

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None

from config import config

logger = logging.getLogger(__name__)

MIME_TO_EXT = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/svg+xml": "svg",
}


class LocalImageStore:
    """Local filesystem fallback for dev environments."""

    def __init__(self):
        self.base_dir = pathlib.Path(
            os.getenv("IMAGE_STORE_LOCAL_DIR",
                       os.path.join(os.path.expanduser("~"), ".automatos", "generated-images"))
        )
        self.base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        logger.info("Image store: local filesystem at %s", self.base_dir)

    async def save_image(
        self,
        base64_data: str,
        mime_type: str = "image/png",
        workspace_id: Optional[str] = None,
    ) -> str:
        ext = MIME_TO_EXT.get(mime_type, "png")
        image_id = str(uuid4())
        ws = workspace_id or "default"
        rel_path = f"{ws}/{image_id}.{ext}"
        full_path = self.base_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        image_bytes = base64.b64decode(base64_data)
        full_path.write_bytes(image_bytes)
        logger.info("Saved image locally: %s (%d bytes)", rel_path, len(image_bytes))
        return image_id

    async def get_image(self, image_id: str, workspace_id: Optional[str] = None) -> Optional[Tuple[bytes, str]]:
        ws = workspace_id or "default"
        parent = self.base_dir / ws
        if not parent.exists():
            # Search all workspace dirs
            for d in self.base_dir.iterdir():
                if d.is_dir():
                    for f in d.iterdir():
                        if f.stem == image_id:
                            mime = _ext_to_mime(f.suffix.lstrip("."))
                            return f.read_bytes(), mime
            return None
        for f in parent.iterdir():
            if f.stem == image_id:
                mime = _ext_to_mime(f.suffix.lstrip("."))
                return f.read_bytes(), mime
        # Fallback: search all workspaces
        for d in self.base_dir.iterdir():
            if d.is_dir():
                for f in d.iterdir():
                    if f.stem == image_id:
                        mime = _ext_to_mime(f.suffix.lstrip("."))
                        return f.read_bytes(), mime
        return None


class S3ImageStore:
    """S3-backed image store for production."""

    def __init__(self):
        self.bucket = getattr(config, "RECIPE_LOG_S3_BUCKET", "automatos-ai")
        boto_cfg = BotoConfig(
            region_name=config.AWS_REGION or "us-east-1",
            signature_version="v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        if config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY:
            self.client = boto3.client(
                "s3",
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                config=boto_cfg,
            )
        else:
            self.client = boto3.client("s3", config=boto_cfg)
        logger.info("Image store: S3 (bucket=%s)", self.bucket)

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


def _ext_to_mime(ext: str) -> str:
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "svg": "image/svg+xml",
    }.get(ext, "image/png")


# ======================================================================
# Factory
# ======================================================================

_image_store = None


def get_image_store():
    """Get or create the image store singleton."""
    global _image_store
    if _image_store is not None:
        return _image_store

    use_s3 = (
        boto3 is not None
        and getattr(config, "AWS_ACCESS_KEY_ID", None)
        and getattr(config, "AWS_SECRET_ACCESS_KEY", None)
    )
    if use_s3:
        _image_store = S3ImageStore()
    else:
        _image_store = LocalImageStore()
    return _image_store
