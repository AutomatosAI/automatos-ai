"""
Marketplace S3 Service
======================

S3 client for uploading, extracting, and fetching plugin files
from the automatos-marketplace bucket.

Every client comes from ``core.storage`` (PRD-233 S4): AWS S3 in SaaS, MinIO
locally through the same code path — no filesystem fallback.
"""

import io
import json
import logging
import pathlib
import zipfile
import asyncio
from uuid import uuid4

from config import config
from core.storage import ensure_bucket, get_s3_client

logger = logging.getLogger(__name__)


class MarketplaceS3Service:
    """Marketplace plugin storage on the platform object store (S3 / MinIO)."""

    def __init__(self):
        self.bucket = config.MARKETPLACE_S3_BUCKET

    @property
    def client(self):
        """The process-wide S3 client — lazy, no network at construction."""
        return get_s3_client()

    async def _ensure_bucket(self) -> None:
        """Self-create the marketplace bucket on MinIO before the first write."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: ensure_bucket(self.bucket))

    async def upload_zip(self, slug: str, version: str, zip_bytes: bytes) -> str:
        upload_id = str(uuid4())
        key = f"_uploads/pending/{upload_id}.zip"
        await self._ensure_bucket()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.put_object(
                Bucket=self.bucket, Key=key, Body=zip_bytes, ContentType="application/zip",
            ),
        )
        logger.info("Uploaded zip for %s@%s to %s", slug, version, key)
        return key

    async def extract_plugin(self, slug: str, version: str, zip_bytes: bytes) -> str:
        prefix = f"plugins/{slug}/{version}/"
        await self._ensure_bucket()
        loop = asyncio.get_running_loop()
        max_uncompressed = config.PLUGIN_MAX_UPLOAD_SIZE_MB * 1024 * 1024

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            total_uncompressed = 0
            for info in zf.infolist():
                if info.filename.endswith("/"):
                    continue
                member_path = pathlib.PurePosixPath(info.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe path in zip archive: {info.filename}")
                total_uncompressed += info.file_size

            if total_uncompressed > max_uncompressed:
                raise ValueError(
                    f"Total uncompressed size ({total_uncompressed} bytes) exceeds "
                    f"limit ({max_uncompressed} bytes)"
                )

            for info in zf.infolist():
                if info.filename.endswith("/"):
                    continue
                data = zf.read(info.filename)
                s3_key = f"{prefix}{info.filename}"
                await loop.run_in_executor(
                    None,
                    lambda k=s3_key, d=data: self.client.put_object(
                        Bucket=self.bucket, Key=k, Body=d,
                    ),
                )

        logger.info("Extracted plugin %s@%s to s3://%s/%s", slug, version, self.bucket, prefix)
        return prefix

    async def get_manifest(self, slug: str, version: str) -> dict:
        key = f"plugins/{slug}/{version}/manifest.json"
        content = await self.get_file(key)
        return json.loads(content)

    async def get_file(self, s3_path: str) -> str:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.get_object(Bucket=self.bucket, Key=s3_path),
        )
        body = await loop.run_in_executor(None, lambda: response["Body"].read())
        return body.decode("utf-8")

    async def list_plugin_files(self, slug: str, version: str) -> list:
        prefix = f"plugins/{slug}/{version}/"
        loop = asyncio.get_running_loop()
        keys = []
        continuation_token = None

        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            resp = await loop.run_in_executor(
                None, lambda kw=kwargs: self.client.list_objects_v2(**kw),
            )
            for obj in resp.get("Contents", []):
                keys.append(obj["Key"])
            if resp.get("IsTruncated"):
                continuation_token = resp.get("NextContinuationToken")
            else:
                break
        return keys

    async def delete_plugin(self, slug: str, version: str) -> int:
        keys = await self.list_plugin_files(slug, version)
        if not keys:
            return 0
        loop = asyncio.get_running_loop()
        deleted = 0
        for i in range(0, len(keys), 1000):
            batch = keys[i : i + 1000]
            response = await loop.run_in_executor(
                None,
                lambda b=batch: self.client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": [{"Key": k} for k in b], "Quiet": True},
                ),
            )
            errors = response.get("Errors", [])
            if errors:
                raise RuntimeError(f"S3 partial delete failure for {slug}@{version}")
            deleted += len(response.get("Deleted", batch))
        logger.info("Deleted %d objects for %s@%s", deleted, slug, version)
        return deleted

