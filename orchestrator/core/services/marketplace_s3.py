"""
Marketplace S3 Service
======================

S3 client for uploading, extracting, and fetching plugin files
from the automatos-marketplace bucket.

Falls back to local filesystem storage when boto3 is not installed
or AWS credentials are not configured (development mode).
"""

import io
import json
import logging
import os
import pathlib
import shutil
import zipfile
import asyncio
from uuid import uuid4
from typing import Optional

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None

from config import config

logger = logging.getLogger(__name__)


# ======================================================================
# Local filesystem fallback for development
# ======================================================================

class LocalStorageService:
    """Local filesystem storage that mirrors the S3 API for dev/testing."""

    def __init__(self):
        default_dir = os.path.join(
            os.path.expanduser("~"), ".automatos", "marketplace"
        )
        self.base_dir = pathlib.Path(
            config.MARKETPLACE_LOCAL_DIR or default_dir
        )
        self.base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        logger.info("Using local filesystem storage at %s", self.base_dir)

    async def upload_zip(self, slug: str, version: str, zip_bytes: bytes) -> str:
        upload_id = str(uuid4())
        key = f"_uploads/pending/{upload_id}.zip"
        full_path = self.base_dir / key
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(zip_bytes)
        logger.info("Uploaded zip for %s@%s to %s", slug, version, full_path)
        return key

    async def extract_plugin(self, slug: str, version: str, zip_bytes: bytes) -> str:
        prefix = f"plugins/{slug}/{version}/"
        dest = self.base_dir / prefix
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

        max_uncompressed = config.PLUGIN_MAX_UPLOAD_SIZE_MB * 1024 * 1024

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            total_uncompressed = sum(
                info.file_size for info in zf.infolist() if not info.filename.endswith("/")
            )
            if total_uncompressed > max_uncompressed:
                raise ValueError(
                    f"Total uncompressed size ({total_uncompressed} bytes) exceeds "
                    f"limit ({max_uncompressed} bytes)"
                )

            for info in zf.infolist():
                if info.filename.endswith("/"):
                    continue
                member_path = pathlib.PurePosixPath(info.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe path in zip archive: {info.filename}")
                target = dest / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(info.filename))

        logger.info("Extracted plugin %s@%s to %s", slug, version, dest)
        return prefix

    async def get_manifest(self, slug: str, version: str) -> dict:
        key = f"plugins/{slug}/{version}/manifest.json"
        content = await self.get_file(key)
        return json.loads(content)

    async def get_file(self, s3_path: str) -> str:
        full_path = self.base_dir / s3_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        return full_path.read_text("utf-8")

    async def list_plugin_files(self, slug: str, version: str) -> list:
        prefix_dir = self.base_dir / f"plugins/{slug}/{version}"
        if not prefix_dir.exists():
            return []
        return [
            str(p.relative_to(self.base_dir))
            for p in prefix_dir.rglob("*")
            if p.is_file()
        ]

    async def delete_plugin(self, slug: str, version: str) -> int:
        prefix_dir = self.base_dir / f"plugins/{slug}/{version}"
        if not prefix_dir.exists():
            return 0
        files = list(prefix_dir.rglob("*"))
        count = sum(1 for f in files if f.is_file())
        shutil.rmtree(prefix_dir)
        logger.info("Deleted %d files for %s@%s", count, slug, version)
        return count


# ======================================================================
# S3-backed storage for production
# ======================================================================

class S3StorageService:
    """S3 client for marketplace plugin storage (production)."""

    def __init__(self):
        self.bucket = config.MARKETPLACE_S3_BUCKET

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

    async def upload_zip(self, slug: str, version: str, zip_bytes: bytes) -> str:
        upload_id = str(uuid4())
        key = f"_uploads/pending/{upload_id}.zip"
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


# ======================================================================
# Factory: auto-select backend
# ======================================================================

def MarketplaceS3Service():
    """Factory that returns S3StorageService or LocalStorageService.

    Uses local filesystem when:
      - boto3 is not installed, OR
      - AWS_ACCESS_KEY_ID is not set
    """
    use_s3 = (
        boto3 is not None
        and getattr(config, "AWS_ACCESS_KEY_ID", None)
        and getattr(config, "AWS_SECRET_ACCESS_KEY", None)
    )
    if use_s3:
        logger.info("Marketplace storage: S3 (bucket=%s)", config.MARKETPLACE_S3_BUCKET)
        return S3StorageService()
    else:
        logger.info("Marketplace storage: local filesystem (no AWS credentials)")
        return LocalStorageService()
