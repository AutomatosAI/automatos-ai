"""
Marketplace S3 Service
======================

S3 client for uploading, extracting, and fetching plugin files
from the automatos-marketplace bucket.
"""

import io
import json
import logging
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


class MarketplaceS3Service:
    """S3 client for marketplace plugin storage."""

    def __init__(self):
        if boto3 is None:
            raise ImportError("boto3 package not installed. Run: pip install boto3")

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
            # Fall back to default credential chain
            self.client = boto3.client("s3", config=boto_cfg)

    # ------------------------------------------------------------------
    # Upload helpers
    # ------------------------------------------------------------------

    async def upload_zip(self, slug: str, version: str, zip_bytes: bytes) -> str:
        """Upload a raw zip file to the pending uploads area.

        Returns the S3 key of the uploaded file.
        """
        upload_id = str(uuid4())
        key = f"_uploads/pending/{upload_id}.zip"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=zip_bytes,
                ContentType="application/zip",
            ),
        )
        logger.info("Uploaded zip for %s@%s to %s", slug, version, key)
        return key

    async def extract_plugin(
        self, slug: str, version: str, zip_bytes: bytes
    ) -> str:
        """Extract a zip archive and upload individual files to S3.

        Files are placed under ``plugins/{slug}/{version}/``.
        Returns the S3 prefix.
        """
        prefix = f"plugins/{slug}/{version}/"
        loop = asyncio.get_running_loop()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for member in zf.namelist():
                # Skip directories
                if member.endswith("/"):
                    continue
                data = zf.read(member)
                s3_key = f"{prefix}{member}"
                await loop.run_in_executor(
                    None,
                    lambda k=s3_key, d=data: self.client.put_object(
                        Bucket=self.bucket,
                        Key=k,
                        Body=d,
                    ),
                )

        logger.info("Extracted plugin %s@%s to s3://%s/%s", slug, version, self.bucket, prefix)
        return prefix

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    async def get_manifest(self, slug: str, version: str) -> dict:
        """Download and parse manifest.json for a plugin version."""
        key = f"plugins/{slug}/{version}/manifest.json"
        content = await self.get_file(key)
        return json.loads(content)

    async def get_file(self, s3_path: str) -> str:
        """Download a single file from S3 and return its content as a string."""
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.get_object(Bucket=self.bucket, Key=s3_path),
        )
        body = await loop.run_in_executor(None, lambda: response["Body"].read())
        return body.decode("utf-8")

    async def list_plugin_files(self, slug: str, version: str) -> list[str]:
        """List all file keys under ``plugins/{slug}/{version}/``."""
        prefix = f"plugins/{slug}/{version}/"
        loop = asyncio.get_running_loop()

        keys: list[str] = []
        continuation_token: Optional[str] = None

        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token

            resp = await loop.run_in_executor(
                None,
                lambda kw=kwargs: self.client.list_objects_v2(**kw),
            )

            for obj in resp.get("Contents", []):
                keys.append(obj["Key"])

            if resp.get("IsTruncated"):
                continuation_token = resp.get("NextContinuationToken")
            else:
                break

        return keys

    async def delete_plugin(self, slug: str, version: str) -> int:
        """Delete all files under ``plugins/{slug}/{version}/``.

        Returns the number of deleted objects.
        """
        keys = await self.list_plugin_files(slug, version)
        if not keys:
            return 0

        loop = asyncio.get_running_loop()

        # S3 delete_objects accepts up to 1000 keys per call
        deleted = 0
        for i in range(0, len(keys), 1000):
            batch = keys[i : i + 1000]
            await loop.run_in_executor(
                None,
                lambda b=batch: self.client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": [{"Key": k} for k in b], "Quiet": True},
                ),
            )
            deleted += len(batch)

        logger.info("Deleted %d objects for %s@%s", deleted, slug, version)
        return deleted
