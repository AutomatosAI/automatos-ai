"""
Core Storage — object storage through one S3 client factory
===========================================================

PRD-233 S4 (absorbs PRD-151). ``core.storage.s3`` is the only module in the
platform allowed to call ``boto3.client``; everything else imports from here.

Usage:
    from core.storage import get_s3_client, ensure_bucket, is_storage_configured

    ensure_bucket(config.S3_DOCUMENTS_BUCKET)          # MinIO only; no-op on AWS
    get_s3_client().put_object(Bucket=..., Key=..., Body=...)
    get_public_s3_client().generate_presigned_url(...)  # links that leave the backend
"""

from .s3 import (
    FAST_FAIL,
    STANDARD,
    STORAGE_NOT_CONFIGURED_MESSAGE,
    ClientProfile,
    StorageNotConfigured,
    ensure_bucket,
    get_public_s3_client,
    get_s3_client,
    is_storage_configured,
    reset_s3_client,
)

__all__ = [
    "FAST_FAIL",
    "STANDARD",
    "STORAGE_NOT_CONFIGURED_MESSAGE",
    "ClientProfile",
    "StorageNotConfigured",
    "ensure_bucket",
    "get_public_s3_client",
    "get_s3_client",
    "is_storage_configured",
    "reset_s3_client",
]
