"""
Audio Validation & Storage (PRD-74)
Validates audio uploads and manages S3 storage for voice messages.
"""

import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile, HTTPException

from config import config

logger = logging.getLogger(__name__)

ALLOWED_AUDIO_TYPES = frozenset({
    "audio/webm", "audio/ogg", "audio/wav", "audio/wave",
    "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/mp4",
    "audio/flac", "audio/x-flac",
})

ALLOWED_EXTENSIONS = frozenset({
    ".webm", ".ogg", ".wav", ".mp3", ".m4a", ".flac", ".opus",
})


def _get_s3_client():
    return boto3.client(
        "s3",
        region_name=config.AWS_REGION,
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    )


async def validate_audio(audio: UploadFile) -> bytes:
    """Validate audio file format, size, and return bytes."""
    # Check content type
    content_type = audio.content_type or ""
    filename = audio.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if content_type not in ALLOWED_AUDIO_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {content_type}. Allowed: webm, ogg, wav, mp3, m4a, flac, opus",
        )

    # Read and check size
    audio_bytes = await audio.read()
    max_bytes = config.VOICE_MAX_AUDIO_SIZE_MB * 1024 * 1024

    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({len(audio_bytes) / 1024 / 1024:.1f}MB). Max: {config.VOICE_MAX_AUDIO_SIZE_MB}MB",
        )

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    return audio_bytes


def upload_voice_audio(
    workspace_id: str,
    message_id: str,
    audio_bytes: bytes,
    audio_format: str = "mp3",
) -> str:
    """Upload voice audio to S3. Returns the S3 key."""
    s3_key = f"workspaces/{workspace_id}/voice/{message_id}.{audio_format}"

    content_type_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "opus": "audio/ogg",
        "webm": "audio/webm",
    }

    try:
        client = _get_s3_client()
        client.put_object(
            Bucket=config.S3_DOCUMENTS_BUCKET,
            Key=s3_key,
            Body=audio_bytes,
            ContentType=content_type_map.get(audio_format, "audio/mpeg"),
        )
        logger.info("voice_audio_uploaded", extra={"s3_key": s3_key, "size_bytes": len(audio_bytes)})
        return s3_key
    except ClientError as e:
        logger.error("voice_audio_upload_failed", extra={"s3_key": s3_key, "error": str(e)})
        raise HTTPException(status_code=500, detail="Failed to store audio")


def get_voice_audio_url(s3_key: str, expires_in: int = 3600) -> str:
    """Generate a presigned URL for audio playback."""
    try:
        client = _get_s3_client()
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": config.S3_DOCUMENTS_BUCKET, "Key": s3_key},
            ExpiresIn=expires_in,
        )
    except ClientError as e:
        logger.error("voice_audio_url_failed", extra={"s3_key": s3_key, "error": str(e)})
        raise HTTPException(status_code=404, detail="Audio not found or expired")


def delete_voice_audio(s3_key: str) -> None:
    """Delete voice audio from S3."""
    try:
        client = _get_s3_client()
        client.delete_object(Bucket=config.S3_DOCUMENTS_BUCKET, Key=s3_key)
    except ClientError:
        pass  # Best-effort deletion
