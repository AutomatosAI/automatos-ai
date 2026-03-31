"""
Session Checkpoint Service — PRD-123 Pattern #8
================================================

Writes and reads mission session checkpoints to/from S3 for crash recovery.
Checkpoints are created after each verified task completion.
"""

import json
import logging
from typing import Optional
from uuid import UUID

import boto3

from config import config
from core.models.orchestration import SessionCheckpoint

logger = logging.getLogger(__name__)

_CHECKPOINT_PREFIX = "checkpoints"
_BUCKET = config.RECIPE_LOG_S3_BUCKET  # Reuse existing S3 bucket


def _s3_client():
    """Lazy S3 client initialization."""
    return boto3.client("s3")


def _checkpoint_key(run_id: UUID, checkpoint_number: int) -> str:
    """S3 key for a specific checkpoint."""
    return f"{_CHECKPOINT_PREFIX}/{run_id}/{checkpoint_number}.json"


def _checkpoint_prefix(run_id: UUID) -> str:
    """S3 prefix for all checkpoints of a run."""
    return f"{_CHECKPOINT_PREFIX}/{run_id}/"


async def write_checkpoint(checkpoint: SessionCheckpoint) -> str:
    """
    Write a session checkpoint to S3.

    Args:
        checkpoint: Frozen SessionCheckpoint record.

    Returns:
        The S3 key where the checkpoint was written.
    """
    key = _checkpoint_key(checkpoint.run_id, checkpoint.checkpoint_number)

    payload = {
        "run_id": str(checkpoint.run_id),
        "task_id": str(checkpoint.task_id) if checkpoint.task_id else None,
        "messages": list(checkpoint.messages),
        "memory_snapshot": checkpoint.memory_snapshot,
        "tokens_used": checkpoint.tokens_used,
        "checkpoint_number": checkpoint.checkpoint_number,
        "created_at": checkpoint.created_at.isoformat(),
    }

    try:
        _s3_client().put_object(
            Bucket=_BUCKET,
            Key=key,
            Body=json.dumps(payload),
            ContentType="application/json",
        )
        logger.info(
            "Checkpoint written: run=%s checkpoint=%d key=%s",
            checkpoint.run_id,
            checkpoint.checkpoint_number,
            key,
        )
    except Exception as exc:
        logger.error("Failed to write checkpoint: %s", exc)
        raise

    return key


async def read_checkpoint(
    run_id: UUID,
    checkpoint_number: Optional[int] = None,
) -> Optional[dict]:
    """
    Read a checkpoint from S3.

    Args:
        run_id: The orchestration run ID.
        checkpoint_number: Specific checkpoint number, or None for latest.

    Returns:
        Checkpoint data as dict, or None if not found.
    """
    try:
        if checkpoint_number is not None:
            key = _checkpoint_key(run_id, checkpoint_number)
            response = _s3_client().get_object(Bucket=_BUCKET, Key=key)
            return json.loads(response["Body"].read())

        # Find latest checkpoint — paginate to handle >1000
        prefix = _checkpoint_prefix(run_id)
        client = _s3_client()
        all_keys: list[str] = []
        continuation_token = None

        while True:
            kwargs = {"Bucket": _BUCKET, "Prefix": prefix}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            response = client.list_objects_v2(**kwargs)
            all_keys.extend(obj["Key"] for obj in response.get("Contents", []))
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")

        if not all_keys:
            return None

        latest_key = sorted(all_keys)[-1]
        obj = client.get_object(Bucket=_BUCKET, Key=latest_key)
        return json.loads(obj["Body"].read())

    except Exception as exc:
        logger.warning("Checkpoint read failed for run=%s: %s", run_id, exc)
        return None


async def list_checkpoints(run_id: UUID) -> list[dict]:
    """
    List all available checkpoints for a run.

    Extracts checkpoint numbers from S3 keys to avoid N+1 GET calls.
    Paginates to handle >1000 checkpoints.

    Returns:
        List of {checkpoint_number, s3_key, last_modified, size} dicts.
    """
    try:
        prefix = _checkpoint_prefix(run_id)
        client = _s3_client()
        results = []
        continuation_token = None

        while True:
            kwargs = {"Bucket": _BUCKET, "Prefix": prefix}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token

            response = client.list_objects_v2(**kwargs)

            for obj in response.get("Contents", []):
                key = obj["Key"]
                # Extract checkpoint number from key: checkpoints/{run_id}/{number}.json
                filename = key.rsplit("/", 1)[-1]
                try:
                    cp_number = int(filename.replace(".json", ""))
                except (ValueError, AttributeError):
                    continue
                results.append({
                    "checkpoint_number": cp_number,
                    "s3_key": key,
                    "last_modified": obj.get("LastModified", "").isoformat()
                    if hasattr(obj.get("LastModified", ""), "isoformat")
                    else str(obj.get("LastModified", "")),
                    "size": obj.get("Size", 0),
                })

            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")

        return sorted(results, key=lambda x: x.get("checkpoint_number", 0))

    except Exception as exc:
        logger.warning("Checkpoint list failed for run=%s: %s", run_id, exc)
        return []
