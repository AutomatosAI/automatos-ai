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

        # Find latest checkpoint
        prefix = _checkpoint_prefix(run_id)
        response = _s3_client().list_objects_v2(Bucket=_BUCKET, Prefix=prefix)

        contents = response.get("Contents", [])
        if not contents:
            return None

        # Sort by key (checkpoint number is in filename) and get latest
        latest = sorted(contents, key=lambda x: x["Key"])[-1]
        obj = _s3_client().get_object(Bucket=_BUCKET, Key=latest["Key"])
        return json.loads(obj["Body"].read())

    except _s3_client().__class__.__bases__[0].__subclasses__()[0] if False else Exception as exc:
        # Catch boto3 NoSuchKey or general errors
        logger.warning("Checkpoint read failed for run=%s: %s", run_id, exc)
        return None


async def list_checkpoints(run_id: UUID) -> list[dict]:
    """
    List all available checkpoints for a run.

    Returns:
        List of {checkpoint_number, task_id, created_at} dicts.
    """
    try:
        prefix = _checkpoint_prefix(run_id)
        response = _s3_client().list_objects_v2(Bucket=_BUCKET, Prefix=prefix)

        results = []
        for obj in response.get("Contents", []):
            try:
                data = json.loads(
                    _s3_client().get_object(Bucket=_BUCKET, Key=obj["Key"])["Body"].read()
                )
                results.append({
                    "checkpoint_number": data.get("checkpoint_number"),
                    "task_id": data.get("task_id"),
                    "created_at": data.get("created_at"),
                    "tokens_used": data.get("tokens_used"),
                })
            except Exception:
                continue

        return sorted(results, key=lambda x: x.get("checkpoint_number", 0))

    except Exception as exc:
        logger.warning("Checkpoint list failed for run=%s: %s", run_id, exc)
        return []
