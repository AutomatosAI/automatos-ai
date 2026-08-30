"""PRD-197 S3 — Qdrant memory snapshots (the memory planes' DR arm).

Snapshots the two memory collections — ``durable_memory`` (PRD-187 L3) and
``field_memory`` (the shared field) — on the running Qdrant node, uploads
each snapshot to the platform object store, and prunes both sides to the
retention window. **Memory planes only**: the document plane is S3 Vectors
and its DR is PRD-186's.

Built to the PRD-197 §8-Q3 proposal (daily, 7-day retention, the object
store the platform already uses) — the knobs live in ``config.py``
(``MEMORY_SNAPSHOT_*``). Restore is documented in
``docs/runbooks/DR-qdrant.md``.

Scheduling rides ``services/memory_jobs.py`` (``MemoryJobScheduler``);
this module owns the cycle so it stays testable with every external
boundary (Qdrant client, HTTP download, boto3) injectable.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from config import config

logger = logging.getLogger(__name__)

# The two memory-plane collections. field_memory's name is a module constant
# (modules/context/adapters/vector_field.py SHARED_COLLECTION) — imported
# lazily in _collections() so importing this module never drags the adapter
# (and its embedding stack) into contexts that only need the pure helpers.
_S3_KEY_TIME_FORMAT = "%Y%m%dT%H%M%SZ"


def _collections() -> List[str]:
    from modules.context.adapters.vector_field import SHARED_COLLECTION

    return [config.DURABLE_MEMORY_COLLECTION, SHARED_COLLECTION]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without any client)
# ---------------------------------------------------------------------------

def parse_snapshot_time(creation_time: Optional[str]) -> Optional[datetime]:
    """Qdrant reports snapshot ``creation_time`` as an ISO string (no tz —
    the node writes UTC). Returns an aware UTC datetime, or None if absent
    or unparseable (callers treat unparseable as 'do not prune')."""
    if not creation_time:
        return None
    try:
        parsed = datetime.fromisoformat(creation_time)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def select_snapshots_to_prune(
    snapshots: List[Any],
    now: datetime,
    retention_days: int,
) -> List[str]:
    """Names of node-side snapshots older than the retention window.

    ``snapshots`` are qdrant ``SnapshotDescription``-shaped objects
    (``.name`` + ``.creation_time``). Undated snapshots are kept — a
    missing timestamp must never cause data loss.
    """
    cutoff = now - timedelta(days=retention_days)
    stale: List[str] = []
    for snap in snapshots:
        created = parse_snapshot_time(getattr(snap, "creation_time", None))
        if created is not None and created < cutoff:
            stale.append(snap.name)
    return stale


def object_key(prefix: str, collection: str, snapshot_name: str) -> str:
    """Deterministic object-store key for one uploaded snapshot."""
    return f"{prefix.strip('/')}/{collection}/{snapshot_name}"


# ---------------------------------------------------------------------------
# Boundary builders (overridable in tests)
# ---------------------------------------------------------------------------

def _build_qdrant_client():
    from qdrant_client import AsyncQdrantClient

    return AsyncQdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY or None,
        timeout=120,  # snapshot creation is heavier than a search
    )


def _build_s3_client():
    """The platform S3 client (core.storage, PRD-233 S4) — MinIO locally, AWS in SaaS."""
    from core.storage import get_s3_client

    return get_s3_client()


async def _download_snapshot(collection: str, snapshot_name: str) -> bytes:
    """Pull the snapshot file off the Qdrant node over its REST API.

    The client library exposes create/list/delete but not download; the
    file itself is served at /collections/{c}/snapshots/{name}. Memory
    collections are small (thousands of points), so buffering in memory
    is fine here — revisit only if a collection outgrows that.
    """
    import httpx

    headers = {"api-key": config.QDRANT_API_KEY} if config.QDRANT_API_KEY else {}
    url = f"{config.QDRANT_URL.rstrip('/')}/collections/{collection}/snapshots/{snapshot_name}"
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.content


# ---------------------------------------------------------------------------
# The cycle
# ---------------------------------------------------------------------------

async def run_snapshot_cycle(
    qdrant_client=None,
    s3_client=None,
    download: Optional[Callable[[str, str], Any]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Dict[str, Any]]:
    """One snapshot pass over both memory collections.

    Per collection: create a node-side snapshot → upload it to the object
    store → prune node-side snapshots and object-store copies past the
    retention window. Fail-soft per collection: one collection's failure
    is logged and must not stop the other's snapshot.
    """
    qdrant = qdrant_client or _build_qdrant_client()
    s3 = s3_client or _build_s3_client()
    fetch = download or _download_snapshot
    now = now or datetime.now(timezone.utc)

    bucket = config.MEMORY_SNAPSHOT_S3_BUCKET or config.S3_DOCUMENTS_BUCKET
    prefix = config.MEMORY_SNAPSHOT_S3_PREFIX
    retention_days = config.MEMORY_SNAPSHOT_RETENTION_DAYS
    loop = asyncio.get_running_loop()
    # Self-create the snapshot bucket on MinIO (no-op on AWS) before the first put.
    from core.storage import ensure_bucket

    await loop.run_in_executor(None, lambda: ensure_bucket(bucket, s3))

    summary: Dict[str, Dict[str, Any]] = {}
    for collection in _collections():
        try:
            description = await qdrant.create_snapshot(collection_name=collection)
            snapshot_name = description.name

            payload = await fetch(collection, snapshot_name)
            key = object_key(prefix, collection, snapshot_name)
            await loop.run_in_executor(
                None,
                lambda b=payload, k=key: s3.put_object(
                    Bucket=bucket, Key=k, Body=b,
                    ContentType="application/octet-stream",
                ),
            )

            node_snapshots = await qdrant.list_snapshots(collection_name=collection)
            stale = select_snapshots_to_prune(node_snapshots, now, retention_days)
            for name in stale:
                await qdrant.delete_snapshot(
                    collection_name=collection, snapshot_name=name
                )

            pruned_s3 = await loop.run_in_executor(
                None, _prune_s3_copies, s3, bucket, prefix, collection, now, retention_days
            )

            summary[collection] = {
                "snapshot": snapshot_name,
                "uploaded_key": key,
                "bytes": len(payload),
                "pruned_node": stale,
                "pruned_s3": pruned_s3,
            }
            logger.info(
                "[QdrantSnapshots] %s: snapshot %s uploaded to s3://%s/%s "
                "(%d bytes; pruned %d node / %d s3)",
                collection, snapshot_name, bucket, key,
                len(payload), len(stale), len(pruned_s3),
            )
        except Exception as exc:  # fail-soft per collection
            summary[collection] = {"error": str(exc)}
            logger.warning(
                "[QdrantSnapshots] %s: snapshot cycle failed — %s",
                collection, exc,
            )
    return summary


def _prune_s3_copies(
    s3,
    bucket: str,
    prefix: str,
    collection: str,
    now: datetime,
    retention_days: int,
) -> List[str]:
    """Delete object-store snapshot copies past retention. Sync (runs in
    the executor). Best-effort: a listing failure returns [] and the next
    cycle retries."""
    cutoff = now - timedelta(days=retention_days)
    deleted: List[str] = []
    try:
        listing = s3.list_objects_v2(
            Bucket=bucket, Prefix=f"{prefix.strip('/')}/{collection}/"
        )
        for obj in listing.get("Contents", []):
            modified = obj.get("LastModified")
            if modified is not None and modified.astimezone(timezone.utc) < cutoff:
                s3.delete_object(Bucket=bucket, Key=obj["Key"])
                deleted.append(obj["Key"])
    except Exception as exc:
        logger.warning(
            "[QdrantSnapshots] %s: s3 prune skipped — %s", collection, exc
        )
    return deleted
