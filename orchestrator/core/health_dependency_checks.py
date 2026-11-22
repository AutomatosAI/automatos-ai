"""The checks behind GET /health's dependency entries (PRD-282 US-202).

Each is a ping or a metadata call, made through the client core already uses for that
dependency, and says ``not configured`` without any call when the install has none:

* **Redis:** a PING on the shared client (core/redis/client.py, PRD-278).
* **Object storage:** HeadBucket on the documents bucket, with the storage factory's
  fast-fail client (core/storage/s3.py), the call start-up's own bucket probe makes.
* **The search index:** GetIndex on the shared S3 Vectors index, through the backend's
  own client; with a bucket per workspace there is no single index, so ListVectorBuckets.
  With S3 Vectors off, documents are searched in the database: no separate index.
* **The memory store:** a listing of its collections (``QDRANT_URL``), the call its
  stores make as their health check (core/qdrant/client.py).
* **The model provider:** never called. How the last real call in this process ended
  (core/llm/last_call.py); before one, whether the install's settings name a provider.

Each is synchronous: a Watch runs it on a thread of its own, with a time limit
(core/health_watch.py). Each returns a state and nothing else: no address, credential or
error text. A refusal or no answer reads as ``down``; anything unexpected raises, and the
Watch logs it and reads ``down`` too.
"""
from __future__ import annotations

import asyncio

from config import config
from core.health_readiness import UNCHECKED
from core.llm import last_call

UP, DOWN, NOT_CONFIGURED = "up", "down", "not configured"
_HEALTH_CHECK_WORKSPACE = "health-check"  # any id: it names a bucket only when there is one per workspace


def redis_state() -> str:
    """PING on the shared Redis client."""
    if not config.REDIS_URL:
        return NOT_CONFIGURED
    from redis.exceptions import RedisError

    from core.redis.client import get_redis_client

    client = get_redis_client()
    if client is None:
        return DOWN  # configured, but its client could not be set up (logged where it was tried)
    try:
        client.get_redis().ping()
    except (RedisError, OSError):
        return DOWN
    return UP


def object_storage_state() -> str:
    """HeadBucket on the documents bucket, no retries, bounded timeouts."""
    from core.storage.s3 import FAST_FAIL, get_s3_client, is_storage_configured

    if not is_storage_configured():
        return NOT_CONFIGURED
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        get_s3_client(FAST_FAIL).head_bucket(Bucket=config.S3_DOCUMENTS_BUCKET)
    except (BotoCoreError, ClientError):
        return DOWN
    return UP


def search_index_state() -> str:
    """GetIndex on the shared index, or ListVectorBuckets with a bucket per workspace."""
    if not config.S3_VECTORS_ENABLED:
        return NOT_CONFIGURED  # searched in the database: its entry is the database's
    from botocore.exceptions import BotoCoreError, ClientError

    from modules.search.vector_store.backends.s3_vectors_backend import S3VectorsBackend

    backend = S3VectorsBackend(workspace_id=_HEALTH_CHECK_WORKSPACE)
    try:
        if backend.bucket_name == config.S3_VECTORS_BUCKET:
            backend.client.get_index(vectorBucketName=backend.bucket_name, indexName=backend.index_name)
        else:
            backend.client.list_vector_buckets(maxResults=1)
    except (BotoCoreError, ClientError):
        return DOWN
    return UP


def memory_store_state() -> str:
    """A listing of the memory store's collections, on an event loop of this thread's own."""
    if not config.QDRANT_URL:
        return NOT_CONFIGURED
    return UP if asyncio.run(_memory_store_answers()) else DOWN


async def _memory_store_answers() -> bool:
    from core.qdrant.client import QdrantRestClient
    from core.qdrant.errors import ResponseHandlingException, UnexpectedResponse

    client = QdrantRestClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None,
                              timeout=config.HEALTH_DEPENDENCY_CHECK_TIMEOUT_SECONDS)
    try:
        await client.get_collections()
        return True
    except (ResponseHandlingException, UnexpectedResponse):
        return False
    finally:
        await client.aclose()


def model_provider_state() -> str:
    """How the last real model call ended; before one, ``unchecked`` when the install's
    settings name a provider, else ``not configured``. Never a call."""
    call = last_call.last()
    if call is not None:
        return UP if call.succeeded else DOWN
    return UNCHECKED if _provider_in_settings() else NOT_CONFIGURED


def _provider_in_settings() -> bool:
    """A chat provider's key in the install's settings, or one the pod's own role signs for."""
    from core.llm.providers import chat_slugs, env_api_key, runs_on_workload_identity

    return any(env_api_key(slug) or runs_on_workload_identity(slug) for slug in chat_slugs())
