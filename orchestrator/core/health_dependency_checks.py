"""The checks behind GET /health's dependency entries (PRD-282 US-202).

Each is a ping or a metadata call on a client built once and kept, and says
``not configured`` without any call (and without building a client) when the install
has none:

* **Redis:** a PING on a client of the check's own, built through PRD-278's factory
  (core/redis/connection.py) with the shared client's timeouts and no retries. The
  shared client (core/redis/client.py) is never touched, so the check never races its
  first-use set-up.
* **Object storage:** HeadBucket on the documents bucket, on the storage factory's
  fast-fail client (core/storage/s3.py).
* **The search index:** GetIndex on the shared S3 Vectors index; with a bucket per
  workspace there is no single index, so ListVectorBuckets. On the backend's kept
  metadata client (s3_vectors_backend.metadata_client). With S3 Vectors off, documents
  are searched in the database: no separate index.
* **The memory store:** a listing of its collections at ``QDRANT_URL``. Left at its
  default address (QDRANT_URL_DEFAULT), a store that does not answer is taken as one the
  install does not run: ``not configured``. Set to any other address, it is ``down``.
* **The model provider:** never called. How the last provider call through the LLM
  manager ended (core/llm/last_call.py), as a Reading stamped when that call ended;
  ``down`` only after FAILURES_BEFORE_DOWN failed calls in a row.

Each answer means one thing:

* ``up``: the dependency answered. A bucket or index that is not there is ``up`` too (the
  store answered) only where core makes it on first use: the documents bucket with
  ``S3_ENDPOINT_URL`` set (MinIO; core/storage/s3.py ensure_bucket never creates on AWS),
  the index with ``S3_VECTORS_CREATE_IF_MISSING`` on. Otherwise it is ``down``.
* ``not permitted``: the pod's role may not make the check: an explicit access-denied
  code (AccessDenied, AccessDeniedException). /health shows it as ``unchecked`` and
  /metrics has no series: neither up nor down.
* ``down``: anything else: no answer, a time-out, a refused connection, a 5xx, and a 403
  with no access-denied code. HeadBucket's 403 has no body, and is also what a wrong key,
  a bad signature or a revoked role gives, as are S3 Vectors' UnrecognizedClientException
  and InvalidSignatureException: those must reach an alert, so they are ``down``. A role
  without s3:ListBucket on the bucket gets that same bare 403, so it reads ``down`` too.

Each is synchronous: a Watch runs it on a thread of its own, with a time limit
(core/health_watch.py). Each returns a state and nothing else: no address, credential or
error text. Anything unexpected raises, and the Watch logs it and reads ``down``.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, FrozenSet, Union

from config import QDRANT_URL_DEFAULT, config
from core.health_readiness import UNCHECKED
from core.health_watch import Reading
from core.llm import last_call

UP, DOWN, NOT_CONFIGURED = "up", "down", "not configured"
NOT_PERMITTED = "not permitted"
# What a check the role may not make needs, for its one log line at the state change.
PERMISSIONS = {
    "object_storage": "the check is HeadBucket on S3_DOCUMENTS_BUCKET, which needs s3:ListBucket on that bucket",
    "search_index": ("the check is GetIndex on the shared index, which needs s3vectors:GetIndex, or with a "
                     "bucket per workspace ListVectorBuckets, which needs s3vectors:ListVectorBuckets"),
}
# The codes that say the role was refused, and nothing else: S3's and S3 Vectors'. Not a
# bare 403 (botocore's code for a HEAD's empty 403 is "403"), which bad credentials give too.
_DENIED_CODES: FrozenSet[str] = frozenset({"AccessDenied", "AccessDeniedException"})
_NOTHING_ANSWERED: FrozenSet[str] = frozenset()

_redis_lock = threading.Lock()
_redis_clients: Dict[Any, Any] = {}  # one per Redis target, built once under the lock


def redis_state() -> str:
    """PING on the check's own Redis client."""
    from core.redis.connection import redis_target_from_config

    try:
        target = redis_target_from_config()
    except ValueError:
        return DOWN  # set, but not a usable address (the shared client logs that, without it)
    if target is None:
        return NOT_CONFIGURED
    from redis.exceptions import RedisError

    try:
        _redis_client(target).ping()
    except (RedisError, OSError):
        return DOWN
    return UP


def _redis_client(target: Any) -> Any:
    client = _redis_clients.get(target)
    if client is not None:
        return client
    with _redis_lock:
        client = _redis_clients.get(target)
        if client is None:
            client = _new_redis_client(target)
            _redis_clients[target] = client
    return client


def _new_redis_client(target: Any) -> Any:
    from redis.backoff import NoBackoff
    from redis.retry import Retry

    from core.redis.client import REDIS_CONNECT_TIMEOUT_S, REDIS_SOCKET_TIMEOUT_S
    from core.redis.connection import new_sync_client

    # No retries: one ping that fails is the answer (redis-py's default retries would
    # multiply one time-out).
    return new_sync_client(target, socket_timeout=REDIS_SOCKET_TIMEOUT_S,
                           socket_connect_timeout=REDIS_CONNECT_TIMEOUT_S, retry=Retry(NoBackoff(), 0))


def object_storage_state() -> str:
    """HeadBucket on the documents bucket, one attempt, bounded timeouts."""
    from core.storage.s3 import FAST_FAIL, get_s3_client, is_storage_configured

    if not is_storage_configured():
        return NOT_CONFIGURED
    from botocore.exceptions import BotoCoreError, ClientError

    from core.storage.s3 import _BUCKET_MISSING_CODES

    # ensure_bucket makes a missing bucket only on a configured endpoint (MinIO), never on AWS.
    missing_is_up = _BUCKET_MISSING_CODES if config.S3_ENDPOINT_URL else _NOTHING_ANSWERED
    try:
        get_s3_client(FAST_FAIL).head_bucket(Bucket=config.S3_DOCUMENTS_BUCKET)
    except ClientError as exc:
        return refusal_state(exc, answered=missing_is_up)
    except BotoCoreError:
        return DOWN
    return UP


def search_index_state() -> str:
    """GetIndex on the shared index, or ListVectorBuckets with a bucket per workspace."""
    if not config.S3_VECTORS_ENABLED:
        return NOT_CONFIGURED  # searched in the database: its entry is the database's
    from botocore.exceptions import BotoCoreError, ClientError

    from modules.search.vector_store.backends.s3_vectors_backend import (
        _INDEX_MISSING_CODES,
        _WORKSPACE_PLACEHOLDER,
        metadata_client,
    )

    bucket = config.S3_VECTORS_BUCKET or ""
    missing_is_up = _INDEX_MISSING_CODES if config.S3_VECTORS_CREATE_IF_MISSING else _NOTHING_ANSWERED
    try:
        if _WORKSPACE_PLACEHOLDER in bucket:
            metadata_client().list_vector_buckets(maxResults=1)
        else:
            metadata_client().get_index(vectorBucketName=bucket, indexName=config.S3_VECTORS_INDEX_NAME)
    except ClientError as exc:
        return refusal_state(exc, answered=missing_is_up)
    except BotoCoreError:
        return DOWN
    return UP


def refusal_state(exc: Any, *, answered: FrozenSet[str]) -> str:
    """An AWS refusal's state, from its error code alone (never its message): an explicit
    access-denied code is ``not permitted``; a code in ``answered`` is ``up``; anything
    else, a bare 403 and a bad-credential code included, is ``down``."""
    code = str(exc.response.get("Error", {}).get("Code", ""))
    if code in _DENIED_CODES:
        return NOT_PERMITTED
    return UP if code in answered else DOWN


def memory_store_state() -> str:
    """A listing of the memory store's collections, on an event loop of this thread's own."""
    url = config.QDRANT_URL
    if not url:
        return NOT_CONFIGURED
    return asyncio.run(_memory_store_reply(url))


async def _memory_store_reply(url: str) -> str:
    from core.qdrant.client import QdrantRestClient
    from core.qdrant.errors import ResponseHandlingException, UnexpectedResponse

    client = QdrantRestClient(url=url, api_key=config.QDRANT_API_KEY or None,
                              timeout=config.HEALTH_DEPENDENCY_CHECK_TIMEOUT_SECONDS)
    try:
        await client.get_collections()
    except ResponseHandlingException:  # no answer
        return NOT_CONFIGURED if url == QDRANT_URL_DEFAULT else DOWN
    except UnexpectedResponse:  # it answered, with an error
        return DOWN
    finally:
        await client.aclose()
    return UP


def model_provider_state() -> Union[str, Reading]:
    """How the last provider call through the LLM manager ended, stamped when it ended;
    ``unchecked`` before one. Never a call, and never ``not configured``: a key may be
    stored in Settings, which only a decrypting read could tell."""
    call = last_call.last()
    if call is None:
        return UNCHECKED
    return Reading(UP if call.succeeded else DOWN, call.ended_at)
