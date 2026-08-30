"""
S3 client factory — the ONE constructor of object-storage clients.
==================================================================

PRD-233 S4 (absorbs PRD-151). S3's API *is* the storage interface: there is no
``StorageProvider`` abstraction, only one place that turns ``config`` into a
``boto3`` S3 client. Every call site in ``orchestrator/`` obtains its client
here; a source guard (``tests/test_prd233_s4_storage_factory.py``) fails the
build on any stray ``boto3.client(`` outside this module.

Two editions, one code path:

* **SaaS** — ``S3_ENDPOINT_URL`` unset ⇒ real AWS S3 with the same kwargs the
  legacy per-site constructors passed (region from ``AWS_REGION``, explicit
  ``AWS_*`` credentials when both are set, SigV4, adaptive retries, no
  ``endpoint_url``, virtual-host addressing).
* **Local** — ``S3_ENDPOINT_URL=http://minio:9000`` ⇒ the same client pointed
  at MinIO with path-style addressing; buckets self-create on first use;
  presigned URLs handed to a browser are minted against
  ``S3_PUBLIC_ENDPOINT_URL`` so ``localhost:9000`` links resolve outside the
  compose network.

Nothing here touches the network at import time (PRD-151 FR-4): ``boto3`` is
imported lazily inside the builder and clients are created on first use.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

from config import config

logger = logging.getLogger(__name__)

# The region every legacy site fell back to when ``AWS_REGION`` was blank.
DEFAULT_REGION = "us-east-1"
# SigV4 — what every presign-capable site already forced; wire-identical to
# botocore's S3 default for standard calls (verified against botocore 1.43).
SIGNATURE_VERSION = "v4"
# Bucket-create errors that mean "already there" — idempotent create.
_BUCKET_EXISTS_CODES = frozenset({"BucketAlreadyOwnedByYou", "BucketAlreadyExists"})
# head_bucket errors that mean "missing" (AWS says 404, MinIO says NoSuchBucket).
_BUCKET_MISSING_CODES = frozenset({"404", "NoSuchBucket", "NotFound"})

STORAGE_NOT_CONFIGURED_MESSAGE = (
    "Object storage is not configured — set S3_ENDPOINT_URL (local: the MinIO "
    "service, http://minio:9000) or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY "
    "(AWS S3)."
)


class StorageNotConfigured(RuntimeError):
    """Raised on first storage use when neither an endpoint nor AWS creds exist.

    One typed error with one actionable message (PRD-151 G3) instead of a
    different ``NoCredentialsError`` stack trace per call site.
    """

    def __init__(self, message: str = STORAGE_NOT_CONFIGURED_MESSAGE) -> None:
        super().__init__(message)


@dataclass(frozen=True)
class ClientProfile:
    """Transport knobs a call site legitimately needs beyond the canonical client.

    The canonical profile (``STANDARD``) is the retry policy the platform's
    storage services already standardised on. ``FAST_FAIL`` preserves the
    PRD-164 DocumentManager contract: bounded timeouts, no retries, so a slow
    or unreachable object store fails a document op in seconds, never hangs it.
    """

    max_attempts: int = 3
    retry_mode: Optional[str] = "adaptive"
    connect_timeout: Optional[float] = None
    read_timeout: Optional[float] = None


STANDARD = ClientProfile()
FAST_FAIL = ClientProfile(max_attempts=1, retry_mode=None, connect_timeout=3, read_timeout=5)

_lock = threading.Lock()
_clients: Dict[Tuple[Optional[str], ClientProfile], Any] = {}
_ensured_buckets: Set[str] = set()


def is_storage_configured() -> bool:
    """True when the platform has somewhere to put objects.

    An endpoint (MinIO) or an explicit AWS key pair counts; the ambient boto
    credential chain alone does not — both editions configure storage through
    ``config``, and a silent fallback to ``~/.aws`` is exactly the kind of
    hidden dependency this factory exists to remove.
    """
    if config.S3_ENDPOINT_URL:
        return True
    return bool(config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY)


def _client_kwargs(*, endpoint_url: Optional[str], profile: ClientProfile) -> Dict[str, Any]:
    """The exact ``boto3.client("s3", ...)`` kwargs for one (endpoint, profile).

    With ``endpoint_url`` None and ``S3_USE_PATH_STYLE`` off this reproduces the
    legacy SaaS constructors: no ``endpoint_url`` kwarg, no ``s3`` addressing
    key, credentials only when ``config`` carries both halves.
    """
    from botocore.config import Config as BotoConfig

    retries: Dict[str, Any] = {"max_attempts": profile.max_attempts}
    if profile.retry_mode:
        retries["mode"] = profile.retry_mode
    boto_config: Dict[str, Any] = {
        "signature_version": SIGNATURE_VERSION,
        "retries": retries,
    }
    if profile.connect_timeout is not None:
        boto_config["connect_timeout"] = profile.connect_timeout
    if profile.read_timeout is not None:
        boto_config["read_timeout"] = profile.read_timeout
    if config.S3_USE_PATH_STYLE:
        boto_config["s3"] = {"addressing_style": "path"}

    kwargs: Dict[str, Any] = {
        "region_name": config.AWS_REGION or DEFAULT_REGION,
        "config": BotoConfig(**boto_config),
    }
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = config.AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = config.AWS_SECRET_ACCESS_KEY
    return kwargs


def _get_client(endpoint_url: Optional[str], profile: ClientProfile):
    key = (endpoint_url, profile)
    client = _clients.get(key)
    if client is not None:
        return client
    if not is_storage_configured():
        raise StorageNotConfigured()
    with _lock:
        client = _clients.get(key)
        if client is None:
            import boto3

            client = boto3.client("s3", **_client_kwargs(endpoint_url=endpoint_url, profile=profile))
            _clients[key] = client
            logger.info(
                "S3 client ready: endpoint=%s profile=%s path_style=%s",
                endpoint_url or "aws", "fast-fail" if profile is FAST_FAIL else "standard",
                bool(config.S3_USE_PATH_STYLE),
            )
    return client


def get_s3_client(profile: ClientProfile = STANDARD):
    """The process-wide S3 client for backend-side calls (memoized per profile).

    Raises :class:`StorageNotConfigured` when nothing is configured.
    """
    return _get_client(config.S3_ENDPOINT_URL or None, profile)


def get_public_s3_client(profile: ClientProfile = STANDARD):
    """The client to presign URLs that leave the backend (browser, LLM provider).

    A presigned URL signs its host, so a link a browser must open has to be
    minted against the host the browser can reach: ``S3_PUBLIC_ENDPOINT_URL``
    (``http://localhost:9000`` locally). Unset ⇒ the ordinary client — in SaaS
    the two are the same object.
    """
    public_endpoint = config.S3_PUBLIC_ENDPOINT_URL or None
    if not public_endpoint:
        return get_s3_client(profile)
    return _get_client(public_endpoint, profile)


def reset_s3_client() -> None:
    """Drop every memoized client and bucket memo (tests; config reloads)."""
    with _lock:
        _clients.clear()
        _ensured_buckets.clear()


def ensure_bucket(name: str, client=None) -> bool:
    """Idempotently create ``name`` on a configured endpoint (MinIO).

    Never creates on AWS: without ``S3_ENDPOINT_URL`` this is a no-op, so the
    SaaS bucket topology stays an infrastructure decision, not a side effect of
    the first upload. Returns True only when this call created the bucket.
    Memoized per process — one ``HeadBucket`` per bucket, then free.

    ``client`` lets a site probe with the client it will write with (the
    fast-fail DocumentManager client; an injected client under test); default
    is the standard factory client.
    """
    if not name or not config.S3_ENDPOINT_URL or name in _ensured_buckets:
        return False

    from botocore.exceptions import ClientError

    client = client if client is not None else get_s3_client()
    created = False
    try:
        client.head_bucket(Bucket=name)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code not in _BUCKET_MISSING_CODES:
            raise
        created = _create_bucket(client, name)
    _ensured_buckets.add(name)
    return created


def _create_bucket(client, name: str) -> bool:
    from botocore.exceptions import ClientError

    region = config.AWS_REGION or DEFAULT_REGION
    create_kwargs: Dict[str, Any] = {"Bucket": name}
    if region != DEFAULT_REGION:
        # us-east-1 rejects an explicit LocationConstraint; every other region
        # (and MinIO, which ignores it) accepts one.
        create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    try:
        client.create_bucket(**create_kwargs)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in _BUCKET_EXISTS_CODES:
            return False
        raise
    logger.info("Created bucket %r on %s", name, config.S3_ENDPOINT_URL)
    return True
