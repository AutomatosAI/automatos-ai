"""
Credentials from the pod's own cloud role — the start-up half (PRD-278 Wave 1)
=============================================================================

With ``AWS_USE_WORKLOAD_IDENTITY`` on, no AWS client in core is handed a key
pair: object storage (``core.storage.s3``), the search index
(``S3VectorsBackend``) and Bedrock (``core.llm.providers``) leave credentials to
the SDK's own chain, which in a pod finds the pod's role. This module checks
that choice before the API serves anything:

* **One source of credentials (US-101).** A key pair set beside the setting is
  refused: two sources is a mistake waiting to be found.
* **The role works (US-102).** One cheap call (``HeadBucket``) to the documents
  bucket, off the event loop and inside a short time limit, so a wrong role is
  found at install and not at the first upload. The failure names the bucket and
  the kind of failure, never a credential.

With the setting off nothing here runs and no call is made.
"""

from __future__ import annotations

import asyncio
import logging

from config import config

from .s3 import FAST_FAIL, get_s3_client

logger = logging.getLogger(__name__)

# The whole probe, credentials lookup included. The fast-fail client bounds the
# request itself (3 s to connect, 5 s to read, no retries).
PROBE_TIMEOUT_SECONDS = 10.0
_ACCESS_DENIED_CODES = frozenset({"403", "AccessDenied", "Forbidden"})
_BUCKET_MISSING_CODES = frozenset({"404", "NoSuchBucket", "NotFound"})

CONFLICTING_CREDENTIALS_MESSAGE = (
    "AWS_USE_WORKLOAD_IDENTITY is on and an AWS key (AWS_ACCESS_KEY_ID / "
    "AWS_SECRET_ACCESS_KEY) is also set. Use one source of AWS credentials: unset "
    "the key pair to use the pod's own role, or turn AWS_USE_WORKLOAD_IDENTITY off "
    "to use the key."
)


class WorkloadIdentityError(RuntimeError):
    """Start-up cannot run on the pod's role: a key pair beside it, or the bucket refused it."""


def assert_one_credential_source() -> None:
    """Raise when ``AWS_USE_WORKLOAD_IDENTITY`` is on and any part of a key pair is set."""
    if not config.AWS_USE_WORKLOAD_IDENTITY:
        return
    if config.AWS_ACCESS_KEY_ID or config.AWS_SECRET_ACCESS_KEY:
        raise WorkloadIdentityError(CONFLICTING_CREDENTIALS_MESSAGE)


async def verify_workload_identity() -> None:
    """Start-up: refuse a second credential source, then prove the documents bucket.

    A no-op with ``AWS_USE_WORKLOAD_IDENTITY`` off. Raises ``WorkloadIdentityError``
    (which aborts boot) on a conflict, a refusal, or no answer in time.
    """
    if not config.AWS_USE_WORKLOAD_IDENTITY:
        return
    assert_one_credential_source()
    bucket = config.S3_DOCUMENTS_BUCKET
    try:
        await asyncio.wait_for(asyncio.to_thread(_probe_bucket, bucket), timeout=PROBE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise _probe_failure(bucket, f"no answer within {PROBE_TIMEOUT_SECONDS:g} seconds") from exc
    logger.info("Workload identity: the pod's role reached the documents bucket '%s'", bucket)


def _probe_bucket(bucket: str) -> None:
    """One ``HeadBucket`` with the fast-fail client. Runs in a worker thread."""
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

    try:
        get_s3_client(FAST_FAIL).head_bucket(Bucket=bucket)
    except ClientError as exc:
        raise _probe_failure(bucket, _client_error_kind(exc)) from exc
    except NoCredentialsError as exc:
        raise _probe_failure(bucket, "no credentials: the SDK found no role for this pod") from exc
    except BotoCoreError as exc:
        raise _probe_failure(bucket, f"the request failed ({type(exc).__name__})") from exc


def _client_error_kind(exc) -> str:
    """The kind of refusal, from AWS's error code alone (never its message)."""
    code = str(exc.response.get("Error", {}).get("Code", ""))
    if code in _ACCESS_DENIED_CODES:
        return "access denied: the pod's role may not read this bucket"
    if code in _BUCKET_MISSING_CODES:
        return "the bucket does not exist"
    return f"refused with error code {code or 'unknown'}"


def _probe_failure(bucket: str, kind: str) -> WorkloadIdentityError:
    """The start-up failure for ``bucket``: names it and the kind of failure, logged once."""
    message = (
        f"AWS_USE_WORKLOAD_IDENTITY is on, but the documents bucket '{bucket}' "
        f"(S3_DOCUMENTS_BUCKET) could not be reached with the pod's role: {kind}."
    )
    logger.error(message)
    return WorkloadIdentityError(message)


__all__ = [
    "CONFLICTING_CREDENTIALS_MESSAGE",
    "PROBE_TIMEOUT_SECONDS",
    "WorkloadIdentityError",
    "assert_one_credential_source",
    "verify_workload_identity",
]
