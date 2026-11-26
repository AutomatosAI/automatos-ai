"""No file hash from a request path reaches a metric, a log line or a trace (PRD-264 FR-17).

The public detection route takes the SHA-256 of a visitor's file in its path
(``/api/public/provenance/{sha256}``). Kept raw, every checked hash would become its own
Prometheus label (unbounded series, readable on the unauthenticated ``/metrics``), and
would sit in the access log, the request-context log fields and the server span: a
record of which files people checked. Everything that copies a request path out of the
process runs it through :func:`redact_digest_segments` first, which collapses a path
segment of exactly 64 hex characters to ``/{sha256}``.

PRD-277 US-402 (§5.11): sign-in's callback carries the one-time code and the state in its query
(``/api/v1/auth/oidc/callback?code=…&state=…``), and uvicorn's access line logs the query. The
access-log filter also replaces the whole query of every ``/api/v1/auth/oidc/`` path with
``?{redacted}`` (:func:`redact_secret_queries`). Every other copy of a path (metrics, log fields,
spans) holds the path alone, and spans redact query values already.

#1642: a workspace webhook's key is its credential, and it is the last segment of its path
(``/api/webhooks/ws/{workspace_key}``, a 32-hex ``uuid4().hex``), so every copy of a path runs
through :func:`redact_path`, which collapses that segment to ``{workspace_key}`` as well as a
hash. The webhook's verification ``GET`` carries ``hub.verify_token`` in its query, and the
public invitation lookup (``/api/team/invitation-info?token=…``) its invitation's token, so the
access-log filter drops those queries as it drops sign-in's.

It imports nothing but ``logging`` and ``re``, like ``request_log_guard``, so the
logging and tracing setup can load it before the app's config.
"""
from __future__ import annotations

import logging
import re

# A whole path segment of 64 hex characters (a SHA-256), up to the next "/", "?", "#"
# or the end. Case-insensitive: the route refuses upper case, but the path still arrives.
_SHA256_SEGMENT = re.compile(r"/[0-9a-f]{64}(?=[/?#]|$)", re.IGNORECASE)
DIGEST_PLACEHOLDER = "/{sha256}"
# A workspace webhook's key: the whole segment after its prefix, whatever it holds (#1642).
_WEBHOOK_KEY_SEGMENT = re.compile(r"(/api/webhooks/ws/)[^/?#\s\"]+")
WEBHOOK_KEY_PLACEHOLDER = "{workspace_key}"
# A path whose query holds a credential (sign-in's code and state, a webhook's verify token, an
# invitation's token), then its query, up to the next quote, space or "#".
_SECRET_QUERY = re.compile(
    r"(/api/(?:v1/auth/oidc/|webhooks/ws/|team/invitation-info)[^?#\s\"]*)\?[^#\s\"]*")
QUERY_PLACEHOLDER = "?{redacted}"
# uvicorn's per-request access line carries the path and query (its args), and its
# logger does not propagate to the root handlers, so the filter sits on the logger.
ACCESS_LOGGERS = ("uvicorn.access",)


def redact_digest_segments(path: str) -> str:
    """``path`` with every 64-hex segment replaced by ``/{sha256}``; anything else kept."""
    return _SHA256_SEGMENT.sub(DIGEST_PLACEHOLDER, path)


def redact_path(path: str) -> str:
    """``path`` with no credential or hash in its segments: a webhook's key becomes
    ``{workspace_key}`` and a 64-hex segment ``{sha256}``; anything else kept."""
    return redact_digest_segments(_WEBHOOK_KEY_SEGMENT.sub(r"\g<1>" + WEBHOOK_KEY_PLACEHOLDER, path))


def redact_secret_queries(line: str) -> str:
    """``line`` with the query of every path that carries a credential in it (sign-in, a
    workspace webhook, the invitation lookup) replaced by ``?{redacted}``; anything else kept."""
    return _SECRET_QUERY.sub(lambda found: found.group(1) + QUERY_PLACEHOLDER, line)


def _redact_access(value: str) -> str:
    return redact_secret_queries(redact_path(value))


class DigestPathFilter(logging.Filter):
    """A logger filter: an access line keeps its method, status and client, never a hash, a
    webhook's key or a credential in a query.

    uvicorn logs ``'%s - "%s %s HTTP/%s" %d'`` with the path as one of its args, so each
    string arg (and a pre-formatted message) is redacted. It never drops a line.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(_redact_access(arg) if isinstance(arg, str) else arg for arg in record.args)
        if isinstance(record.msg, str):
            record.msg = _redact_access(record.msg)
        return True


def guard_access_log_digests() -> None:
    """Put :class:`DigestPathFilter` on each access logger, once. Called by
    ``setup_logging``; uvicorn configures its loggers before it imports the app, and a
    logger's filters outlive that, so the filter holds in every worker."""
    for name in ACCESS_LOGGERS:
        access_logger = logging.getLogger(name)
        if not any(isinstance(f, DigestPathFilter) for f in access_logger.filters):
            access_logger.addFilter(DigestPathFilter())


__all__ = [
    "ACCESS_LOGGERS",
    "DIGEST_PLACEHOLDER",
    "QUERY_PLACEHOLDER",
    "WEBHOOK_KEY_PLACEHOLDER",
    "DigestPathFilter",
    "guard_access_log_digests",
    "redact_digest_segments",
    "redact_path",
    "redact_secret_queries",
]
