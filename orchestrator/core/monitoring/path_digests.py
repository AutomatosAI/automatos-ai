"""No file hash from a request path reaches a metric, a log line or a trace (PRD-264 FR-17).

The public detection route takes the SHA-256 of a visitor's file in its path
(``/api/public/provenance/{sha256}``). Kept raw, every checked hash would become its own
Prometheus label (unbounded series, readable on the unauthenticated ``/metrics``), and
would sit in the access log, the request-context log fields and the server span: a
record of which files people checked. Everything that copies a request path out of the
process runs it through :func:`redact_digest_segments` first, which collapses a path
segment of exactly 64 hex characters to ``/{sha256}``.

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
# uvicorn's per-request access line carries the path and query (its args), and its
# logger does not propagate to the root handlers, so the filter sits on the logger.
ACCESS_LOGGERS = ("uvicorn.access",)


def redact_digest_segments(path: str) -> str:
    """``path`` with every 64-hex segment replaced by ``/{sha256}``; anything else kept."""
    return _SHA256_SEGMENT.sub(DIGEST_PLACEHOLDER, path)


class DigestPathFilter(logging.Filter):
    """A logger filter: an access line keeps its method, status and client, never a hash.

    uvicorn logs ``'%s - "%s %s HTTP/%s" %d'`` with the path as one of its args, so each
    string arg (and a pre-formatted message) is redacted. It never drops a line.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                redact_digest_segments(arg) if isinstance(arg, str) else arg for arg in record.args
            )
        if isinstance(record.msg, str):
            record.msg = redact_digest_segments(record.msg)
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
    "DigestPathFilter",
    "guard_access_log_digests",
    "redact_digest_segments",
]
