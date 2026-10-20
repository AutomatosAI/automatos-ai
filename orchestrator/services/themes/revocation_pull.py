"""PRD-261 C-05h — the local edition's pull of the hosted revocation feed, and its sync status.

The local edition copies Automatos's Marketplace revocations (``revocation_feed``, plan §2.2.7)
into its own ``theme_revocations``, where ``revocations.py`` reads them on every serve, so a theme
Automatos revokes falls back to its base here on the next load.

* **Where from.** ``THEME_REVOCATIONS_URL`` (default ``DEFAULT_FEED_URL``, the hosted feed):
  ``https://``, or ``http://`` to a loopback host only; never a URL carrying a user name or
  password. Empty turns the pull off. QB-25 is still open; this is the plan's default: on, every
  ``THEME_REVOCATIONS_PULL_MINUTES`` (60), with no credentials (the request shows the instance's IP).
* **What is sent.** One ``GET`` with ``Accept``, ``Accept-Encoding: identity`` and a ``User-Agent``,
  from a client of its own: no ``Authorization``, ``Cookie`` or ``X-Workspace-ID``. Redirects are
  never followed (a 3xx is a failure). Each read times out after ``PULL_TIMEOUT_SECONDS`` and the
  whole read after ``PULL_DEADLINE_SECONDS``; the body is read as sent (an encoded one is refused)
  and refused past ``THEME_REVOCATIONS_FEED_MAX_KB``.
* **What changes.** Each feed row is upserted by its hosted id with ``origin`` 'feed': inserted when
  new, its target, level and reason refreshed when it is already a feed row. It is lifted only by
  an explicit ``liftedAt`` and never un-lifted; a row absent from the feed stays as it is (never
  deleted); a row with ``origin`` 'local' is never written, even under a feed id. ``revokedKeys``
  replaces ``theme_revocation_sync.revoked_key_ids``, which C-05c's signature check reads.
* **Fail closed.** Any failure (a refused URL, unreachable, a timeout, a status other than 200, a
  3xx, an encoded or oversize body, not JSON, a refused feed, a database error) rolls back every
  row change and records ``last_attempt_at`` and ``last_error`` (a plain sentence);
  ``last_success_at``, ``feed_generated_at`` and the revoked keys stay as they were.

``pull_feed`` runs from the scheduler (``services/theme_revocation_scheduler.py``) and from an
admin's Check now (``api/admin_theme_revocation_sync.py``), one pull at a time per process.
Synchronous; it commits.
"""

from __future__ import annotations

import ipaddress
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlsplit

import httpx
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config import config
from core.models.themes import (
    THEME_REVOCATION_ORIGIN_FEED,
    THEME_REVOCATION_SYNC_ROW_ID,
    ThemeRevocation,
    ThemeRevocationSync,
)
from services.themes.revocation_feed import (
    BYTES_PER_KB,
    TOO_LARGE,
    Feed,
    FeedError,
    FeedRevocation,
    as_utc,
    feed_max_bytes,
    iso_time,
    parse_feed,
)

logger = logging.getLogger(__name__)

URL_SETTING = "THEME_REVOCATIONS_URL"
DEFAULT_FEED_URL = "https://api.automatos.app/api/themes/revocations"
HTTPS_SCHEME, HTTP_SCHEME = "https", "http"
LOOPBACK_NAMES = frozenset({"localhost"})
PULL_TIMEOUT_SECONDS = 10.0
PULL_DEADLINE_SECONDS = 30.0
OK_STATUS = 200
REDIRECT_STATUSES = range(300, 400)
IDENTITY_ENCODINGS = frozenset({"", "identity"})
USER_AGENT = "Automatos-ThemeRevocations/1"
REQUEST_HEADERS = {"Accept": "application/json", "Accept-Encoding": "identity", "User-Agent": USER_AGENT}
STALE_AFTER = timedelta(days=7)
LAST_ERROR_MAX = 500

PULL_OFF = "Revocation sync is off: THEME_REVOCATIONS_URL is empty."
URL_NOT_HTTPS = "THEME_REVOCATIONS_URL must be an https:// URL (http:// only to this machine)."
URL_CREDENTIALS = "THEME_REVOCATIONS_URL must not carry a user name or password."
UNREACHABLE = "The revocation feed couldn't be reached ({error})."
TIMED_OUT = "The revocation feed didn't answer in time."
REDIRECTED = "The revocation feed answered a redirect ({status}); redirects aren't followed."
BAD_STATUS = "The revocation feed answered {status}."
ENCODED = "The revocation feed was sent encoded; only an unencoded body is read."
NOT_SAVED = "The pulled revocations couldn't be saved."

ClientFactory = Callable[[], httpx.Client]
_PULL_LOCK = threading.Lock()


class PullFailed(ValueError):
    """A pull that could not fetch the feed; the message is plain and echoes nothing it received."""


@dataclass(frozen=True)
class PullOutcome:
    """One pull: ``ok`` with the rows it wrote, or the plain ``error`` it recorded."""

    ok: bool
    rows: int = 0
    error: Optional[str] = None


def feed_url() -> str:
    """``THEME_REVOCATIONS_URL``, trimmed; empty turns the pull off."""
    return str(getattr(config, URL_SETTING, DEFAULT_FEED_URL) or "").strip()


def pull_enabled() -> bool:
    """True in the local edition with a feed URL set (the hosted edition serves the feed)."""
    return bool(config.IS_LOCAL_EDITION) and bool(feed_url())


def _is_loopback(host: str) -> bool:
    if host.lower() in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _allowed_target(scheme: str, host: Optional[str]) -> bool:
    """https to any host, http only to a loopback one."""
    if not host:
        return False
    return scheme == HTTPS_SCHEME or (scheme == HTTP_SCHEME and _is_loopback(host))


def checked_url(url: str) -> str:
    """``url`` when a pull may fetch it: https, or http to a loopback host, with no credentials."""
    if not url:
        raise PullFailed(PULL_OFF)
    try:
        parts = urlsplit(url)
        host = parts.hostname  # a malformed IPv6 host raises here
    except ValueError as bad:
        raise PullFailed(URL_NOT_HTTPS) from bad
    if parts.username is not None or parts.password is not None:
        raise PullFailed(URL_CREDENTIALS)
    if not _allowed_target(parts.scheme, host):
        raise PullFailed(URL_NOT_HTTPS)
    return url


# --------------------------------------------------------------------------- fetching


def _client() -> httpx.Client:
    """The HTTP client (a seam: tests give it a mock transport). Its own: no auth, no cookies, no redirects."""
    return httpx.Client(follow_redirects=False, timeout=PULL_TIMEOUT_SECONDS)


def _too_large(max_bytes: int) -> PullFailed:
    return PullFailed(TOO_LARGE.format(kb=max_bytes // BYTES_PER_KB))


def _read_capped(response: httpx.Response, max_bytes: int, deadline: float) -> bytes:
    """The body as sent, refused past ``max_bytes`` or the deadline."""
    if response.is_stream_consumed:  # a transport that loaded the body already: its bytes, capped
        if len(response.content) > max_bytes:
            raise _too_large(max_bytes)
        return response.content
    chunks: List[bytes] = []
    size = 0
    for chunk in response.iter_raw():  # each chunk as it arrives, so the cap bites early
        size += len(chunk)
        if size > max_bytes:
            raise _too_large(max_bytes)
        if time.monotonic() > deadline:
            raise PullFailed(TIMED_OUT)
        chunks.append(chunk)
    return b"".join(chunks)


def _read_answer(response: httpx.Response, max_bytes: int, deadline: float) -> bytes:
    status = response.status_code
    if status in REDIRECT_STATUSES:
        raise PullFailed(REDIRECTED.format(status=status))
    if status != OK_STATUS:
        raise PullFailed(BAD_STATUS.format(status=status))
    if (response.headers.get("content-encoding") or "").strip().lower() not in IDENTITY_ENCODINGS:
        raise PullFailed(ENCODED)
    declared = response.headers.get("content-length") or ""
    if declared.isdigit() and int(declared) > max_bytes:
        raise _too_large(max_bytes)
    return _read_capped(response, max_bytes, deadline)


def fetch_feed(url: str, max_bytes: int, client_factory: Optional[ClientFactory] = None) -> bytes:
    """GET the feed's bytes from ``url`` (already checked), never following a redirect."""
    deadline = time.monotonic() + PULL_DEADLINE_SECONDS
    try:
        with (client_factory or _client)() as client:
            with client.stream("GET", url, headers=REQUEST_HEADERS, follow_redirects=False) as response:
                return _read_answer(response, max_bytes, deadline)
    except httpx.InvalidURL as bad:  # not an HTTPError
        raise PullFailed(URL_NOT_HTTPS) from bad
    except httpx.TimeoutException as slow:
        raise PullFailed(TIMED_OUT) from slow
    except httpx.HTTPError as failed:
        raise PullFailed(UNREACHABLE.format(error=type(failed).__name__)) from failed


# --------------------------------------------------------------------------- applying


def sync_row(db: Session) -> ThemeRevocationSync:
    """The one ``theme_revocation_sync`` row, added when there is none yet."""
    row = db.get(ThemeRevocationSync, THEME_REVOCATION_SYNC_ROW_ID)
    if row is None:
        row = ThemeRevocationSync(id=THEME_REVOCATION_SYNC_ROW_ID, revoked_key_ids=[])
        db.add(row)
    return row


def _refreshed_fields(item: FeedRevocation) -> Dict[str, Any]:
    return {"digest": item.digest, "theme_id": item.theme_id, "version_range": item.version_range,
            "level": item.level, "reason": item.reason}


def _upsert(db: Session, item: FeedRevocation) -> bool:
    """Write one feed row by its hosted id; False when a local row holds that id (never written)."""
    existing = db.get(ThemeRevocation, item.id)
    if existing is None:
        db.add(ThemeRevocation(
            id=item.id, origin=THEME_REVOCATION_ORIGIN_FEED, created_at=item.created_at,
            lifted_at=item.lifted_at, **_refreshed_fields(item),
        ))
        return True
    if existing.origin != THEME_REVOCATION_ORIGIN_FEED:
        logger.warning("Feed revocation %s has the id of a local revocation: the local row is kept", item.id)
        return False
    for name, value in _refreshed_fields(item).items():
        if getattr(existing, name) != value:
            setattr(existing, name, value)
    if item.lifted_at is not None and existing.lifted_at is None:
        existing.lifted_at = item.lifted_at
    return True


def apply_feed(db: Session, feed: Feed, attempted_at: datetime) -> int:
    """Upsert every feed row and record the success; the caller commits. The number of rows written."""
    written = 0
    for item in feed.revocations:
        written += int(_upsert(db, item))
    sync = sync_row(db)
    sync.last_attempt_at = attempted_at
    sync.last_success_at = attempted_at
    sync.feed_generated_at = feed.generated_at
    sync.last_error = None
    sync.revoked_key_ids = list(feed.revoked_keys)
    return written


def _failed(db: Session, attempted_at: datetime, message: str) -> PullOutcome:
    """Roll every row change back and record the attempt and its error; nothing else changes."""
    db.rollback()
    logger.warning("Theme revocation pull failed: %s", message)
    try:
        sync = sync_row(db)
        sync.last_attempt_at = attempted_at
        sync.last_error = message[:LAST_ERROR_MAX]
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Theme revocation pull: the failure couldn't be recorded")
    return PullOutcome(ok=False, error=message)


def pull_feed(db: Session, client_factory: Optional[ClientFactory] = None) -> PullOutcome:
    """Fetch, check and apply the feed in one transaction; any failure changes no revocation."""
    with _PULL_LOCK:
        attempted_at = datetime.now(timezone.utc)
        try:
            data = fetch_feed(checked_url(feed_url()), feed_max_bytes(), client_factory)
            written = apply_feed(db, parse_feed(data), attempted_at)
            db.commit()
        except (PullFailed, FeedError) as refused:
            return _failed(db, attempted_at, str(refused))
        except SQLAlchemyError:
            logger.exception("Theme revocation pull: the pulled rows couldn't be saved")
            return _failed(db, attempted_at, NOT_SAVED)
    logger.info("Theme revocation pull: %d revocations from the feed", written)
    return PullOutcome(ok=True, rows=written)


# --------------------------------------------------------------------------- status


def stored_revoked_keys(db: Session) -> List[str]:
    """The revoked key ids the last successful pull stored (empty before one)."""
    row = db.get(ThemeRevocationSync, THEME_REVOCATION_SYNC_ROW_ID)
    return [] if row is None else list(row.revoked_key_ids or [])


def is_stale(last_success_at: Optional[datetime], now: datetime) -> bool:
    """True when no pull succeeded in the last ``STALE_AFTER`` (or ever)."""
    return last_success_at is None or as_utc(now) - as_utc(last_success_at) > STALE_AFTER


def sync_status(db: Session, now: Optional[datetime] = None) -> Dict[str, Any]:
    """What the admin's sync panel shows: the last attempt, success and error, the feed's rows here."""
    row = db.get(ThemeRevocationSync, THEME_REVOCATION_SYNC_ROW_ID)
    feed_rows = (
        db.query(func.count(ThemeRevocation.id))
        .filter(ThemeRevocation.origin == THEME_REVOCATION_ORIGIN_FEED)
        .scalar()
    )
    last_success_at = None if row is None else row.last_success_at
    return {
        "enabled": pull_enabled(),
        "lastAttemptAt": None if row is None else iso_time(row.last_attempt_at),
        "lastSuccessAt": iso_time(last_success_at),
        "lastError": None if row is None else row.last_error,
        "feedGeneratedAt": None if row is None else iso_time(row.feed_generated_at),
        "rows": int(feed_rows or 0),
        "revokedKeys": stored_revoked_keys(db),
        "stale": is_stale(last_success_at, now or datetime.now(timezone.utc)),
    }


__all__ = [
    "DEFAULT_FEED_URL", "PULL_DEADLINE_SECONDS", "PULL_OFF", "PULL_TIMEOUT_SECONDS", "REQUEST_HEADERS", "STALE_AFTER",
    "URL_SETTING", "ClientFactory", "PullFailed", "PullOutcome", "apply_feed", "checked_url", "feed_url",
    "fetch_feed", "is_stale", "pull_enabled", "pull_feed", "stored_revoked_keys", "sync_row", "sync_status",
]
