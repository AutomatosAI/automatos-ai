"""PRD-261 C-05g — the hosted revocation feed (plan §2.2.7, schema 1): what it carries, and its parser.

``GET /api/themes/revocations`` (``api/themes_revocations_feed.py``, hosted edition only) answers
``build_feed``::

    {"schema": 1, "generatedAt": <time>, "revokedKeys": [<keyId>, ...],
     "revocations": [{"id", "digest" | "themeId" + "versionRange", "level", "reason", "createdAt", "liftedAt"}]}

* **Only Marketplace revocations.** A row goes out when it covers a Marketplace package that was
  listed (``approved``, or ``deactivated`` after it was: ``serve.INSTALLED_MARKETPLACE_STATUSES``):
  a digest one of them has, or a theme id (and range) one of their versions falls in. A revocation
  that covers only a workspace's private upload, or a submission never approved, never appears:
  the feed is public, and a private upload's digest or id says something about a workspace. Only
  rows written here (``origin`` 'local') go out; a pulled row is never served again.
* **Lifted rows** stay in the feed for ``THEME_REVOCATIONS_LIFTED_DAYS`` after ``lifted_at``, so a
  local edition that pulls within that window lifts its copy.
* **Levels** go out as this server reads them (``revocations.effective_level``); a version range it
  can't read goes out as no range, which covers every version, as it does here.
* ``revokedKeys``: the ``revoked`` keys of ``signing-keys.json``, so a local edition distrusts a key
  before its own code ships the new keys file (C-05c reads them from ``theme_revocation_sync``).

``parse_feed`` is the local edition's reader (C-05h). It refuses the **whole** feed, with a plain
message that echoes nothing from it, for: more than ``THEME_REVOCATIONS_FEED_MAX_KB``; not UTF-8
JSON; a schema other than 1; more than ``FEED_MAX_ROWS`` rows; a row whose target is not exactly a
64-hex digest, or a theme id (``manifest.ID_PATTERN``) with an optional range
(``manifest.RANGE_PATTERN``); a reason that isn't one line of plain text of 1 to
``FEED_REASON_MAX`` characters (``contract.plain_text_problem``, the rule an admin's reason meets in
``revocation_admin.checked_reason``: every member reads it in the host's notice, so a bidi override
or a control character refuses the feed); an id that is not a UUID; a time without an offset; an
id twice; a key id that is not ``ed25519:`` and 16 hex. A level it doesn't know parses as
``disable`` (fail closed, as ``revocations.effective_level`` reads one). Every row ``build_feed``
writes passes the same row check (one that fails it is left out and logged), so
``parse_feed(feed_bytes(build_feed(db)))`` round-trips. A hosted reason goes out on one line
(``_one_line``): a deactivation's reason may hold line breaks (``review.plain_text``), and the row
must not be left out for them.

The feed is unsigned (QB-26, still open: this is the plan's default): HTTPS and this strict check
protect it, so an emergency revocation waits for no signer. Synchronous; nothing here writes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from sqlalchemy import or_
from sqlalchemy.orm import Session

from config import config
from core.models.themes import THEME_REVOCATION_LEVELS, THEME_REVOCATION_ORIGIN_LOCAL, ThemePackage, ThemeRevocation
from services.themes import signature_verify
from services.themes.contract import is_unsafe_char, plain_text_problem
from services.themes.manifest import ID_PATTERN, RANGE_PATTERN, satisfies_range
from services.themes.revocations import REVOKE_DISABLE, effective_level
from services.themes.serve import DIGEST_PATTERN, INSTALLED_MARKETPLACE_STATUSES, MARKETPLACE

logger = logging.getLogger(__name__)

FEED_SCHEMA = 1
FEED_MAX_ROWS = 10_000
FEED_MAX_KEYS = 1_000
FEED_REASON_MAX = 2000
REASON_FIELD = "reason"
# The reason a row goes out with when none of its own is left on one line (a stored reason of
# nothing but control or bidi characters).
FEED_REASON_FALLBACK = "Revoked by Automatos."
LIFTED_DAYS_SETTING = "THEME_REVOCATIONS_LIFTED_DAYS"
DEFAULT_LIFTED_DAYS = 365
FEED_MAX_KB_SETTING = "THEME_REVOCATIONS_FEED_MAX_KB"
DEFAULT_FEED_MAX_KB = 1024
BYTES_PER_KB = 1024
KEY_ID_PATTERN = re.compile(
    re.escape(signature_verify.KEY_ID_PREFIX) + "[0-9a-f]{%d}" % signature_verify.KEY_ID_HEX_CHARS,
)
WEAK_ETAG_PREFIX = "W/"
ANY_ETAG = "*"

FEED_WHERE = "The revocation feed"
ROW_WHERE = "Revocation {number} in the feed"
TOO_LARGE = "The revocation feed is larger than {kb} KB."
NOT_JSON = "The revocation feed isn't UTF-8 JSON."
NOT_SCHEMA = "The revocation feed isn't schema 1 ({schema: 1, generatedAt, revocations, revokedKeys})."
TOO_MANY = f"The revocation feed has more than {FEED_MAX_ROWS} revocations."
DUPLICATE_ID = "The revocation feed names one revocation id twice."
BAD_KEYS = f"The revocation feed's revokedKeys must be a list of at most {FEED_MAX_KEYS} ed25519 key ids."
ROW_SHAPE = ROW_WHERE + " isn't an object."
ROW_TARGET = ROW_WHERE + " must name a digest, or a theme id with an optional version range: one of them."
ROW_DIGEST = ROW_WHERE + " has a digest that isn't 64 lowercase hex."
ROW_THEME_ID = ROW_WHERE + " has a theme id that isn't publisher/slug."
ROW_RANGE = ROW_WHERE + " has a version range that isn't X.Y.Z or ^X.Y.Z."
ROW_ID = ROW_WHERE + " has an id that isn't a UUID."
ROW_REASON = ROW_WHERE + f" has a reason that isn't one line of plain text of 1 to {FEED_REASON_MAX} characters."
BAD_TIME = "{where} has a {field} that isn't a time with an offset."


class FeedError(ValueError):
    """The feed can't be used; the message is plain and echoes nothing from the feed."""


@dataclass(frozen=True)
class FeedRevocation:
    """One revocation of a parsed feed, its times in UTC."""

    id: UUID
    digest: Optional[str]
    theme_id: Optional[str]
    version_range: Optional[str]
    level: str
    reason: str
    created_at: datetime
    lifted_at: Optional[datetime]


@dataclass(frozen=True)
class Feed:
    """A parsed feed: when the hosted edition built it, its revocations and its revoked key ids."""

    generated_at: datetime
    revocations: Tuple[FeedRevocation, ...]
    revoked_keys: Tuple[str, ...]


@dataclass(frozen=True)
class _Listed:
    """The Marketplace packages that were listed: their digests, and each theme id's versions."""

    digests: frozenset
    versions: Mapping[str, Tuple[str, ...]]


def lifted_days() -> int:
    """``THEME_REVOCATIONS_LIFTED_DAYS``: how long a lifted revocation stays in the feed."""
    return int(getattr(config, LIFTED_DAYS_SETTING, DEFAULT_LIFTED_DAYS))


def feed_max_bytes() -> int:
    """``THEME_REVOCATIONS_FEED_MAX_KB`` in bytes: the largest feed a local edition reads."""
    return int(getattr(config, FEED_MAX_KB_SETTING, DEFAULT_FEED_MAX_KB)) * BYTES_PER_KB


def as_utc(moment: datetime) -> datetime:
    """``moment`` in UTC; a naive one (SQLite reads them back so) is taken as UTC."""
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)


def iso_time(moment: Optional[datetime]) -> Optional[str]:
    """ISO 8601 with the UTC offset, or None."""
    return None if moment is None else as_utc(moment).isoformat()


# --------------------------------------------------------------------------- the feed, built


def _listed(db: Session) -> _Listed:
    rows = (
        db.query(ThemePackage.digest, ThemePackage.theme_id, ThemePackage.version)
        .filter(ThemePackage.owner_type == MARKETPLACE, ThemePackage.status.in_(INSTALLED_MARKETPLACE_STATUSES))
        .all()
    )
    versions: Dict[str, List[str]] = {}
    for _digest, theme_id, version in rows:
        versions.setdefault(theme_id, []).append(version)
    return _Listed(frozenset(row[0] for row in rows), {key: tuple(value) for key, value in versions.items()})


def _candidates(db: Session, moment: datetime) -> List[ThemeRevocation]:
    cutoff = moment - timedelta(days=lifted_days())
    return (
        db.query(ThemeRevocation)
        .filter(ThemeRevocation.origin == THEME_REVOCATION_ORIGIN_LOCAL)
        .filter(or_(ThemeRevocation.lifted_at.is_(None), ThemeRevocation.lifted_at >= cutoff))
        .order_by(ThemeRevocation.created_at, ThemeRevocation.id)
        .all()
    )


def _feed_range(row: ThemeRevocation) -> Optional[str]:
    """The row's range; one this code can't read goes out as none (every version), as it reads here."""
    if row.version_range is None or RANGE_PATTERN.fullmatch(row.version_range) is not None:
        return row.version_range
    logger.warning("Theme revocation %s has a version range that can't be read: the feed sends every version", row.id)
    return None


def _covers_listed(row: ThemeRevocation, version_range: Optional[str], listed: _Listed) -> bool:
    if row.digest is not None:
        return row.digest in listed.digests
    versions = listed.versions.get(row.theme_id, ())
    return any(version_range is None or satisfies_range(version_range, version) for version in versions)


def _one_line(reason: Optional[str]) -> str:
    """``reason`` as the feed sends it: each run of unsafe characters (``contract.is_unsafe_char``: a
    line break, a bidi override) and spaces one space, cut to ``FEED_REASON_MAX`` characters."""
    spaced = "".join(" " if is_unsafe_char(char) else char for char in reason or "")
    line = " ".join(spaced.split())[:FEED_REASON_MAX].rstrip()
    return line or FEED_REASON_FALLBACK


def _wire_row(row: ThemeRevocation, version_range: Optional[str]) -> Dict[str, Any]:
    target = {"digest": row.digest} if row.digest is not None else {
        "themeId": row.theme_id, "versionRange": version_range,
    }
    return {
        "id": str(row.id), **target, "level": effective_level(row), "reason": _one_line(row.reason),
        "createdAt": iso_time(row.created_at), "liftedAt": iso_time(row.lifted_at),
    }


def _feed_row(row: ThemeRevocation, listed: _Listed) -> Optional[Dict[str, Any]]:
    """The row as the feed sends it, or None when it covers no listed package or fails the row check."""
    version_range = None if row.digest is not None else _feed_range(row)
    if not _covers_listed(row, version_range, listed):
        return None
    wire = _wire_row(row, version_range)
    try:
        _parse_row(wire, 1)
    except FeedError as refused:
        logger.error("Theme revocation %s is left out of the feed: %s", row.id, refused)
        return None
    return wire


def revoked_key_ids(keys: Optional[Sequence[signature_verify.SigningKey]] = None) -> List[str]:
    """The ids of the ``revoked`` keys in the keys file (``SigningKeysError`` when it doesn't load)."""
    held = signature_verify.load_keys() if keys is None else keys
    return sorted(key.key_id for key in held if key.status == signature_verify.STATUS_REVOKED)


def build_feed(db: Session, now: Optional[datetime] = None) -> Dict[str, Any]:
    """The feed: every live or recently lifted revocation that covers a listed Marketplace package."""
    moment = as_utc(now or datetime.now(timezone.utc)).replace(microsecond=0)
    listed = _listed(db)
    rows = [wire for wire in (_feed_row(row, listed) for row in _candidates(db, moment)) if wire is not None]
    if len(rows) > FEED_MAX_ROWS:
        logger.error("The revocation feed has %d rows, more than the %d a local edition reads",
                     len(rows), FEED_MAX_ROWS)
    return {"schema": FEED_SCHEMA, "generatedAt": moment.isoformat(), "revocations": rows,
            "revokedKeys": revoked_key_ids()}


def feed_bytes(feed: Mapping[str, Any]) -> bytes:
    """The feed as the route sends it: compact UTF-8 JSON."""
    return json.dumps(feed, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def feed_etag(feed: Mapping[str, Any]) -> str:
    """A weak ETag over everything but ``generatedAt``: equal while no revocation or key changes."""
    stable = {key: value for key, value in feed.items() if key != "generatedAt"}
    canonical = json.dumps(stable, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f'{WEAK_ETAG_PREFIX}"{hashlib.sha256(canonical).hexdigest()}"'


def etag_matches(if_none_match: Optional[str], etag: str) -> bool:
    """True when an ``If-None-Match`` list names ``etag`` (weak comparison) or is ``*``."""
    if not if_none_match:
        return False
    opaque = etag.removeprefix(WEAK_ETAG_PREFIX)
    tags = (tag.strip() for tag in if_none_match.split(","))
    return any(tag == ANY_ETAG or tag.removeprefix(WEAK_ETAG_PREFIX) == opaque for tag in tags)


# --------------------------------------------------------------------------- the feed, parsed


def parse_feed(data: bytes, max_bytes: Optional[int] = None) -> Feed:
    """The feed's bytes, checked whole: one bad row refuses all of it (``FeedError``)."""
    limit = feed_max_bytes() if max_bytes is None else max_bytes
    if len(data) > limit:
        raise FeedError(TOO_LARGE.format(kb=limit // BYTES_PER_KB))
    raw = _json_object(data)
    rows = raw.get("revocations")
    if not _is_schema_one(raw.get("schema")) or not isinstance(rows, list):
        raise FeedError(NOT_SCHEMA)
    if len(rows) > FEED_MAX_ROWS:
        raise FeedError(TOO_MANY)
    revocations = tuple(_parse_row(row, number) for number, row in enumerate(rows, start=1))
    if len({item.id for item in revocations}) != len(revocations):
        raise FeedError(DUPLICATE_ID)
    generated_at = _moment(raw.get("generatedAt"), FEED_WHERE, "generatedAt")
    return Feed(generated_at, revocations, _revoked_keys(raw.get("revokedKeys")))


def _json_object(data: bytes) -> Dict[str, Any]:
    try:
        raw = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError) as bad:
        raise FeedError(NOT_JSON) from bad
    if not isinstance(raw, dict):
        raise FeedError(NOT_SCHEMA)
    return raw


def _is_schema_one(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == FEED_SCHEMA


def _matches(pattern: re.Pattern, value: Any) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _parse_row(row: Any, number: int) -> FeedRevocation:
    if not isinstance(row, dict):
        raise FeedError(ROW_SHAPE.format(number=number))
    digest, theme_id, version_range = _target(row, number)
    where = ROW_WHERE.format(number=number)
    return FeedRevocation(
        id=_row_id(row.get("id"), number), digest=digest, theme_id=theme_id, version_range=version_range,
        level=_level(row.get("level"), number), reason=_reason(row.get("reason"), number),
        created_at=_moment(row.get("createdAt"), where, "createdAt"),
        lifted_at=None if row.get("liftedAt") is None else _moment(row.get("liftedAt"), where, "liftedAt"),
    )


def _target(row: Mapping[str, Any], number: int) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``(digest, themeId, versionRange)``: exactly a digest, or a theme id with an optional range."""
    digest, theme_id, version_range = row.get("digest"), row.get("themeId"), row.get("versionRange")
    if (digest is None) == (theme_id is None) or (digest is not None and version_range is not None):
        raise FeedError(ROW_TARGET.format(number=number))
    if digest is not None:
        if not _matches(DIGEST_PATTERN, digest):
            raise FeedError(ROW_DIGEST.format(number=number))
        return digest, None, None
    if not _matches(ID_PATTERN, theme_id):
        raise FeedError(ROW_THEME_ID.format(number=number))
    if version_range is not None and not _matches(RANGE_PATTERN, version_range):
        raise FeedError(ROW_RANGE.format(number=number))
    return None, theme_id, version_range


def _row_id(value: Any, number: int) -> UUID:
    try:
        parsed = UUID(value) if isinstance(value, str) else None
    except ValueError:
        parsed = None
    if parsed is None or str(parsed) != value:
        raise FeedError(ROW_ID.format(number=number))
    return parsed


def _level(value: Any, number: int) -> str:
    """The row's level; one this code doesn't know is ``disable`` (fail closed) and logged."""
    if isinstance(value, str) and value in THEME_REVOCATION_LEVELS:
        return value
    logger.warning("Revocation %d in the feed has a level this code doesn't know: read as disable", number)
    return REVOKE_DISABLE


def _reason(value: Any, number: int) -> str:
    """The row's reason, as sent: one line of plain text of 1 to ``FEED_REASON_MAX`` characters."""
    if plain_text_problem(value, REASON_FIELD, FEED_REASON_MAX) is not None:
        raise FeedError(ROW_REASON.format(number=number))
    return value


def _moment(value: Any, where: str, field: str) -> datetime:
    """An ISO 8601 time with an offset, in UTC."""
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        parsed = None
    if parsed is None or parsed.tzinfo is None:
        raise FeedError(BAD_TIME.format(where=where, field=field))
    return parsed.astimezone(timezone.utc)


def _revoked_keys(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list) or len(value) > FEED_MAX_KEYS:
        raise FeedError(BAD_KEYS)
    if not all(_matches(KEY_ID_PATTERN, key_id) for key_id in value):
        raise FeedError(BAD_KEYS)
    return tuple(sorted(set(value)))


__all__ = [
    "DEFAULT_FEED_MAX_KB", "DEFAULT_LIFTED_DAYS", "FEED_MAX_KB_SETTING", "FEED_MAX_ROWS", "FEED_REASON_MAX",
    "FEED_SCHEMA", "LIFTED_DAYS_SETTING", "Feed", "FeedError", "FeedRevocation", "as_utc", "build_feed",
    "etag_matches", "feed_bytes", "feed_etag", "feed_max_bytes", "iso_time", "lifted_days", "parse_feed",
    "revoked_key_ids",
]
