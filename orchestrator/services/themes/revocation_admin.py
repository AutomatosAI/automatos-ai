"""PRD-261 C-05e — the admin's theme revocations: list, create, lift.

A revocation (``theme_revocations``, read on every serve by ``revocations.py``) names one
package digest, or a theme id with an optional version range, at ``warn`` (the theme still
runs and the host shows the reason) or ``disable`` (members see the theme's base and its files
answer 404). This module is the admin's hand on that table:

* **list**: every row, live ones first, then the newest, each with its ``origin``: ``local``
  (written in this install) or ``feed`` (pulled from the hosted revocation feed by the local
  edition's pull, C-05h);
* **create** writes a ``local`` row, and checks every field before anything is written, so the
  reader never has to guess about a stored row: a digest is 64 lower-case hex characters, a
  theme id matches ``manifest.ID_PATTERN``, a version range matches ``manifest.RANGE_PATTERN``
  (exact ``X.Y.Z`` or caret ``^X.Y.Z``; the reader counts an unreadable range as every version,
  so one is refused here instead of stored), the level is ``warn`` or ``disable`` and the reason
  is 1 to ``REASON_MAX`` characters of one line of plain text (``contract.plain_text_problem``:
  every member reads it in the host's notice). A request that repeats a live ``local`` row (the
  same target and level) answers that row instead of writing a second one, so a retry or a
  double click writes once. No index backs that check (as with ``review.deactivate``): two
  creates in the same instant can both write, and each then needs its own lift. A live ``feed``
  row is not a repeat: the admin's own row stands after the feed lifts its row;
* **lift** sets ``lifted_at`` on a ``local`` row, once (lifting a lifted row answers it as it
  is). A ``feed`` row is the hosted list's to lift: here it is a 409, and the local pull lifts
  it when the feed says ``liftedAt``.

Nothing is deleted: the hosted feed reports a lifted row for ``THEME_REVOCATIONS_LIFTED_DAYS``.
Synchronous; nothing here commits: the router does, once.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.themes import THEME_REVOCATION_LEVELS, THEME_REVOCATION_ORIGIN_LOCAL, ThemeRevocation
from services.themes.choices import BAD_REQUEST, CONFLICT, NOT_FOUND, ThemeInstallError
from services.themes.contract import plain_text_problem
from services.themes.manifest import ID_PATTERN, RANGE_PATTERN, SHA256_PATTERN

REASON_MAX = 2000
REASON_FIELD = "reason"
LIST_LIMIT_DEFAULT, LIST_LIMIT_MAX = 100, 500

ONE_TARGET = (
    "Name what to revoke: a package digest, or a theme id with an optional version range, not both."
)
RANGE_NEEDS_THEME = "A version range goes with a theme id. A digest names one package already."
BAD_DIGEST = "A package digest is 64 lower-case hex characters (its sha256)."
BAD_THEME_ID = "A theme id is publisher/slug in lower-case letters, digits and dashes, such as acme/ink."
BAD_RANGE = "A version range is an exact version such as 1.2.0, or a caret range such as ^1.0.0."
BAD_LEVEL = "The level is 'warn' (the theme still runs, with a notice) or 'disable' (members see the base)."
REVOCATION_NOT_FOUND = "That revocation doesn't exist."
FEED_ROW = (
    "This revocation came from the Automatos revocation feed, so it can't be lifted here. "
    "It lifts when the feed lifts it."
)


class RevocationAdminError(ThemeInstallError):
    """A refusal; the router answers its status and plain message."""


class NewRevocation(NamedTuple):
    """What the admin asked for, as the request carried it (checked by :func:`create_revocation`)."""

    digest: Optional[str]
    theme_id: Optional[str]
    version_range: Optional[str]
    level: str
    reason: str


class Target(NamedTuple):
    """A checked target: a digest alone, or a theme id with an optional version range."""

    digest: Optional[str]
    theme_id: Optional[str]
    version_range: Optional[str]


def _refused(message: str) -> RevocationAdminError:
    return RevocationAdminError(BAD_REQUEST, message)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def checked_target(new: NewRevocation) -> Target:
    """The target, or a 400: exactly one of digest and theme id, each in its pattern."""
    if (new.digest is None) == (new.theme_id is None):
        raise _refused(ONE_TARGET)
    if new.digest is not None:
        if new.version_range is not None:
            raise _refused(RANGE_NEEDS_THEME)
        if SHA256_PATTERN.fullmatch(new.digest) is None:
            raise _refused(BAD_DIGEST)
        return Target(new.digest, None, None)
    if ID_PATTERN.fullmatch(new.theme_id) is None:
        raise _refused(BAD_THEME_ID)
    if new.version_range is not None and RANGE_PATTERN.fullmatch(new.version_range) is None:
        raise _refused(BAD_RANGE)
    return Target(None, new.theme_id, new.version_range)


def checked_level(level: str) -> str:
    """``warn`` or ``disable`` (the table's CHECK), or a 400."""
    if level not in THEME_REVOCATION_LEVELS:
        raise _refused(BAD_LEVEL)
    return level


def checked_reason(reason: str) -> str:
    """The trimmed reason: one line of plain text of 1 to ``REASON_MAX`` characters, or a 400."""
    text = reason.strip()
    problem = plain_text_problem(text, REASON_FIELD, REASON_MAX)
    if problem:
        raise _refused(problem)
    return text


def _same(column: Any, value: Optional[str]) -> Any:
    return column.is_(None) if value is None else column == value


def _live_local_repeat(db: Session, target: Target, level: str) -> Optional[ThemeRevocation]:
    """The oldest live ``local`` row with this target and level, or None."""
    return (
        db.query(ThemeRevocation)
        .filter(
            ThemeRevocation.lifted_at.is_(None),
            ThemeRevocation.origin == THEME_REVOCATION_ORIGIN_LOCAL,
            ThemeRevocation.level == level,
            _same(ThemeRevocation.digest, target.digest),
            _same(ThemeRevocation.theme_id, target.theme_id),
            _same(ThemeRevocation.version_range, target.version_range),
        )
        .order_by(ThemeRevocation.created_at, ThemeRevocation.id)
        .first()
    )


def create_revocation(
    db: Session, new: NewRevocation, actor_pk: Optional[int],
) -> Tuple[ThemeRevocation, bool]:
    """Write a ``local`` revocation once every field checks out: ``(row, True)``. A live ``local``
    row with the same target and level answers instead: ``(that row, False)``, nothing written."""
    target = checked_target(new)
    level = checked_level(new.level)
    reason = checked_reason(new.reason)
    standing = _live_local_repeat(db, target, level)
    if standing is not None:
        return standing, False
    row = ThemeRevocation(
        digest=target.digest, theme_id=target.theme_id, version_range=target.version_range, level=level,
        reason=reason, created_by=actor_pk, created_at=_now(), origin=THEME_REVOCATION_ORIGIN_LOCAL,
    )
    db.add(row)
    db.flush()
    return row, True


def lift_revocation(db: Session, revocation_id: UUID) -> Tuple[ThemeRevocation, bool]:
    """Set ``lifted_at`` on a ``local`` row: ``(row, True)``; an already lifted one answers
    ``(row, False)`` unchanged. A missing row is a 404; a ``feed`` row a 409, unchanged."""
    row = db.query(ThemeRevocation).filter(ThemeRevocation.id == revocation_id).with_for_update().first()
    if row is None:
        raise RevocationAdminError(NOT_FOUND, REVOCATION_NOT_FOUND)
    if row.origin != THEME_REVOCATION_ORIGIN_LOCAL:
        raise RevocationAdminError(CONFLICT, FEED_ROW)
    if row.lifted_at is not None:
        return row, False
    row.lifted_at = _now()
    db.flush()
    return row, True


def _iso(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else value.isoformat()


def revocation_view(row: ThemeRevocation) -> Dict[str, Any]:
    """One row as the admin panel reads it (the feed's fields, plus ``origin``)."""
    return {
        "id": str(row.id),
        "digest": row.digest,
        "themeId": row.theme_id,
        "versionRange": row.version_range,
        "level": row.level,
        "reason": row.reason,
        "origin": row.origin,
        "createdAt": _iso(row.created_at),
        "liftedAt": _iso(row.lifted_at),
    }


def list_revocations(db: Session, limit: int) -> List[Dict[str, Any]]:
    """At most ``limit`` rows: the live ones first, then the newest first."""
    rows = (
        db.query(ThemeRevocation)
        .order_by(ThemeRevocation.lifted_at.isnot(None), ThemeRevocation.created_at.desc(), ThemeRevocation.id)
        .limit(limit)
        .all()
    )
    return [revocation_view(row) for row in rows]


__all__ = [
    "LIST_LIMIT_DEFAULT", "LIST_LIMIT_MAX", "NewRevocation", "REASON_MAX", "RevocationAdminError", "Target",
    "checked_level", "checked_reason", "checked_target", "create_revocation", "lift_revocation",
    "list_revocations", "revocation_view",
]
