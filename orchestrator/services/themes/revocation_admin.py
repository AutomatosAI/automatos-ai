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
  double click writes once. The partial unique index ``uq_theme_revocations_live_local`` backs
  that check: of two creates in the same instant one writes, the other's flush is refused, its
  transaction is rolled back and it answers the row that won. A live ``feed`` row is not a
  repeat: the admin's own row stands after the feed lifts its row;
* **lift** sets ``lifted_at`` on a ``local`` row, once (lifting a lifted row answers it as it
  is). A ``feed`` row is the hosted list's to lift while revocation sync works: then it is a
  409, and the local pull lifts it when the feed says ``liftedAt``. While sync is off
  (``revocation_pull.pull_enabled`` is false: the hosted edition, or no feed URL) or stale (no
  successful pull for ``revocation_pull.STALE_AFTER``), the admin may lift it here, and the
  router logs who did; a later pull never un-lifts it. The last live ``local`` disable covering
  a deactivated Marketplace version (``review.deactivate``) is a 409 too, whatever it was written
  for: a deactivation is not undone by lifting a revocation. Those versions are locked first, so
  two lifts of a version's last two disables can't both pass.

Nothing is deleted: the hosted feed reports a lifted row for ``THEME_REVOCATIONS_LIFTED_DAYS``.
Synchronous; nothing here commits: the router does, once. A create that loses the race to an
identical one rolls its transaction back (it had written nothing else).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from uuid import UUID

from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.models.themes import (
    THEME_REVOCATION_LEVELS,
    THEME_REVOCATION_ORIGIN_LOCAL,
    THEME_REVOCATION_SYNC_ROW_ID,
    ThemePackage,
    ThemeRevocation,
    ThemeRevocationSync,
)
from services.themes.choices import BAD_REQUEST, CONFLICT, MARKETPLACE_OWNER, NOT_FOUND, ThemeInstallError
from services.themes.contract import plain_text_problem
from services.themes.manifest import ID_PATTERN, RANGE_PATTERN, SHA256_PATTERN
from services.themes.review import STATUS_DEACTIVATED
from services.themes.revocations import REVOKE_DISABLE, covers

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
    "This revocation came from the Automatos revocation feed, and revocation sync is working, so it "
    "lifts when the feed lifts it. It can be lifted here only while sync is off or stale."
)
HOLDS_DEACTIVATION = (
    "This is the last revocation keeping {theme_id} {version}, a deactivated Marketplace version, "
    "disabled, so it can't be lifted. A deactivation isn't undone by lifting a revocation."
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
    """The live ``local`` row with this target and level, or None. The partial unique index
    ``uq_theme_revocations_live_local`` holds at most one such row, so the order only makes the
    answer certain; a create that races this check loses at its flush (:func:`create_revocation`)."""
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
    row with the same target and level answers instead: ``(that row, False)``, nothing written; so
    does one an identical create wrote in the same instant (the transaction is rolled back)."""
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
    try:
        db.flush()
    except IntegrityError:
        # An identical create committed between the check and this write: the index refused this one.
        db.rollback()
        winner = _live_local_repeat(db, target, level)
        if winner is None:
            raise
        return winner, False
    return row, True


def _covers(row: ThemeRevocation, package: ThemePackage) -> bool:
    """True when ``row`` covers ``package``, read as every revocation is (``revocations.covers``)."""
    return covers(row, package.digest, package.theme_id, package.version)


def _other_live_local_disable(db: Session, package: ThemePackage, row: ThemeRevocation) -> bool:
    """True when a live ``local`` disable besides ``row`` covers ``package``."""
    others = (
        db.query(ThemeRevocation)
        .filter(
            ThemeRevocation.id != row.id,
            ThemeRevocation.lifted_at.is_(None),
            ThemeRevocation.origin == THEME_REVOCATION_ORIGIN_LOCAL,
            ThemeRevocation.level == REVOKE_DISABLE,
            or_(ThemeRevocation.digest == package.digest, ThemeRevocation.theme_id == package.theme_id),
        )
        .all()
    )
    return any(_covers(other, package) for other in others)


def _held_deactivation(db: Session, row: ThemeRevocation) -> Optional[ThemePackage]:
    """A deactivated Marketplace version that no live ``local`` disable but ``row`` covers, or None.
    The deactivated versions ``row`` names are locked, so a concurrent lift of their other disable
    waits for this one and then sees it."""
    if row.level != REVOKE_DISABLE:
        return None
    named = ThemePackage.digest == row.digest if row.digest is not None else ThemePackage.theme_id == row.theme_id
    deactivated = (
        db.query(ThemePackage)
        .filter(ThemePackage.owner_type == MARKETPLACE_OWNER, ThemePackage.status == STATUS_DEACTIVATED, named)
        .order_by(ThemePackage.id)
        .with_for_update()
        .all()
    )
    for package in deactivated:
        if _covers(row, package) and not _other_live_local_disable(db, package, row):
            return package
    return None


def _feed_rows_liftable_here(db: Session) -> bool:
    """True while this install doesn't keep its ``feed`` rows in step with the hosted list: its pull
    is off (``revocation_pull.pull_enabled``) or stale (``revocation_pull.is_stale``, the sync panel's
    rule: no successful pull for ``STALE_AFTER``, or none yet)."""
    # Imported here, not at the top: revocation_pull sits in the revocations / signature_state /
    # revocation_feed import chain (a cycle there was W-C3 hand-off item 8); this module stays out of it.
    from services.themes import revocation_pull

    if not revocation_pull.pull_enabled():
        return True
    sync = db.get(ThemeRevocationSync, THEME_REVOCATION_SYNC_ROW_ID)
    return revocation_pull.is_stale(None if sync is None else sync.last_success_at, _now())


def lift_revocation(db: Session, revocation_id: UUID) -> Tuple[ThemeRevocation, bool]:
    """Set ``lifted_at`` on a ``local`` row, or on a ``feed`` row while sync is off or stale
    (:func:`_feed_rows_liftable_here`): ``(row, True)``; an already lifted one answers ``(row, False)``
    unchanged. A missing row is a 404; a ``feed`` row while sync works a 409, unchanged; so is the
    last live ``local`` disable of a deactivated Marketplace version (:func:`_held_deactivation`)."""
    row = db.query(ThemeRevocation).filter(ThemeRevocation.id == revocation_id).with_for_update().first()
    if row is None:
        raise RevocationAdminError(NOT_FOUND, REVOCATION_NOT_FOUND)
    local = row.origin == THEME_REVOCATION_ORIGIN_LOCAL
    if not local and not _feed_rows_liftable_here(db):
        raise RevocationAdminError(CONFLICT, FEED_ROW)
    if row.lifted_at is not None:
        return row, False
    held = _held_deactivation(db, row) if local else None
    if held is not None:
        raise RevocationAdminError(CONFLICT, HOLDS_DEACTIVATION.format(theme_id=held.theme_id, version=held.version))
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
