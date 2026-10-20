"""PRD-261 A-08a — is a theme package revoked? ``theme_revocations``, read on every serve.

A revocation names a package digest, or a theme id with an optional version range (an exact
``X.Y.Z`` or a caret ``^X.Y.Z``, the ranges ``engines`` takes, ``manifest.satisfies_range``).
Its level is ``warn`` (still runs; the host shows a notice) or ``disable`` (the files route
answers 404, the boot route the base boot, ``active`` the theme's base, and no install, update,
rollback or default may move to it). A lifted revocation (``lifted_at`` set) no longer counts.

Both readings fail closed and are logged: a version range this code can't read matches every
version, and a level it doesn't know counts as ``disable``.

This is the one revocation rule: the files and boot routes (``serve.py``), ``active``
(``defaults.py``) and the install routes (``updates.py``) all read it here.

PRD-261 C-05c (QB-28): a version whose stored signature doesn't check out (it fails, or its key
is revoked in the keys file or by the hosted feed; ``signature_state.py``) reads as a ``disable``
too. ``revocation_of`` answers it as a **transient** ``ThemeRevocation`` (``signature_disable``)
with ``signature_state.SIGNATURE_DISABLE_REASON``: built on each load, never added to a session,
so it is never stored and goes away once the version is re-signed. A stored ``disable`` keeps
its own reason, and the signature is then not checked.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import or_

from core.models.themes import ThemeRevocation
from services.themes.manifest import RANGE_PATTERN, satisfies_range
from services.themes.signature_state import SIGNATURE_DISABLE_REASON, signature_fails

logger = logging.getLogger(__name__)

REVOKE_WARN = "warn"
REVOKE_DISABLE = "disable"
_STRENGTH = {REVOKE_WARN: 1, REVOKE_DISABLE: 2}


def _applies(revocation: ThemeRevocation, digest: str, version: str) -> bool:
    """True when one revocation row covers this digest and version."""
    if revocation.digest is not None:
        return revocation.digest == digest
    version_range = revocation.version_range
    if version_range is None:
        return True
    if RANGE_PATTERN.fullmatch(version_range) is None:
        logger.warning(
            "Theme revocation %s has a version range that can't be read (%r): it covers every version of %s",
            revocation.id, version_range, revocation.theme_id,
        )
        return True
    return satisfies_range(version_range, version)


def effective_level(revocation: ThemeRevocation) -> str:
    """The row's level; one this code doesn't know counts as ``disable`` (fail closed)."""
    if revocation.level in _STRENGTH:
        return revocation.level
    logger.warning("Theme revocation %s has an unknown level %r: read as disable", revocation.id, revocation.level)
    return REVOKE_DISABLE


def _matching(db: Any, digest: str, theme_id: str, version: str) -> List[ThemeRevocation]:
    rows = (
        db.query(ThemeRevocation)
        .filter(ThemeRevocation.lifted_at.is_(None))
        .filter(or_(ThemeRevocation.digest == digest, ThemeRevocation.theme_id == theme_id))
        .all()
    )
    return [row for row in rows if _applies(row, digest, version)]


def _strongest(rows: List[ThemeRevocation]) -> Optional[ThemeRevocation]:
    return max(rows, key=lambda row: _STRENGTH[effective_level(row)]) if rows else None


def revocation_level(db: Any, digest: str, theme_id: str, version: str) -> Optional[str]:
    """``'disable'``, ``'warn'`` or None for one package version (the strongest live revocation wins).

    The stored rows only: a signature check needs the package row (``revocation_of``).
    """
    row = _strongest(_matching(db, digest, theme_id, version))
    return None if row is None else effective_level(row)


def signature_disable(package: Any) -> ThemeRevocation:
    """The transient ``disable`` of a version whose stored signature doesn't check out (C-05c).

    A new object on every call and never added to a session: nothing ever stores it.
    """
    return ThemeRevocation(digest=package.digest, level=REVOKE_DISABLE, reason=SIGNATURE_DISABLE_REASON)


def revocation_of(db: Any, package: Any) -> Optional[ThemeRevocation]:
    """The strongest live revocation of a ``ThemePackage`` row, or None.

    A stored ``disable`` is answered as it is; otherwise a stored signature that doesn't check
    out answers the transient ``signature_disable``, and anything else the strongest stored row.
    """
    stored = _strongest(_matching(db, package.digest, package.theme_id, package.version))
    if disables(stored):
        return stored
    return signature_disable(package) if signature_fails(db, package) else stored


def disables(revocation: Optional[ThemeRevocation]) -> bool:
    """True for a revocation whose effective level is ``disable``."""
    return revocation is not None and effective_level(revocation) == REVOKE_DISABLE


def is_disabled(db: Any, package: Any) -> bool:
    """True when a live ``disable`` revocation covers ``package``."""
    return disables(revocation_of(db, package))


def revocation_body(revocation: Optional[ThemeRevocation]) -> Optional[Dict[str, str]]:
    """``{level, reason}`` of a live revocation (its effective level), or None."""
    return None if revocation is None else {"level": effective_level(revocation), "reason": revocation.reason}


__all__ = [
    "REVOKE_DISABLE", "REVOKE_WARN", "disables", "effective_level", "is_disabled", "revocation_body",
    "revocation_level", "revocation_of", "signature_disable",
]
