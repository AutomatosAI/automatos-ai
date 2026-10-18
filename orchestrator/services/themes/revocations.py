"""PRD-261 A-08a — is a theme package revoked? ``theme_revocations``, read on every serve.

A revocation names a package digest, or a theme id with an optional version range (an exact
``X.Y.Z`` or a caret ``^X.Y.Z``, the ranges ``engines`` takes, ``manifest.satisfies_range``).
Its level is ``warn`` (still runs; the host shows a notice) or ``disable`` (the files route
answers 404 and the boot route the base boot). A lifted revocation (``lifted_at`` set) no
longer counts. A version range this code can't read is taken to match every version and is
logged: a revocation fails closed.

``active`` (A-07b) reads the same answer through :func:`package_revocation`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import or_

from core.models.themes import ThemeRevocation
from services.themes.manifest import RANGE_PATTERN, satisfies_range

logger = logging.getLogger(__name__)

REVOKE_WARN = "warn"
REVOKE_DISABLE = "disable"


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


def revocation_level(db: Any, digest: str, theme_id: str, version: str) -> Optional[str]:
    """``'disable'``, ``'warn'`` or None for one package version (the strongest live revocation wins)."""
    rows = (
        db.query(ThemeRevocation)
        .filter(ThemeRevocation.lifted_at.is_(None))
        .filter(or_(ThemeRevocation.digest == digest, ThemeRevocation.theme_id == theme_id))
        .all()
    )
    levels = {row.level for row in rows if _applies(row, digest, version)}
    if REVOKE_DISABLE in levels:
        return REVOKE_DISABLE
    return REVOKE_WARN if REVOKE_WARN in levels else None


def package_revocation(db: Any, package: Any) -> Optional[str]:
    """:func:`revocation_level` for a ``ThemePackage`` row."""
    return revocation_level(db, package.digest, package.theme_id, package.version)


def is_disabled(db: Any, package: Any) -> bool:
    """True when a live ``disable`` revocation covers ``package``."""
    return package_revocation(db, package) == REVOKE_DISABLE


__all__ = ["REVOKE_DISABLE", "REVOKE_WARN", "is_disabled", "package_revocation", "revocation_level"]
