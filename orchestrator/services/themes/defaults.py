"""PRD-261 A-07b — the workspace default, and the theme a member sees (``GET /api/themes/active``).

Resolution order (plan §2.6 A-07b): the member's own choice (when the workspace lets members
choose, A-07e), then the workspace default, then Studio. A choice naming an install that is
gone (uninstalled, or another workspace's) falls through to the default.

Q6: a private (unreviewed) theme is never a workspace default, because a default runs for
members who never said yes to its code. ``PUT /api/themes/default`` refuses one with 409,
and resolution skips a default row that somehow points at one.

The answer's shape is the one the host reads (``frontend/lib/theme-host/manifest.ts``
``parseActiveInstall``: ``theme.installId``, ``theme.digest``, ``theme.manifest``)::

    {"choice": "classic" | "studio" | "builtin:play" | "<installId>",
     "theme": null | {installId, themeId, packageId, version, digest, manifest, reviewed,
                      optIn: {required, wording, packageId}, revoked: null | {level, reason},
                      needsUpdate, integrity: {url: "sha256-<base64>"}}}

A ``disable`` revocation answers the theme's base (``studio`` or ``classic``) with no theme,
plus a top-level ``revoked`` naming the install, so the host can say why.
"""

from __future__ import annotations

import base64
import logging
import re
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.themes import ThemeInstallation, ThemePackage, ThemeRevocation
from services.themes.choices import (
    BUILTIN_CHOICES,
    CONFLICT,
    FALLBACK_CHOICE,
    PLAIN_CHOICES,
    REVOCATION_DISABLE,
    ThemeInstallError,
    active_install,
    find_active_install,
    install_id_of,
    is_private,
    members_may_choose,
    package_is_visible,
    revocation_of,
    stored_choice,
)
from services.themes.contract import THEME_BASES
from services.themes.drift import needs_update
from services.themes.optin import opt_in_state

logger = logging.getLogger(__name__)

# The browser fetches a package's files under this path (frontend/lib/theme-host/constants.ts THEME_FILES_PATH).
THEME_FILES_PATH = "/api/themes/files/"
MODULE_SUFFIX = ".js"
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
INTEGRITY_PREFIX = "sha256-"

PRIVATE_DEFAULT = "An unreviewed theme can't be the workspace default: members would run its code without saying yes."
WITHDRAWN = "That theme version was withdrawn, so it can't be used."


def can_be_default(package: ThemePackage) -> bool:
    """Only reviewed Marketplace code may run for members who never opted in (Q6)."""
    return not is_private(package)


def integrity_map(package: ThemePackage) -> Dict[str, str]:
    """``{module URL: 'sha256-<base64>'}`` for every ``.js`` file, from the stored hex (PA-13)."""
    pinned: Dict[str, str] = {}
    for path, entry in sorted((package.file_manifest or {}).items()):
        sha = entry.get("sha256") if isinstance(entry, dict) else None
        if not path.endswith(MODULE_SUFFIX):
            continue
        if not isinstance(sha, str) or SHA256_HEX.fullmatch(sha) is None:
            logger.error("theme package %s has no valid sha256 for %s", package.id, path)
            continue
        url = f"{THEME_FILES_PATH}{package.digest}/{path}"
        pinned[url] = INTEGRITY_PREFIX + base64.b64encode(bytes.fromhex(sha)).decode("ascii")
    return pinned


def revocation_body(revocation: Optional[ThemeRevocation]) -> Optional[Dict[str, str]]:
    """``{level, reason}`` of a standing revocation, or None."""
    return None if revocation is None else {"level": revocation.level, "reason": revocation.reason}


def _base_of(package: ThemePackage) -> str:
    base = (package.manifest or {}).get("base")
    return base if base in THEME_BASES else FALLBACK_CHOICE


def theme_payload(
    db: Session, workspace_id: UUID, user_pk: Optional[int], install: ThemeInstallation, package: ThemePackage,
) -> Dict[str, Any]:
    """The ``theme`` object of the active answer for one install and its current package."""
    manifest = package.manifest or {}
    return {
        "installId": str(install.id),
        "themeId": install.theme_id,
        "packageId": str(package.id),
        "version": package.version,
        "digest": package.digest,
        "manifest": manifest,
        "reviewed": bool(package.reviewed),
        "optIn": opt_in_state(db, workspace_id, user_pk, package),
        "revoked": revocation_body(revocation_of(db, package)),
        "needsUpdate": needs_update(manifest),
        "integrity": integrity_map(package),
    }


def default_install(db: Session, workspace_id: UUID) -> Optional[ThemeInstallation]:
    """The workspace's active default install, when its package may be a default."""
    row = (
        db.query(ThemeInstallation)
        .filter(
            ThemeInstallation.workspace_id == workspace_id,
            ThemeInstallation.uninstalled_at.is_(None),
            ThemeInstallation.is_default.is_(True),
        )
        .one_or_none()
    )
    if row is None:
        return None
    package = db.get(ThemePackage, row.package_id)
    if not package_is_visible(package, workspace_id) or not can_be_default(package):
        logger.warning("theme install %s is marked default but its package may not be one", row.id)
        return None
    return row


def _pick(db: Session, workspace_id: UUID, choice: Optional[str]) -> Tuple[str, Optional[ThemeInstallation]]:
    if choice in PLAIN_CHOICES or choice in BUILTIN_CHOICES:
        return choice, None
    install_id = install_id_of(choice)
    install = None if install_id is None else find_active_install(db, workspace_id, install_id)
    if install is not None:
        return choice, install
    fallback = default_install(db, workspace_id)
    return (str(fallback.id), fallback) if fallback is not None else (FALLBACK_CHOICE, None)


def resolve_active(db: Session, workspace_id: UUID, user_pk: Optional[int]) -> Dict[str, Any]:
    """What this member sees: their choice, else the workspace default, else Studio."""
    choice = stored_choice(db, workspace_id, user_pk) if members_may_choose(db, workspace_id) else None
    picked, install = _pick(db, workspace_id, choice)
    if install is None:
        return {"choice": picked, "theme": None}
    package = db.get(ThemePackage, install.package_id)
    revocation = revocation_of(db, package)
    if revocation is not None and revocation.level == REVOCATION_DISABLE:
        revoked = {"installId": str(install.id), **revocation_body(revocation)}
        return {"choice": _base_of(package), "theme": None, "revoked": revoked}
    return {"choice": picked, "theme": theme_payload(db, workspace_id, user_pk, install, package)}


def _clear_default(db: Session, workspace_id: UUID) -> None:
    db.query(ThemeInstallation).filter(
        ThemeInstallation.workspace_id == workspace_id, ThemeInstallation.is_default.is_(True),
    ).update({ThemeInstallation.is_default: False}, synchronize_session="fetch")
    db.flush()


def set_default(db: Session, workspace_id: UUID, install_id: Optional[UUID]) -> Optional[ThemeInstallation]:
    """Make an install the workspace default (None clears it); 409 for unreviewed or withdrawn code."""
    target = None if install_id is None else active_install(db, workspace_id, install_id)
    if target is not None:
        package = db.get(ThemePackage, target.package_id)
        if not can_be_default(package):
            raise ThemeInstallError(CONFLICT, PRIVATE_DEFAULT)
        revocation = revocation_of(db, package)
        if revocation is not None and revocation.level == REVOCATION_DISABLE:
            raise ThemeInstallError(CONFLICT, WITHDRAWN)
    _clear_default(db, workspace_id)
    if target is not None:
        target.is_default = True
        db.flush()
    return target
