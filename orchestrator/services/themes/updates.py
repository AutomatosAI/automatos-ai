"""PRD-261 A-07b — an install's versions: list, compare, update, roll back, uninstall.

* ``update {packageId}`` moves the install to a newer package of the same theme (normally
  its ``pending_package_id``, set when a higher version is uploaded or listed), keeps the
  old one as ``previous_package_id`` and keeps every member's looks (they are keyed by
  theme, not package). A private package is a new opt-in by construction: opt-ins are
  keyed by package. A pinned install (A-07e) refuses any other version.
* ``rollback`` swaps the current and previous packages.
* ``uninstall`` is soft (``uninstalled_at``), keeps looks, and drops the members' choices
  of it so they see the workspace default.
* ``compare`` lists the changed files from the two ``file_manifest``s and the scan diff.

Every package an install moves to passes the tenancy check (a Marketplace package or this
workspace's own; 404 otherwise), and a workspace default never moves to unreviewed code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.themes import ThemeInstallation, ThemePackage
from services.themes.choices import (
    CONFLICT,
    MARKETPLACE_OWNER,
    NOT_FOUND,
    PACKAGE_NOT_FOUND,
    ThemeInstallError,
    active_install,
    active_installs,
    forget_choices_of,
    is_private,
    package_is_visible,
    visible_package,
)
from services.themes.defaults import PRIVATE_DEFAULT, WITHDRAWN, can_be_default
from services.themes.drift import needs_update
from services.themes.review_stage import member_view
from services.themes.optin import opt_in_state
from services.themes.records import version_key
from services.themes.revocations import disables, revocation_body, revocation_of
from services.themes.signed_upload import signature_view

APPROVED = "approved"
CLOSED_STATUSES = ("rejected", "deactivated")

OTHER_THEME = "That package is a different theme."
SAME_VERSION = "The install already runs that version."
NOT_NEWER = "That version is not newer than the one installed. Use rollback to go back to the previous version."
NOT_APPROVED = "That Marketplace version isn't approved yet."
CLOSED = "That version was rejected or withdrawn."
NO_PREVIOUS = "There is no earlier version to go back to."


def _iso(moment: Optional[datetime]) -> Optional[str]:
    return None if moment is None else moment.isoformat()


def _version_of(db: Session, workspace_id: UUID, package_id: Optional[UUID]) -> Optional[str]:
    package = None if package_id is None else db.get(ThemePackage, package_id)
    return package.version if package_is_visible(package, workspace_id) else None


def install_row(db: Session, workspace_id: UUID, user_pk: Optional[int], install: ThemeInstallation) -> Dict[str, Any]:
    """One row of ``GET /api/themes/installed``; ``signature`` is the package's ``{state, keyId}`` (C-05d)."""
    package = db.get(ThemePackage, install.package_id)
    manifest = package.manifest or {}
    return {
        "installId": str(install.id),
        "themeId": install.theme_id,
        "name": manifest.get("name"),
        "packageId": str(package.id),
        "version": package.version,
        "ownerType": package.owner_type,
        "reviewed": bool(package.reviewed),
        "private": is_private(package),
        "isDefault": bool(install.is_default),
        "pinnedVersion": install.pinned_version,
        "previousPackageId": None if install.previous_package_id is None else str(install.previous_package_id),
        "pendingPackageId": None if install.pending_package_id is None else str(install.pending_package_id),
        "pendingVersion": _version_of(db, workspace_id, install.pending_package_id),
        "installedAt": _iso(install.installed_at),
        "optIn": opt_in_state(db, workspace_id, user_pk, package),
        "revoked": revocation_body(revocation_of(db, package)),
        "needsUpdate": needs_update(manifest),
        "signature": signature_view(package),
    }


def list_installed(db: Session, workspace_id: UUID, user_pk: Optional[int]) -> List[Dict[str, Any]]:
    """Every active install of the workspace, as ``install_row`` gives it."""
    return [install_row(db, workspace_id, user_pk, install) for install in active_installs(db, workspace_id)]


def _unlisted_marketplace(package: ThemePackage) -> bool:
    """A Marketplace version that is not approved (pending, rejected or deactivated)."""
    return package.owner_type == MARKETPLACE_OWNER and package.status != APPROVED


def _refuse_unusable(package: ThemePackage) -> None:
    if _unlisted_marketplace(package):
        raise ThemeInstallError(CONFLICT, NOT_APPROVED)
    if package.status in CLOSED_STATUSES:
        raise ThemeInstallError(CONFLICT, CLOSED)


def _check_switch(db: Session, install: ThemeInstallation, target: ThemePackage) -> None:
    """Refusals shared by update and rollback: pin, default and withdrawn code."""
    if install.pinned_version is not None and target.version != install.pinned_version:
        raise ThemeInstallError(CONFLICT, (
            f"This install is pinned to version {install.pinned_version}. Unpin it in the theme settings first."
        ))
    if install.is_default and not can_be_default(target):
        raise ThemeInstallError(CONFLICT, PRIVATE_DEFAULT)
    if disables(revocation_of(db, target)):
        raise ThemeInstallError(CONFLICT, WITHDRAWN)


def _target(db: Session, workspace_id: UUID, install: ThemeInstallation, package_id: UUID) -> ThemePackage:
    target = visible_package(db, workspace_id, package_id)
    if target.theme_id != install.theme_id:
        raise ThemeInstallError(CONFLICT, OTHER_THEME)
    if target.id == install.package_id:
        raise ThemeInstallError(CONFLICT, SAME_VERSION)
    return target


def _changed_files(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, List[str]]:
    def sha(manifest: Dict[str, Any], path: str) -> Any:
        entry = manifest.get(path)
        return entry.get("sha256") if isinstance(entry, dict) else entry

    shared = set(before) & set(after)
    return {
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "changed": sorted(path for path in shared if sha(before, path) != sha(after, path)),
    }


def _scan_diff(before: ThemePackage, after: ThemePackage) -> Dict[str, Any]:
    """The verdicts and what changed between the two reports, as a workspace may see them
    (``member_view``: never the LLM check's score or findings)."""
    old = member_view(before.scan_report) or {}
    new = member_view(after.scan_report) or {}
    changed = {
        key: {"from": old.get(key), "to": new.get(key)}
        for key in sorted(set(old) | set(new)) if old.get(key) != new.get(key)
    }
    return {"verdict": {"from": before.scan_verdict, "to": after.scan_verdict}, "changed": changed}


def compare(db: Session, workspace_id: UUID, install_id: UUID, package_id: UUID) -> Dict[str, Any]:
    """What moving the install to ``package_id`` changes: versions, files and the scan."""
    install = active_install(db, workspace_id, install_id)
    current = visible_package(db, workspace_id, install.package_id)
    target = _target(db, workspace_id, install, package_id)
    if _unlisted_marketplace(target):
        # A pending or rejected Marketplace version is not one this workspace may see: 404, as missing.
        raise ThemeInstallError(NOT_FOUND, PACKAGE_NOT_FOUND)
    return {
        "installId": str(install.id),
        "from": {"packageId": str(current.id), "version": current.version, "reviewed": bool(current.reviewed)},
        "to": {"packageId": str(target.id), "version": target.version, "reviewed": bool(target.reviewed)},
        "files": _changed_files(current.file_manifest or {}, target.file_manifest or {}),
        "scan": _scan_diff(current, target),
    }


def apply_update(db: Session, workspace_id: UUID, install_id: UUID, package_id: UUID) -> ThemeInstallation:
    """Move the install to ``package_id``; the old package becomes the rollback target."""
    install = active_install(db, workspace_id, install_id)
    target = _target(db, workspace_id, install, package_id)
    current = visible_package(db, workspace_id, install.package_id)
    if version_key(target.version) <= version_key(current.version):
        raise ThemeInstallError(CONFLICT, NOT_NEWER)
    _refuse_unusable(target)
    _check_switch(db, install, target)
    install.previous_package_id = install.package_id
    install.package_id = target.id
    if install.pending_package_id == target.id:
        install.pending_package_id = None
    db.flush()
    return install


def rollback(db: Session, workspace_id: UUID, install_id: UUID) -> ThemeInstallation:
    """Swap the install's current and previous packages."""
    install = active_install(db, workspace_id, install_id)
    if install.previous_package_id is None:
        raise ThemeInstallError(CONFLICT, NO_PREVIOUS)
    previous = visible_package(db, workspace_id, install.previous_package_id)
    _refuse_unusable(previous)
    _check_switch(db, install, previous)
    install.package_id, install.previous_package_id = previous.id, install.package_id
    db.flush()
    return install


def uninstall(db: Session, workspace_id: UUID, install_id: UUID) -> ThemeInstallation:
    """Soft-remove the install; looks stay, and its members fall back to the default."""
    install = active_install(db, workspace_id, install_id)
    install.uninstalled_at = datetime.now(timezone.utc)
    install.is_default = False
    forget_choices_of(db, workspace_id, install.id)
    db.flush()
    return install
