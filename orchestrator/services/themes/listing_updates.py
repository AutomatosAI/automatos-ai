"""PRD-261 C-04a — offer a newly approved Marketplace version to the installs that run an older one.

``GET /api/themes/installed`` (A-07b) shows "update available" from an install's
``pending_package_id``. A private upload sets it at upload time (``records._offer_update``) and a
listing install sets it at install time (``listings.install_listing``); a version a person
approves **later** reaches existing installs only through ``offer_listed_update``.

Wired at C1-wire: C-03a's ``review.approve`` calls ``offer_listed_update(db, package)`` once it has
set ``status='approved'`` and ``reviewed=True``, in the same transaction. Only installs that run a
Marketplace version of the theme are offered it (a workspace's own upload is not a listing
install), and an install that already waits on a version at least as new keeps that one. The offer
is not the last step: ``review.approve`` then moves every offered install that is not pinned to the
new version (``auto_update.apply_listed_update``, publish contract D1). A pinned install, and one
the move refuses, keeps the offer here.

A version taken off the Marketplace (``review.deactivate``) withdraws its offers through
``withdraw_offers``: each install it was offered to is offered the newest listed version newer
than the one it runs instead, or nothing. A pending version that is no longer listed never counts
as a newer offer still waiting, so a later fix numbered below it is still offered. The installs
that RUN the version taken off are moved back by ``auto_update.roll_back_installs``. Nothing here
commits.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from core.models.themes import ThemeInstallation, ThemePackage
from services.themes.choices import MARKETPLACE_OWNER
from services.themes.listings import APPROVED, is_compatible, listed_versions, newer_compatible
from services.themes.records import version_key
from services.themes.revocations import is_disabled


def is_listed(db: Session, package: ThemePackage) -> bool:
    """True for an approved, reviewed Marketplace version this host runs and no ``disable`` covers."""
    return (
        package.owner_type == MARKETPLACE_OWNER
        and package.status == APPROVED
        and bool(package.reviewed)
        and is_compatible(package)
        and not is_disabled(db, package)
    )


def _waits_on_newer(db: Session, install: ThemeInstallation, package: ThemePackage) -> bool:
    """True when the install's pending version is at least as new as ``package`` and still on offer:
    a workspace's own upload, or a Marketplace version that is still listed."""
    pending: Optional[ThemePackage] = (
        db.get(ThemePackage, install.pending_package_id) if install.pending_package_id else None
    )
    if pending is None or (pending.owner_type == MARKETPLACE_OWNER and not is_listed(db, pending)):
        return False
    return version_key(pending.version) >= version_key(package.version)


def offer_listed_update(db: Session, package: ThemePackage) -> int:
    """Point every active listing install of an older version at ``package``; how many it offered."""
    if not is_listed(db, package):
        return 0
    rows = (
        db.query(ThemeInstallation, ThemePackage)
        .join(ThemePackage, ThemePackage.id == ThemeInstallation.package_id)
        .filter(
            ThemeInstallation.theme_id == package.theme_id,
            ThemeInstallation.uninstalled_at.is_(None),
            ThemePackage.owner_type == MARKETPLACE_OWNER,
        )
        .all()
    )
    offered = 0
    for install, current in rows:
        if version_key(current.version) >= version_key(package.version) or _waits_on_newer(db, install, package):
            continue
        install.pending_package_id = package.id
        offered += 1
    db.flush()
    return offered


def withdraw_offers(db: Session, package: ThemePackage) -> int:
    """Re-point every install offered ``package`` (just taken off the Marketplace) at the newest listed
    version newer than the one it runs, or clear its offer; how many installs it changed."""
    installs = db.query(ThemeInstallation).filter(ThemeInstallation.pending_package_id == package.id).all()
    versions = [row for row in listed_versions(db, package.theme_id) if row.id != package.id]
    for install in installs:
        current = db.get(ThemePackage, install.package_id)
        newer = newer_compatible(versions, current) if current is not None else None
        install.pending_package_id = newer.id if newer is not None else None
    db.flush()
    return len(installs)
