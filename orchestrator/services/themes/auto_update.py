"""Installed Marketplace themes follow their listing (publish contract D1, 30 Oct 2026).

* **Approval** (``review.approve``): once ``listing_updates.offer_listed_update`` has offered the
  new version, every listing install it offered that is not pinned moves to it through
  ``updates.apply_update``, the path a person's **Apply update** takes, so the same refusals hold
  (tenancy, a newer approved version, the pin, the default's review rule, a ``disable``). An
  install it refuses keeps the offer, exactly as before this module.
* **Deactivation** (``review.deactivate``): every listing install that runs the version taken
  down, and is not pinned, moves back through ``updates.move_back`` to the newest listed version
  below it that this host runs. With none, it stays where it is and the ``disable`` revocation
  sends its members to its base on the next load, as before.

A pinned install never moves: pinning is how a workspace holds a version (it keeps the offer). An
install keeps its id (Automatos Go keys the theme it paints on it), its looks and its members'
choices. No route, role or step-up check runs here, and nothing assumes a Clerk user: the platform
acts on an approval or a deactivation that an admin (or the publish key) already made, the same in
both editions. ``apply_update`` writes no event row of its own; the caller's ``approved`` or
``deactivated`` event records how many installs moved.

Each move runs in its own savepoint. A refusal, or a database error, is logged with the install,
workspace and package ids and the status code, leaves that install as it was, and never stops the
others or the approval. Nothing here commits.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core.models.themes import ThemeInstallation, ThemePackage
from services.themes import updates
from services.themes.choices import MARKETPLACE_OWNER, ThemeInstallError
from services.themes.listing_updates import is_listed
from services.themes.listings import is_compatible, listed_versions
from services.themes.records import version_key

logger = logging.getLogger(__name__)

UPDATE, ROLL_BACK = "update", "roll back"

Move = Callable[[Session, UUID, UUID, UUID], ThemeInstallation]


def _unpinned_listing_installs(db: Session, theme_id: str, *criteria) -> List[ThemeInstallation]:
    """The theme's active, unpinned installs that run a Marketplace version and match ``criteria``, oldest first."""
    return (
        db.query(ThemeInstallation)
        .join(ThemePackage, ThemePackage.id == ThemeInstallation.package_id)
        .filter(
            ThemeInstallation.theme_id == theme_id,
            ThemeInstallation.uninstalled_at.is_(None),
            ThemeInstallation.pinned_version.is_(None),
            ThemePackage.owner_type == MARKETPLACE_OWNER,
            *criteria,
        )
        .order_by(ThemeInstallation.installed_at, ThemeInstallation.id)
        .all()
    )


def _moved(db: Session, install: ThemeInstallation, target: ThemePackage, action: str, move: Move) -> bool:
    """Move one install to ``target`` in its own savepoint; False, logged and untouched, when refused or failed."""
    install_id, workspace_id = install.id, install.workspace_id
    try:
        with db.begin_nested():
            move(db, workspace_id, install_id, target.id)
    except ThemeInstallError as refusal:
        logger.warning(
            "[Themes] %s of install %s (workspace %s) to %s %s (package %s) refused with %s: %s",
            action, install_id, workspace_id, target.theme_id, target.version, target.id,
            refusal.status_code, refusal.message,
        )
        return False
    except SQLAlchemyError:
        logger.exception(
            "[Themes] %s of install %s (workspace %s) to package %s failed in the database; it stays as it was",
            action, install_id, workspace_id, target.id,
        )
        return False
    return True


def apply_listed_update(db: Session, package: ThemePackage) -> int:
    """Move every unpinned listing install offered ``package`` (just approved) to it; how many moved.

    Call it after ``offer_listed_update``: the installs it moves are the ones that offer pointed at
    ``package``. An install that already waits on a newer listed version was not offered it and stays.
    """
    if not is_listed(db, package):
        return 0
    offered = _unpinned_listing_installs(db, package.theme_id, ThemeInstallation.pending_package_id == package.id)
    return sum(_moved(db, install, package, UPDATE, updates.apply_update) for install in offered)


def previous_listed(db: Session, package: ThemePackage) -> Optional[ThemePackage]:
    """The newest listed version of ``package``'s theme below it that this host runs, or None.

    Listed means approved, reviewed and under no ``disable`` (``listings.listed_versions``).
    """
    floor = version_key(package.version)
    below = (row for row in listed_versions(db, package.theme_id)
             if row.id != package.id and version_key(row.version) < floor)
    return next((row for row in below if is_compatible(row)), None)


def roll_back_installs(db: Session, package: ThemePackage) -> int:
    """Move every unpinned listing install that runs ``package`` (just taken off the Marketplace) back to
    the previous listed version; how many moved. With no previous version, none moves (the ``disable``
    sends their members to the theme's base on the next load)."""
    running = _unpinned_listing_installs(db, package.theme_id, ThemeInstallation.package_id == package.id)
    if not running:
        return 0
    target = previous_listed(db, package)
    if target is None:
        logger.info(
            "[Themes] %s %s was taken down with no listed version below it: %d install(s) stay on it, disabled",
            package.theme_id, package.version, len(running),
        )
        return 0
    return sum(_moved(db, install, target, ROLL_BACK, updates.move_back) for install in running)


__all__ = ["apply_listed_update", "previous_listed", "roll_back_installs"]
