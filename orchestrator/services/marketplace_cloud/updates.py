"""PRD-270 US-270-04b (§2, §5.3, K16) — after a Sync, unpinned installs move to the newest Cloud version.

``apply_cloud_updates`` runs after every successful Sync (``api/marketplace_cloud_sync.py``). For each theme
that has a Cloud row and an active listing install, and no local-origin Marketplace row (the operator's own
listing, whatever its status: a Sync never writes such a theme, and its installs never move from here), it
takes the newest compatible listed version (``listings.newest_compatible`` over ``listed_versions``). When
that version is a Cloud row and an install runs an older Marketplace version:

* if one of those installs is unpinned, ``package.ensure_stored`` fetches, checks and stores the version's
  files first, with no transaction held while Automatos Cloud answers: the plan's reads are ended before it
  and every row is read again after it. A refused or failed download is logged with the theme and version and
  changes no install, not even an offer; the theme counts as failed and the next Sync tries again;
* then the two calls ``review.approve`` makes once a version is approved: ``offer_listed_update`` (every
  older install is offered it, a pinned one too) and, only when the files were stored first,
  ``auto_update.apply_listed_update`` (every unpinned install it offered moves through ``updates.apply_update``,
  each in its own savepoint, with that path's refusals, which it logs and neither count shows).
  ``review.approve`` is never called: it refuses a row that isn't pending and records a signature request,
  which a signed Cloud row must not get. Nothing here writes an event row;
* each theme is committed on its own; a database error, or anything unexpected, rolls back that theme alone,
  is logged and counts it failed.

State-based, not event-based: whatever the Sync brought, an unpinned install on an older version is moved
to the newest one on the next pass, so a download that failed is retried by the next Sync. Pinning is how a
workspace holds an older version: a pinned install only keeps the offer, and nothing is fetched for a theme
whose only older installs are pinned (one unpinned since the plan was read moves on the next Sync, once its
files are stored). One pass at a time per process: a second caller waits for the first (at most
``package.wait_seconds``, then it moves nothing and says so in the log; the next Sync applies what is left).
Synchronous and blocking (plain-``def`` routes only); it commits.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, NamedTuple, Optional

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core.models.themes import THEME_SOURCE_CLOUD, ThemeInstallation, ThemePackage
from services.marketplace_cloud.fetch import ClientFactory
from services.marketplace_cloud.package import CloudPackageError, ensure_stored, is_cloud, wait_seconds
from services.themes.auto_update import apply_listed_update
from services.themes.choices import MARKETPLACE_OWNER
from services.themes.listing_updates import offer_listed_update
from services.themes.listings import listed_versions, newest_compatible
from services.themes.records import version_key

logger = logging.getLogger(__name__)

_UPDATE_LOCK = threading.Lock()


@dataclass(frozen=True)
class CloudUpdates:
    """One pass: ``applied`` installs moved to a newer Cloud version; ``failed`` themes whose newest Cloud version
    wasn't applied because its download or the database failed (their installs stay as they were until the next
    Sync)."""

    applied: int = 0
    failed: int = 0

    def plus(self, other: "CloudUpdates") -> "CloudUpdates":
        """These counts and ``other``'s together."""
        return CloudUpdates(applied=self.applied + other.applied, failed=self.failed + other.failed)


class Plan(NamedTuple):
    """One theme's update: the Cloud version to move to and whether an unpinned install runs an older one."""

    target: ThemePackage
    moves: bool


def updatable_themes(db: Session) -> List[str]:
    """The theme ids with a Cloud row and an active listing install, and no local-origin Marketplace row; sorted."""
    has_cloud = select(ThemePackage.theme_id).where(ThemePackage.source == THEME_SOURCE_CLOUD)
    has_local = select(ThemePackage.theme_id).where(
        ThemePackage.owner_type == MARKETPLACE_OWNER, ThemePackage.source.is_(None),
    )
    rows = (
        db.query(ThemeInstallation.theme_id)
        .join(ThemePackage, ThemePackage.id == ThemeInstallation.package_id)
        .filter(
            ThemeInstallation.uninstalled_at.is_(None),
            ThemePackage.owner_type == MARKETPLACE_OWNER,
            ThemeInstallation.theme_id.in_(has_cloud),
            ThemeInstallation.theme_id.not_in(has_local),
        )
        .distinct()
        .order_by(ThemeInstallation.theme_id)
        .all()
    )
    return [row.theme_id for row in rows]


def _older_pins(db: Session, target: ThemePackage) -> List[Optional[str]]:
    """The pins (None: unpinned) of the theme's active installs that run a Marketplace version older than
    ``target`` (columns only: the offer and the moves read the installs themselves)."""
    rows = (
        db.query(ThemeInstallation.pinned_version, ThemePackage.version)
        .join(ThemePackage, ThemePackage.id == ThemeInstallation.package_id)
        .filter(
            ThemeInstallation.theme_id == target.theme_id,
            ThemeInstallation.uninstalled_at.is_(None),
            ThemePackage.owner_type == MARKETPLACE_OWNER,
        )
        .all()
    )
    floor = version_key(target.version)
    return [pin for pin, version in rows if version_key(version) < floor]


def plan_for(db: Session, theme_id: str) -> Optional[Plan]:
    """The theme's update when its newest compatible listed version is a Cloud row an install runs below, else None."""
    target = newest_compatible(listed_versions(db, theme_id))
    if target is None or not is_cloud(target):
        return None
    pins = _older_pins(db, target)
    if not pins:
        return None
    return Plan(target=target, moves=any(pin is None for pin in pins))


def _fetched(db: Session, target: ThemePackage, client_factory: Optional[ClientFactory]) -> bool:
    """``ensure_stored`` for ``target`` with no transaction held over the download; False, logged with the theme
    and version, when it was refused. ``target`` keeps its loaded columns, detached from the session, whose reads
    so far are rolled back, so every row the offer and the moves read comes from the database again."""
    db.expunge(target)
    db.rollback()
    try:
        ensure_stored(db, target, client_factory)
    except CloudPackageError as refused:
        logger.warning(
            "Automatos Cloud: %s %s wasn't stored (%s, %s), so no install moves to it until the next Sync",
            target.theme_id, target.version, refused.status_code, refused.code,
        )
        db.rollback()  # ends the signature check's reads; nothing was written for this theme
        return False
    return True


def _apply(db: Session, plan: Plan, client_factory: Optional[ClientFactory]) -> CloudUpdates:
    """Store the version's files when an install will move, then offer it and, only then, move the unpinned installs."""
    target = plan.target
    theme_id, version, target_id = target.theme_id, target.version, target.id
    if plan.moves:
        if not _fetched(db, target, client_factory):
            return CloudUpdates(failed=1)
        target = db.get(ThemePackage, target_id)
    offered = offer_listed_update(db, target)
    applied = apply_listed_update(db, target) if plan.moves else 0
    db.commit()
    if offered or applied:
        logger.info("Automatos Cloud: %s %s offered to %d install(s), %d moved", theme_id, version, offered, applied)
    return CloudUpdates(applied=applied)


def _update_theme(db: Session, theme_id: str, client_factory: Optional[ClientFactory]) -> CloudUpdates:
    """One theme's update, committed on its own; any failure rolls this theme back alone and counts it failed."""
    try:
        plan = plan_for(db, theme_id)
        return CloudUpdates() if plan is None else _apply(db, plan, client_factory)
    except Exception:  # the database, or anything unexpected: the Sync stands and the other themes still update
        db.rollback()
        logger.exception("Automatos Cloud: the update of %s failed; its installs stay as they were", theme_id)
        return CloudUpdates(failed=1)


def _each_theme(db: Session, client_factory: Optional[ClientFactory]) -> CloudUpdates:
    """Every updatable theme in turn; a pass that can't read them counts one failure and moves nothing."""
    try:
        themes = updatable_themes(db)
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Automatos Cloud: the themes to update couldn't be read; no install moved")
        return CloudUpdates(failed=1)
    result = CloudUpdates()
    for theme_id in themes:
        result = result.plus(_update_theme(db, theme_id, client_factory))
    return result


def apply_cloud_updates(db: Session, client_factory: Optional[ClientFactory] = None) -> CloudUpdates:
    """After a successful Sync: move every unpinned install on an older version to the newest Cloud version,
    its files stored first; how many installs moved and how many themes failed."""
    if not _UPDATE_LOCK.acquire(timeout=wait_seconds()):
        logger.warning("Automatos Cloud: another update pass is still running, so this one moved nothing; "
                       "the next Sync applies what it leaves")
        return CloudUpdates()
    try:
        return _each_theme(db, client_factory)
    finally:
        _UPDATE_LOCK.release()


__all__ = ["CloudUpdates", "Plan", "apply_cloud_updates", "plan_for", "updatable_themes"]
