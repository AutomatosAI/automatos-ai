"""PRD-261 A-07b — a member's theme choice, and the lookups every install route shares.

PA-12: the server is the one source of truth for the choice. ``theme_choices`` holds one
row per member per workspace; the value is one of ``'classic'``, ``'studio'`` or the
lowercase uuid of an active install in that workspace. A row written before PRD-261 A-19,
when Play was compiled into the app, may still name it; it reads as no choice
(``defaults._pick``), and a new save of that value is refused like any other.

Tenancy (the W-B2 review carry): ``theme_installations.package_id`` and
``theme_opt_ins.package_id`` have no database constraint against another workspace's
private package, so every lookup here checks the package itself: a Marketplace package,
or one owned by the caller's workspace. Anything else answers 404 exactly as a missing
row does, so a caller can't learn that another workspace's package exists.

Services raise :class:`ThemeInstallError` with an HTTP status and a plain message; the
router turns it into the response. Nothing here commits: the router does, once.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, object_session
from sqlalchemy.orm.exc import UnmappedInstanceError

from core.models.themes import ThemeChoice, ThemeInstallation, ThemePackage, ThemeWorkspaceSettings
from services.themes.records import SIGNED_UPLOAD_COUNTS_AS_REVIEWED

logger = logging.getLogger(__name__)

PLAIN_CHOICES = ("classic", "studio")
FALLBACK_CHOICE = "studio"
MARKETPLACE_OWNER = "marketplace"
WORKSPACE_OWNER = "workspace"

NOT_FOUND = 404
CONFLICT = 409
FORBIDDEN = 403
BAD_REQUEST = 400

INSTALL_NOT_FOUND = "That theme isn't installed in this workspace."
PACKAGE_NOT_FOUND = "That theme version doesn't exist in this workspace."
CHOICE_LOCKED = "This workspace's admins choose its theme; the workspace default applies."
NEEDS_MEMBER = "Choosing a theme needs a signed-in member of this workspace."
CHOICE_RACED = "Your theme choice changed in another tab at the same moment. Try again."
BAD_CHOICE = "A theme choice is 'classic', 'studio' or an installed theme's id."


class ThemeInstallError(Exception):
    """A refusal the router answers with ``status_code`` and ``message`` (no internals)."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _session_of(package: ThemePackage) -> Optional[Session]:
    """The session the row was loaded through (None for a detached row or a stand-in object)."""
    try:
        return object_session(package)
    except UnmappedInstanceError:
        return None


def _vouched_for(package: ThemePackage) -> bool:
    """An active Automatos key nobody revoked verifies the stored signature (C-05c's check, QB-24's question).

    Imported here, not at the top: ``signature_state`` imports ``signature_verify``, which imports
    ``signing``, which imports this module.
    A row with no session can't read the keys the hosted feed revoked, so it vouches for nothing.
    """
    if not getattr(package, "signature", None):
        return False
    db = _session_of(package)
    if db is None:
        return False
    from services.themes.signature_state import signature_vouches

    return signature_vouches(db, package)


def is_private(package: ThemePackage) -> bool:
    """Code a member runs only after opting in to it (Q6).

    A Marketplace package is private until it is reviewed. A workspace's own upload is private
    unless Automatos vouched for it: QB-24 (``records.SIGNED_UPLOAD_COUNTS_AS_REVIEWED``, C-05d)
    counts a private upload as reviewed while its stored signature verifies with an active key
    that neither the keys file nor the hosted feed revoked (``signature_state.signature_vouches``,
    C-05c), never because a signature is merely stored.
    """
    if package.owner_type == WORKSPACE_OWNER:
        return not (SIGNED_UPLOAD_COUNTS_AS_REVIEWED and _vouched_for(package))
    return not package.reviewed


def package_is_visible(package: Optional[ThemePackage], workspace_id: UUID) -> bool:
    """A Marketplace package, or one this workspace owns."""
    if package is None:
        return False
    if package.owner_type == MARKETPLACE_OWNER:
        return True
    return package.owner_type == WORKSPACE_OWNER and package.workspace_id == workspace_id


def visible_package(db: Session, workspace_id: UUID, package_id: UUID) -> ThemePackage:
    """The package, when this workspace may use it; else 404."""
    package = db.get(ThemePackage, package_id)
    if not package_is_visible(package, workspace_id):
        raise ThemeInstallError(NOT_FOUND, PACKAGE_NOT_FOUND)
    return package


def find_active_install(db: Session, workspace_id: UUID, install_id: UUID) -> Optional[ThemeInstallation]:
    """An active install of this workspace whose current package it may use, or None."""
    install = db.get(ThemeInstallation, install_id)
    if install is None or install.workspace_id != workspace_id or install.uninstalled_at is not None:
        return None
    if not package_is_visible(db.get(ThemePackage, install.package_id), workspace_id):
        logger.warning("theme install %s points at a package its workspace may not use", install_id)
        return None
    return install


def active_install(db: Session, workspace_id: UUID, install_id: UUID) -> ThemeInstallation:
    """:func:`find_active_install`, or 404."""
    install = find_active_install(db, workspace_id, install_id)
    if install is None:
        raise ThemeInstallError(NOT_FOUND, INSTALL_NOT_FOUND)
    return install


def active_installs(db: Session, workspace_id: UUID) -> List[ThemeInstallation]:
    """This workspace's active installs whose packages it may use, oldest first."""
    rows = (
        db.query(ThemeInstallation)
        .filter(ThemeInstallation.workspace_id == workspace_id, ThemeInstallation.uninstalled_at.is_(None))
        .order_by(ThemeInstallation.installed_at, ThemeInstallation.theme_id)
        .all()
    )
    return [row for row in rows if package_is_visible(db.get(ThemePackage, row.package_id), workspace_id)]


def install_id_of(choice: Optional[str]) -> Optional[UUID]:
    """The install a choice names (a lowercase uuid), or None for any other value."""
    if not isinstance(choice, str):
        return None
    try:
        parsed = UUID(choice)
    except ValueError:
        return None
    return parsed if str(parsed) == choice else None


def members_may_choose(db: Session, workspace_id: UUID) -> bool:
    """The A-07e setting; a workspace with no settings row lets members choose."""
    settings = db.get(ThemeWorkspaceSettings, workspace_id)
    return True if settings is None else bool(settings.members_may_choose)


def stored_choice(db: Session, workspace_id: UUID, user_pk: Optional[int]) -> Optional[str]:
    """The member's stored choice in this workspace, or None."""
    if user_pk is None:
        return None
    row = _choice_row(db, workspace_id, user_pk)
    return None if row is None else row.choice


def _checked_choice(db: Session, workspace_id: UUID, choice: str) -> str:
    if choice in PLAIN_CHOICES:
        return choice
    install_id = install_id_of(choice)
    if install_id is None:
        raise ThemeInstallError(BAD_REQUEST, BAD_CHOICE)
    active_install(db, workspace_id, install_id)
    return choice


def set_choice(db: Session, workspace_id: UUID, user_pk: Optional[int], choice: str) -> str:
    """Store the member's choice: 403 when members may not choose, 404 for an unknown install."""
    if user_pk is None:
        raise ThemeInstallError(FORBIDDEN, NEEDS_MEMBER)
    if not members_may_choose(db, workspace_id):
        raise ThemeInstallError(FORBIDDEN, CHOICE_LOCKED)
    checked = _checked_choice(db, workspace_id, choice)
    row = _choice_row(db, workspace_id, user_pk)
    if row is None:
        if _inserted_choice(db, workspace_id, user_pk, checked):
            return checked
        # A concurrent save (two tabs, a double click) wrote the row between our read and
        # our insert: re-check the choice in the fresh transaction, then update that row.
        checked = _checked_choice(db, workspace_id, choice)
        row = _choice_row(db, workspace_id, user_pk)
        if row is None:
            # That concurrent row is gone again (an uninstall dropped it): one more insert, else 409.
            if _inserted_choice(db, workspace_id, user_pk, checked):
                return checked
            raise ThemeInstallError(CONFLICT, CHOICE_RACED)
    row.choice = checked
    db.flush()
    return checked


def _choice_row(db: Session, workspace_id: UUID, user_pk: int) -> Optional[ThemeChoice]:
    return db.query(ThemeChoice).filter_by(workspace_id=workspace_id, user_id=user_pk).one_or_none()


def _inserted_choice(db: Session, workspace_id: UUID, user_pk: int, checked: str) -> bool:
    """Insert the member's first choice; False when another request inserted it first."""
    db.add(ThemeChoice(workspace_id=workspace_id, user_id=user_pk, choice=checked))
    try:
        db.flush()
    except IntegrityError:
        # uq_theme_choices_workspace_user holds one row per member. This request has only
        # read so far, so rolling back loses nothing (the optin.py pattern).
        db.rollback()
        logger.info("theme choice for workspace %s was saved concurrently; updating it", workspace_id)
        return False
    return True


def forget_choices_of(db: Session, workspace_id: UUID, install_id: UUID) -> None:
    """Drop every member's choice of an install (uninstall moves them to the default)."""
    db.query(ThemeChoice).filter(
        ThemeChoice.workspace_id == workspace_id, ThemeChoice.choice == str(install_id),
    ).delete(synchronize_session=False)
