"""PRD-261 A-07e — a workspace's theme settings (PRD §5 admin controls, D1-b).

One ``theme_workspace_settings`` row per workspace, created on the first change; no row
means every setting is at its default:

* **private code uploads** (``private_uploads``): ``on``, ``off`` or NULL. NULL is the
  edition default, which A-07a's ``services/themes/records.py`` computes from the platform
  switch ``THEMES_PRIVATE_UPLOAD`` (``on|off|admins|auto``): ``auto`` is on in the local
  edition and, in SaaS, on only in a single-member workspace (Q6). ``auto`` sets the default,
  not a ceiling, so an admin may switch it on; the platform ``off`` is a ceiling, and turning
  the setting on under it is refused (409). Under ``admins`` only a platform admin uploads.
* **members may choose** (``members_may_choose``, default on): off, ``PUT /api/themes/choice``
  answers 403 and ``active`` shows the workspace default (A-07b enforces both).
* **who may install** (``install_role``): a workspace permission the installer must hold,
  ``workspace:manage`` (owners and admins) by default (Q5). It only narrows: PRD §6 keeps
  private code at ``workspace:manage`` at most, so the one other choice is "the owner only",
  stored as ``workspace:delete``, the permission only the owner role holds
  (``modules/policy/roles.py``). A stored value this module doesn't know is read as the
  owner only (fail closed) and logged. Changing it needs both the old and the new
  permission, so an admin can neither make installs owner-only nor undo the owner's choice.
  Install routes check it with :func:`install_refusal` or :func:`require_install_role`.
* **a pinned version per install** (``theme_installations.pinned_version``): the install
  refuses an update or rollback to any other version (A-07b's ``updates._check_switch``).
  A pin names the install's current, pending or previous version; setting or clearing one
  needs the install permission above.

The workspace default (``is_default``) is set by A-07b's ``PUT /api/themes/default``, which
answers 409 for an unreviewed theme (Q6); the settings answer only reports it.

Tenancy: every read and write is scoped to the caller's workspace. Installs and packages are
looked up through A-07b's ``services/themes/choices.py`` helpers, so another workspace's
install, or an install pointing at another workspace's private package, answers 404.

Services raise :class:`ThemeInstallError` with a status and a plain message; nothing here
commits: the router does, once. Synchronous: called from plain-``def`` routes only.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from fastapi import Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import workspace_permission_granted
from core.database.database import get_db
from core.models.themes import THEME_INSTALL_ROLE_DEFAULT, ThemeInstallation, ThemePackage, ThemeWorkspaceSettings
from services.themes.choices import (
    BAD_REQUEST,
    CONFLICT,
    FORBIDDEN,
    ThemeInstallError,
    active_install,
    active_installs,
    package_is_visible,
)
from services.themes.manifest import VERSION_PATTERN
from services.themes.records import (
    PLATFORM_OFF,
    SETTING_OFF,
    SETTING_ON,
    SWITCH_ADMINS,
    SWITCH_OFF,
    default_private_uploads,
    private_upload_switch,
)

logger = logging.getLogger(__name__)

INSTALL_ROLE_ADMINS = "admins"
INSTALL_ROLE_OWNER = "owner"
OWNER_ONLY_PERMISSION = "workspace:delete"
# The label the API speaks -> the permission ``install_role`` stores and the gate checks.
INSTALL_ROLE_PERMISSIONS: Dict[str, str] = {
    INSTALL_ROLE_ADMINS: THEME_INSTALL_ROLE_DEFAULT,
    INSTALL_ROLE_OWNER: OWNER_ONLY_PERMISSION,
}
INSTALL_ROLE_CHOICES: Tuple[str, ...] = tuple(INSTALL_ROLE_PERMISSIONS)
_LABEL_OF_PERMISSION = {permission: label for label, permission in INSTALL_ROLE_PERMISSIONS.items()}

PRIVATE_UPLOAD_CHOICES: Tuple[str, ...] = (SETTING_ON, SETTING_OFF)
EFFECTIVE_ADMINS = "admins"
SOURCE_WORKSPACE, SOURCE_DEFAULT = "workspace", "default"
MEMBERS_MAY_CHOOSE_DEFAULT = True
MAX_PINS = 50

# The keys ``apply_settings`` reads; a key that is present is a change (None = back to the default).
CHANGE_PRIVATE_UPLOADS = "private_uploads"
CHANGE_MEMBERS_MAY_CHOOSE = "members_may_choose"
CHANGE_INSTALL_ROLE = "install_role"
CHANGE_PINS = "pins"
CHANGE_KEYS: Tuple[str, ...] = (CHANGE_PRIVATE_UPLOADS, CHANGE_MEMBERS_MAY_CHOOSE, CHANGE_INSTALL_ROLE, CHANGE_PINS)

WHO_INSTALLS = {
    INSTALL_ROLE_ADMINS: "a workspace owner or admin",
    INSTALL_ROLE_OWNER: "the workspace owner",
}
NOTHING_TO_CHANGE = "Send at least one theme setting to change."
UNKNOWN_SETTING = "A theme setting this server doesn't know was sent."
BAD_PRIVATE_UPLOADS = "privateUploads is 'on', 'off' or null (the default)."
BAD_MEMBERS_MAY_CHOOSE = "membersMayChoose is true, false or null (the default)."
BAD_INSTALL_ROLE = "installRole is 'admins', 'owner' or null (the default)."
TOO_MANY_PINS = f"Send at most {MAX_PINS} pins at once."
DUPLICATE_PIN = "Each install can be pinned once per request."
BAD_PIN_VERSION = "A pinned version looks like 1.2.3."
ROLE_CHANGE_REFUSED = "Only the workspace owner can change who may install themes here."
NO_WORKSPACE = "Choose a workspace first."


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def settings_row(db: Session, workspace_id: UUID) -> Optional[ThemeWorkspaceSettings]:
    """The workspace's settings row, or None while every setting is at its default."""
    return db.get(ThemeWorkspaceSettings, workspace_id)


def install_role_label(row: Optional[ThemeWorkspaceSettings]) -> str:
    """Who may install, as the API names it; an unknown stored value reads as the owner only."""
    stored = THEME_INSTALL_ROLE_DEFAULT if row is None or not row.install_role else row.install_role
    label = _LABEL_OF_PERMISSION.get(stored)
    if label is None:
        logger.warning("[Themes] workspace %s has an unknown install_role %r: owner only", row.workspace_id, stored)
        return INSTALL_ROLE_OWNER
    return label


def install_permission(db: Session, workspace_id: UUID) -> str:
    """The workspace permission a caller needs to install, update or pin a theme here."""
    return INSTALL_ROLE_PERMISSIONS[install_role_label(settings_row(db, workspace_id))]


def install_refusal(db: Session, ctx: Any) -> Optional[str]:
    """Why this caller may not install a theme in its workspace (the setting), or None."""
    workspace_id = getattr(ctx, "workspace_id", None)
    if workspace_id is None:
        return NO_WORKSPACE
    label = install_role_label(settings_row(db, _as_uuid(workspace_id)))
    if workspace_permission_granted(db, ctx, INSTALL_ROLE_PERMISSIONS[label]):
        return None
    return f"In this workspace only {WHO_INSTALLS[label]} can install or update themes."


def require_install_role():
    """Depends factory for install routes: 403 unless the caller satisfies "who may install".

    Layer it after the route's ``require_workspace_permission("workspace:manage")`` gate (the
    authZ sweep reads that one); this adds only the workspace's narrower choice.
    """

    def _dep(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)) -> RequestContext:
        refusal = install_refusal(db, ctx)
        if refusal is not None:
            raise HTTPException(status_code=FORBIDDEN, detail=refusal)
        return ctx

    return _dep


def private_uploads_state(db: Session, workspace_id: UUID, row: Optional[ThemeWorkspaceSettings]) -> Dict[str, Any]:
    """The setting, its edition default, the platform switch and what applies."""
    switch = private_upload_switch()
    stored = None if row is None else row.private_uploads
    default = default_private_uploads(db, workspace_id, switch)
    value = stored if stored in PRIVATE_UPLOAD_CHOICES else default
    if switch == SWITCH_OFF or value == SETTING_OFF:
        effective = SETTING_OFF
    else:
        effective = EFFECTIVE_ADMINS if switch == SWITCH_ADMINS else SETTING_ON
    return {
        "setting": stored,
        "source": SOURCE_DEFAULT if stored is None else SOURCE_WORKSPACE,
        "default": default,
        "platform": switch,
        "effective": effective,
    }


def _package_version(db: Session, workspace_id: UUID, package_id: Optional[UUID]) -> Optional[str]:
    if package_id is None:
        return None
    package = db.get(ThemePackage, package_id)
    return package.version if package_is_visible(package, workspace_id) else None


def pinnable_versions(db: Session, workspace_id: UUID, install: ThemeInstallation) -> List[str]:
    """The versions an install may be pinned to: its current, pending and previous ones."""
    ids = (install.package_id, install.pending_package_id, install.previous_package_id)
    versions = [_package_version(db, workspace_id, package_id) for package_id in ids]
    return list(dict.fromkeys(version for version in versions if version is not None))


def _pin_row(db: Session, workspace_id: UUID, install: ThemeInstallation) -> Dict[str, Any]:
    versions = pinnable_versions(db, workspace_id, install)
    return {
        "installId": str(install.id),
        "themeId": install.theme_id,
        "version": versions[0] if versions else None,
        "pinnedVersion": install.pinned_version,
        "pinnableVersions": versions,
    }


def read_settings(db: Session, workspace_id: UUID, ctx: Any) -> Dict[str, Any]:
    """``GET /api/themes/settings``: every setting with its default, plus the caller's install right."""
    row = settings_row(db, workspace_id)
    installs = active_installs(db, workspace_id)
    default = next((install for install in installs if install.is_default), None)
    return {
        "privateUploads": private_uploads_state(db, workspace_id, row),
        "membersMayChoose": MEMBERS_MAY_CHOOSE_DEFAULT if row is None else bool(row.members_may_choose),
        "installRole": install_role_label(row),
        "installRoles": list(INSTALL_ROLE_CHOICES),
        "canInstall": install_refusal(db, ctx) is None,
        "defaultInstallId": None if default is None else str(default.id),
        "pins": [_pin_row(db, workspace_id, install) for install in installs],
        "updatedAt": None if row is None or row.updated_at is None else row.updated_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def _as_uuid(value: Any) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def _check_change_shape(changes: Mapping[str, Any]) -> None:
    if not changes:
        raise ThemeInstallError(BAD_REQUEST, NOTHING_TO_CHANGE)
    if any(key not in CHANGE_KEYS for key in changes):
        raise ThemeInstallError(BAD_REQUEST, UNKNOWN_SETTING)
    if changes.get(CHANGE_PRIVATE_UPLOADS) not in (None,) + PRIVATE_UPLOAD_CHOICES:
        raise ThemeInstallError(BAD_REQUEST, BAD_PRIVATE_UPLOADS)
    if not isinstance(changes.get(CHANGE_MEMBERS_MAY_CHOOSE), (bool, type(None))):
        raise ThemeInstallError(BAD_REQUEST, BAD_MEMBERS_MAY_CHOOSE)
    if changes.get(CHANGE_INSTALL_ROLE) not in (None,) + INSTALL_ROLE_CHOICES:
        raise ThemeInstallError(BAD_REQUEST, BAD_INSTALL_ROLE)
    _check_pins_shape(changes.get(CHANGE_PINS) or ())


def _check_pins_shape(pins: Sequence[Tuple[UUID, Optional[str]]]) -> None:
    if len(pins) > MAX_PINS:
        raise ThemeInstallError(BAD_REQUEST, TOO_MANY_PINS)
    install_ids = [install_id for install_id, _ in pins]
    if len(set(install_ids)) != len(install_ids):
        raise ThemeInstallError(BAD_REQUEST, DUPLICATE_PIN)
    for _, version in pins:
        if version is not None and (not isinstance(version, str) or VERSION_PATTERN.fullmatch(version) is None):
            raise ThemeInstallError(BAD_REQUEST, BAD_PIN_VERSION)


def _check_rights(db: Session, ctx: Any, workspace_id: UUID, changes: Mapping[str, Any]) -> None:
    """Refusals that need no write: the role change, the pin right and the platform ceiling."""
    current = install_role_label(settings_row(db, workspace_id))
    if CHANGE_INSTALL_ROLE in changes:
        wanted = changes[CHANGE_INSTALL_ROLE] or INSTALL_ROLE_ADMINS
        needed = {INSTALL_ROLE_PERMISSIONS[current], INSTALL_ROLE_PERMISSIONS[wanted]}
        if not all(workspace_permission_granted(db, ctx, permission) for permission in needed):
            raise ThemeInstallError(FORBIDDEN, ROLE_CHANGE_REFUSED)
    if changes.get(CHANGE_PINS):
        refusal = install_refusal(db, ctx)
        if refusal is not None:
            raise ThemeInstallError(FORBIDDEN, refusal)
    if changes.get(CHANGE_PRIVATE_UPLOADS) == SETTING_ON and private_upload_switch() == SWITCH_OFF:
        raise ThemeInstallError(CONFLICT, PLATFORM_OFF)


def _writable_row(db: Session, workspace_id: UUID, user_pk: Optional[int]) -> ThemeWorkspaceSettings:
    """The settings row, created on the first change (a concurrent first change updates it)."""
    row = settings_row(db, workspace_id)
    if row is not None:
        return row
    db.add(ThemeWorkspaceSettings(workspace_id=workspace_id, updated_by=user_pk))
    try:
        db.flush()
    except IntegrityError:
        # Another request created the row between our read and our insert. Nothing has
        # been written in this transaction yet, so rolling back loses nothing.
        db.rollback()
        logger.info("[Themes] settings for workspace %s were created concurrently; updating them", workspace_id)
    row = settings_row(db, workspace_id)
    if row is None:
        raise ThemeInstallError(CONFLICT, "The theme settings changed while saving. Try again.")
    return row


def _apply_pin(db: Session, workspace_id: UUID, install_id: UUID, version: Optional[str]) -> None:
    install = active_install(db, workspace_id, install_id)
    if version is not None and version not in pinnable_versions(db, workspace_id, install):
        raise ThemeInstallError(CONFLICT, (
            f"Version {version} isn't one this install holds. Pin its current, update or rollback version."
        ))
    install.pinned_version = version


def _apply_fields(row: ThemeWorkspaceSettings, changes: Mapping[str, Any]) -> None:
    if CHANGE_PRIVATE_UPLOADS in changes:
        row.private_uploads = changes[CHANGE_PRIVATE_UPLOADS]
    if CHANGE_MEMBERS_MAY_CHOOSE in changes:
        chosen = changes[CHANGE_MEMBERS_MAY_CHOOSE]
        row.members_may_choose = MEMBERS_MAY_CHOOSE_DEFAULT if chosen is None else chosen
    if CHANGE_INSTALL_ROLE in changes:
        row.install_role = INSTALL_ROLE_PERMISSIONS[changes[CHANGE_INSTALL_ROLE] or INSTALL_ROLE_ADMINS]


def apply_settings(
    db: Session, ctx: Any, workspace_id: UUID, user_pk: Optional[int], changes: Mapping[str, Any],
) -> None:
    """``PUT /api/themes/settings``: apply every change or none (the router commits once).

    ``changes`` holds only the keys being changed (:data:`CHANGE_KEYS`); None puts a
    setting back to its default. ``pins`` is a sequence of ``(install_id, version | None)``.
    """
    _check_change_shape(changes)
    _check_rights(db, ctx, workspace_id, changes)
    row = _writable_row(db, workspace_id, user_pk)
    _apply_fields(row, changes)
    for install_id, version in changes.get(CHANGE_PINS) or ():
        _apply_pin(db, workspace_id, install_id, version)
    row.updated_by = user_pk
    db.flush()


__all__ = [
    "INSTALL_ROLE_ADMINS",
    "INSTALL_ROLE_OWNER",
    "INSTALL_ROLE_CHOICES",
    "INSTALL_ROLE_PERMISSIONS",
    "MAX_PINS",
    "apply_settings",
    "install_permission",
    "install_refusal",
    "pinnable_versions",
    "private_uploads_state",
    "read_settings",
    "require_install_role",
    "settings_row",
]
