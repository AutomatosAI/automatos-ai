"""PRD-261 §5 — a workspace's private theme uploads in the database: who may upload, the package
row, the install row and the "update available" pointer.

Tenant isolation: every query is filtered by the caller's workspace. A private package
(``owner_type`` 'workspace') belongs to the one workspace that uploaded it; two workspaces that
upload identical bytes each get their own row and their own files.

Who may upload (A-07a with A-07e's workspace setting, D1-b):

* the platform switch ``THEMES_PRIVATE_UPLOAD`` (``on|off|admins|auto``, default ``auto``) is
  read from ``config``. ``off`` is a ceiling; ``admins`` lets only a
  platform admin (``caller_is_admin``) upload; ``on`` and ``auto`` defer to the workspace;
* the workspace setting (``theme_workspace_settings.private_uploads``) is ``on``, ``off`` or NULL.
  NULL means the default: ``on``, except under ``auto`` in the SaaS edition, where it is ``on``
  only in a workspace of one person (Q6): the distinct people in its active member rows plus
  its owner (``workspaces.owner_id``, who may have no member row). ``auto`` sets the default, not a ceiling: a workspace
  admin may switch it on.

An upload of a version this workspace already holds with other files is refused, and so is one
not newer than the installed version. A newer version is stored as its own package and set as the
install's ``pending_package_id`` ("update available"); the install never switches by itself.

A signed upload (C-05d): the route stores only a signature an active Automatos key verified over
the upload's own files (``services/themes/signed_upload.py``), so a workspace row with a
``signature`` is one Automatos vouched for. It may carry a reserved publisher's id without a
platform admin, and the same bytes uploaded again with a signature gain it.
``SIGNED_UPLOAD_COUNTS_AS_REVIEWED`` holds QB-24's answer for ``choices.is_private``.

Synchronous (a plain SQLAlchemy session): called from plain-``def`` routes only.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple
from uuid import UUID

from sqlalchemy import and_
from sqlalchemy.orm import Session

from config import config
from core.auth.roles import caller_is_admin
from core.models.themes import ThemeInstallation, ThemePackage, ThemeWorkspaceSettings
from core.models.workspaces import Workspace
from core.workspaces.models import WorkspaceMember
from services.themes.contract import RESERVED_PUBLISHERS, echo
from services.themes.manifest import VERSION_PATTERN
from services.themes.package import ThemePackage as CheckedThemePackage

logger = logging.getLogger(__name__)

PRIVATE_UPLOAD_SETTING = "THEMES_PRIVATE_UPLOAD"
SWITCH_ON, SWITCH_OFF, SWITCH_ADMINS, SWITCH_AUTO = "on", "off", "admins", "auto"
PRIVATE_UPLOAD_SWITCHES: Tuple[str, ...] = (SWITCH_ON, SWITCH_OFF, SWITCH_ADMINS, SWITCH_AUTO)
PRIVATE_UPLOAD_DEFAULT = SWITCH_AUTO
SETTING_ON, SETTING_OFF = "on", "off"
SINGLE_PERSON = 1
OWNER_WORKSPACE = "workspace"
PUBLISHER_SEPARATOR = "/"
# QB-24 (D10; open on 21 Oct, so the plan's recommended answer, yes): a private upload whose
# signature an active Automatos key verified counts as reviewed: no per-version opt-in, it may be
# the workspace default, and it reads "Signed by Automatos", not "Private · not reviewed".
SIGNED_UPLOAD_COUNTS_AS_REVIEWED = True

PLATFORM_OFF = "Private theme uploads are turned off on this platform."
ADMINS_ONLY = "On this platform only an Automatos admin can upload a private theme."
WORKSPACE_OFF = (
    "Private theme uploads are off for this workspace. "
    "A workspace admin can turn them on in Settings, Appearance, Themes."
)
DEFAULT_OFF = (
    "Private theme uploads are off by default in a workspace with more than one member. "
    "A workspace admin can turn them on in Settings, Appearance, Themes."
)


class UploadConflict(ValueError):
    """The upload clashes with what this workspace already holds; the message says how to fix it."""


class SavedUpload(NamedTuple):
    """The stored package, the workspace's install of its theme, and what the upload did."""

    package: ThemePackage
    installation: Optional[ThemeInstallation]
    created: bool  # False: these exact bytes were already in this workspace


def _workspace(workspace_id: Any) -> UUID:
    return UUID(str(workspace_id))


# ---------------------------------------------------------------------------
# Who may upload
# ---------------------------------------------------------------------------


def private_upload_switch() -> str:
    """The platform switch; a value it doesn't know is read as ``off`` (fail closed) and logged."""
    raw = config.THEMES_PRIVATE_UPLOAD
    value = str(raw or PRIVATE_UPLOAD_DEFAULT).strip().lower()
    if value not in PRIVATE_UPLOAD_SWITCHES:
        logger.warning("[Themes] %s=%r is not one of %s: private uploads are off", PRIVATE_UPLOAD_SETTING, raw, PRIVATE_UPLOAD_SWITCHES)
        return SWITCH_OFF
    return value


def people_count(db: Session, workspace_id: Any) -> int:
    """How many distinct people the workspace has: its active members plus its owner.

    ``workspaces.owner_id`` names an owner who may have no member row, so counting member rows
    alone would read an owner with one member as a single-person workspace.
    """
    ws = _workspace(workspace_id)
    member_rows = (
        db.query(WorkspaceMember.user_id)
        .filter(WorkspaceMember.workspace_id == ws, WorkspaceMember.is_active.is_(True))
        .distinct()
        .all()
    )
    people = {row.user_id for row in member_rows}
    owner_id = db.query(Workspace.owner_id).filter(Workspace.id == ws).scalar()
    if owner_id is not None:
        people.add(owner_id)
    return len(people)


def default_private_uploads(db: Session, workspace_id: Any, switch: str) -> str:
    """The workspace setting's value while it is unset (NULL), under ``switch``."""
    if switch == SWITCH_OFF:
        return SETTING_OFF
    if switch != SWITCH_AUTO or config.IS_LOCAL_EDITION:
        return SETTING_ON
    return SETTING_ON if people_count(db, workspace_id) <= SINGLE_PERSON else SETTING_OFF


def stored_private_uploads(db: Session, workspace_id: Any) -> Optional[str]:
    """The workspace's own choice (``on``/``off``), or None while it uses the default."""
    row = db.get(ThemeWorkspaceSettings, _workspace(workspace_id))
    return row.private_uploads if row is not None else None


def private_upload_refusal(db: Session, ctx: Any) -> Optional[str]:
    """Why this caller may not upload a private theme in its workspace, or None when it may."""
    switch = private_upload_switch()
    if switch == SWITCH_OFF:
        return PLATFORM_OFF
    if switch == SWITCH_ADMINS and not caller_is_admin(getattr(ctx, "user", None)):
        return ADMINS_ONLY
    stored = stored_private_uploads(db, ctx.workspace_id)
    if stored == SETTING_OFF:
        return WORKSPACE_OFF
    if stored is None and default_private_uploads(db, ctx.workspace_id, switch) == SETTING_OFF:
        return DEFAULT_OFF
    return None


def publisher_of(theme_id: str) -> str:
    """The ``publisher`` of a ``publisher/slug`` theme id."""
    return theme_id.split(PUBLISHER_SEPARATOR, 1)[0]


def reserved_publisher_refusal(theme_id: str, user: Any, signed: bool = False) -> Optional[str]:
    """Why ``user`` may not upload under this id's publisher (a reserved one), or None.

    ``signed``: an active Automatos key verified the upload's signature, which vouches for this id.
    """
    publisher = publisher_of(theme_id)
    if publisher in RESERVED_PUBLISHERS and not signed and not caller_is_admin(user):
        return f'Only an Automatos admin can upload a theme under the "{echo(publisher)}" publisher.'
    return None


# ---------------------------------------------------------------------------
# Reads, always within one workspace
# ---------------------------------------------------------------------------


def _own_private(ws: UUID):
    return and_(ThemePackage.owner_type == OWNER_WORKSPACE, ThemePackage.workspace_id == ws)


def workspace_package(db: Session, workspace_id: Any, package_id: UUID) -> Optional[ThemePackage]:
    """This workspace's own private package, or None (another workspace's is simply not found)."""
    return db.query(ThemePackage).filter(ThemePackage.id == package_id, _own_private(_workspace(workspace_id))).first()


def package_by_digest(db: Session, workspace_id: Any, digest: str) -> Optional[ThemePackage]:
    """This workspace's private package with these exact bytes, or None."""
    return db.query(ThemePackage).filter(_own_private(_workspace(workspace_id)), ThemePackage.digest == digest).first()


def active_installation(db: Session, workspace_id: Any, theme_id: str, lock: bool = False) -> Optional[ThemeInstallation]:
    """The workspace's active install of ``theme_id`` (at most one, by the partial unique index)."""
    query = db.query(ThemeInstallation).filter(
        ThemeInstallation.workspace_id == _workspace(workspace_id),
        ThemeInstallation.theme_id == theme_id,
        ThemeInstallation.uninstalled_at.is_(None),
    )
    return (query.with_for_update() if lock else query).first()


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def version_key(version: str) -> Tuple[int, int, int]:
    """``MAJOR.MINOR.PATCH`` as numbers, for ordering versions."""
    match = VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ValueError(f"Not a theme version: {version!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _check_new_version(db: Session, ws: UUID, manifest: Dict[str, Any], install: Optional[ThemeInstallation]) -> None:
    """Refuse a version this workspace holds with other files, or one not newer than the installed one."""
    theme_id, version = manifest["id"], manifest["version"]
    held = db.query(ThemePackage.id).filter(
        _own_private(ws), ThemePackage.theme_id == theme_id, ThemePackage.version == version,
    ).first()
    if held is not None:
        raise UploadConflict(
            f"This workspace already has {echo(theme_id)} {version} with different files. "
            "Raise the version in theme.json and upload it again."
        )
    current = db.get(ThemePackage, install.package_id) if install is not None else None
    if current is not None and version_key(version) <= version_key(current.version):
        raise UploadConflict(
            f"This workspace has {echo(theme_id)} {current.version} installed. "
            "Upload a higher version to offer it as an update."
        )


def upload_conflict(db: Session, workspace_id: Any, manifest: Dict[str, Any]) -> Optional[str]:
    """The version clash :func:`save_private_upload` would refuse (``UploadConflict``'s message), or None. Read-only.

    Run before the review stage, so a refused version never pays for it; and after a concurrent
    upload won a unique index, where it sees the winner's rows, so a same-version-other-files
    race is reported as that, not as "try again".
    """
    ws = _workspace(workspace_id)
    try:
        _check_new_version(db, ws, manifest, active_installation(db, ws, manifest["id"]))
    except UploadConflict as exc:
        return str(exc)
    return None


def _new_package(ws: UUID, user_pk: Optional[int], checked: CheckedThemePackage, prefix: str,
                 scan_report: Dict[str, Any], signature: Optional[str]) -> ThemePackage:
    manifest = checked.manifest
    return ThemePackage(
        theme_id=manifest["id"], version=manifest["version"], owner_type=OWNER_WORKSPACE, workspace_id=ws,
        digest=checked.digest, manifest=manifest, file_manifest=checked.file_manifest, storage_prefix=prefix,
        reviewed=False, scan_report=scan_report, scan_verdict=scan_report["verdict"], submitted_by=user_pk,
        signature=signature,
    )


def _install(db: Session, ws: UUID, user_pk: Optional[int], row: ThemePackage) -> ThemeInstallation:
    install = ThemeInstallation(workspace_id=ws, theme_id=row.theme_id, package_id=row.id, installed_by=user_pk)
    db.add(install)
    return install


def _offer_update(db: Session, install: ThemeInstallation, row: ThemePackage) -> None:
    """Point the install's "update available" at ``row`` unless a newer upload already waits."""
    pending = db.get(ThemePackage, install.pending_package_id) if install.pending_package_id else None
    if pending is None or version_key(row.version) > version_key(pending.version):
        install.pending_package_id = row.id


def _commit_after_store(db: Session, store: Callable[[], None]) -> None:
    """Write the files once the rows are flushed, then commit; a failed write commits nothing."""
    try:
        store()
    except Exception:
        logger.exception("[Themes] Storing an upload's files failed: rolling its rows back")
        db.rollback()
        raise
    db.commit()


def _gains_signature(existing: ThemePackage, signature: Optional[str]) -> bool:
    """Give the row of the same bytes the upload's verified signature; True when that changed it."""
    if signature is None or existing.signature == signature:
        return False
    existing.signature = signature
    return True


def _reuse_existing(
    db: Session, ws: UUID, user_pk: Optional[int], existing: ThemePackage, signature: Optional[str],
) -> SavedUpload:
    """The same bytes again: the existing package, installed again when nothing installs its theme,
    and signed when this upload carried a verified signature (C-05d)."""
    install = active_installation(db, ws, existing.theme_id, lock=True)
    changed = _gains_signature(existing, signature)
    if install is None:
        install = _install(db, ws, user_pk, existing)
        changed = True
    if changed:
        db.commit()
        db.refresh(install)
    return SavedUpload(existing, install, created=False)


def save_private_upload(
    db: Session, workspace_id: Any, user_pk: Optional[int], checked: CheckedThemePackage, prefix: str,
    scan_report: Dict[str, Any], store: Callable[[], None], signature: Optional[str] = None,
) -> SavedUpload:
    """Store a checked upload as this workspace's package and install it, or offer it as an update.

    ``store`` writes the files: it runs once the rows are flushed (so a concurrent duplicate is
    already refused by the unique indexes) and before the commit, so a refused upload writes no
    files and a failed write leaves no row. Raises :class:`UploadConflict` for a clash.
    ``signature``: one an active Automatos key verified over these files (C-05d), stored on the row.
    """
    ws = _workspace(workspace_id)
    existing = package_by_digest(db, ws, checked.digest)
    if existing is not None:
        return _reuse_existing(db, ws, user_pk, existing, signature)
    install = active_installation(db, ws, checked.manifest["id"], lock=True)
    _check_new_version(db, ws, checked.manifest, install)
    row = _new_package(ws, user_pk, checked, prefix, scan_report, signature)
    db.add(row)
    db.flush()
    if install is None:
        install = _install(db, ws, user_pk, row)
    else:
        _offer_update(db, install, row)
    db.flush()
    _commit_after_store(db, store)
    db.refresh(row)
    db.refresh(install)
    return SavedUpload(row, install, created=True)
