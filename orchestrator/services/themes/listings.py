"""PRD-261 C-04a — the Themes tab's listings: list, detail and install by reference.

A **listing** is one Marketplace theme (``owner_type`` 'marketplace'), shown through its
**approved** versions only: ``status`` 'approved' and ``reviewed`` true, as a person approved
them (C-03a, D3). A pending, rejected or deactivated version is never listed, and neither is a
version a live ``disable`` revocation covers; a ``warn`` revocation is listed with its reason.
A workspace's private upload is never a listing.

The listing id is the theme id with its one ``/`` written as ``.`` (``automatos/play`` is
``automatos.play``): a path segment can't carry the slash, and theme ids have no dots
(``manifest.ID_PATTERN``), so the two forms map one to one.

* **Versions map** (PRD §4.2, after Obsidian): every approved version with its ``engines``
  and whether this host meets them, so a local edition that lags behind SaaS installs the
  newest version it can run (``latest``).
* **Previews per preset** (PRD §7 item 5): a preview ``assets/previews/<preset>.png`` or
  ``assets/previews/<preset>-<screen>.png`` belongs to that preset (the longest preset id that
  matches wins); any other preview is in ``otherPreviews``. Previews keep theme.json's order, and
  the card's ``cover`` is the first one it lists. Phone screenshots (theme.json's ``phonePreviews``,
  ``assets/previews/phone/<preset>[-<screen>].png`` or ``.jpg``) follow the same rule into each
  preset's ``phonePreviews``, ``otherPhonePreviews`` and the card's ``phoneCover``; a package without
  them answers ``[]`` and ``null`` there. Every URL is the files route (``serve.module_url``), which
  serves an approved listing's files to any caller (A-08a).
* **Pricing** is a label only in v1 (Q12): the manifest's ``pricing``, else "Free".
* **Official** (QB-21): ``publisher.official`` is true only for a reserved publisher
  (``automatos``, or a ``theme_publishers`` row marked reserved) whose shown version carries a
  signature an active Automatos key verifies (``signature_verify``, C-05b). A reserved name alone
  is never Official: a local edition's operator may submit and approve ``automatos/x`` unsigned.
  The card, the detail and each entry of the versions map carry ``signature {state, keyId}``.
* **Install** writes a ``theme_installations`` row that **references** the approved package:
  every workspace that installs a version shares its one package row. Update, compare and
  rollback of a listing install are A-07b's routes; this module only creates the row. An
  install of an older version gets the newest compatible one as its ``pending_package_id``, so
  ``GET /api/themes/installed`` shows the update too; a version approved later is offered to
  existing installs by ``listing_updates.offer_listed_update``, and the unpinned ones move to it
  (``auto_update.apply_listed_update``).

Services raise ``ThemeInstallError`` (status and plain message); nothing here commits.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import config
from core.models.themes import ThemeInstallation, ThemePackage, ThemePublisher
from services.themes import signature_verify
from services.themes.choices import (
    CONFLICT,
    FORBIDDEN,
    MARKETPLACE_OWNER,
    NOT_FOUND,
    ThemeInstallError,
    package_is_visible,
)
from services.themes.contract import RESERVED_PUBLISHERS, SDK_VERSION, THEME_HOST_VERSION, echo
from services.themes.look_schema import preset_ids
from services.themes.manifest import ID_PATTERN, satisfies_range
from services.themes.records import PUBLISHER_SEPARATOR, active_installation, publisher_of, version_key
from services.themes.revocations import is_disabled, revocation_body, revocation_of
from services.themes.serve import module_url

logger = logging.getLogger(__name__)

APPROVED = "approved"
LISTING_SEPARATOR = "."
FREE_LABEL = "Free"
SCREEN_SEPARATOR = "-"
EXTENSION_SEPARATOR = "."
README_PATH, CHANGELOG_PATH = "README.md", "CHANGELOG.md"
MARKETPLACE_SETTING = "THEMES_MARKETPLACE"
SWITCH_ON, SWITCH_OFF = "on", "off"

MARKETPLACE_OFF = "The theme Marketplace is turned off on this platform."
LISTING_NOT_FOUND = "That theme isn't in the Marketplace."
VERSION_NOT_FOUND = "That version of the theme isn't in the Marketplace."
NO_COMPATIBLE = "No version of this theme runs on this Automatos yet."
CONCURRENT_INSTALL = "Someone installed a theme here at the same moment. Reload and try again."


# ---------------------------------------------------------------------------
# The switch, ids and versions
# ---------------------------------------------------------------------------


def marketplace_switch() -> str:
    """``THEMES_MARKETPLACE``; a value it doesn't know is read as ``off`` (fail closed) and logged."""
    raw = getattr(config, MARKETPLACE_SETTING, SWITCH_ON)
    value = str(raw or SWITCH_ON).strip().lower()
    if value not in (SWITCH_ON, SWITCH_OFF):
        logger.warning("[Themes] %s=%r is not on or off: the Marketplace is off", MARKETPLACE_SETTING, raw)
        return SWITCH_OFF
    return value


def require_marketplace_on() -> None:
    """403 with a plain sentence while the Marketplace switch is off."""
    if marketplace_switch() == SWITCH_OFF:
        raise ThemeInstallError(FORBIDDEN, MARKETPLACE_OFF)


def listing_id_of(theme_id: str) -> str:
    """``publisher/slug`` -> ``publisher.slug``."""
    return theme_id.replace(PUBLISHER_SEPARATOR, LISTING_SEPARATOR, 1)


def theme_id_of(listing_id: str) -> Optional[str]:
    """``publisher.slug`` -> ``publisher/slug``, or None for anything that isn't a listing id."""
    if not isinstance(listing_id, str) or listing_id.count(LISTING_SEPARATOR) != 1:
        return None
    theme_id = listing_id.replace(LISTING_SEPARATOR, PUBLISHER_SEPARATOR, 1)
    return theme_id if ID_PATTERN.fullmatch(theme_id) else None


def is_compatible(package: ThemePackage) -> bool:
    """True when this host meets the version's ``engines`` (checked again at load, PRD §4.2)."""
    engines = (package.manifest or {}).get("engines")
    if not isinstance(engines, dict):
        return False
    return satisfies_range(engines.get("automatos"), THEME_HOST_VERSION) and satisfies_range(
        engines.get("sdk"), SDK_VERSION,
    )


def _approved_query(db: Session):
    return db.query(ThemePackage).filter(
        ThemePackage.owner_type == MARKETPLACE_OWNER,
        ThemePackage.status == APPROVED,
        ThemePackage.reviewed.is_(True),
    )


def _newest_first(rows: Sequence[ThemePackage]) -> List[ThemePackage]:
    return sorted(rows, key=lambda row: version_key(row.version), reverse=True)


def listed_versions(db: Session, theme_id: str) -> List[ThemePackage]:
    """The theme's approved versions that no ``disable`` revocation covers, newest first."""
    rows = _approved_query(db).filter(ThemePackage.theme_id == theme_id).all()
    return _newest_first([row for row in rows if not is_disabled(db, row)])


def all_listings(db: Session) -> Dict[str, List[ThemePackage]]:
    """``{theme_id: listed versions, newest first}`` over the whole Marketplace, by theme id."""
    grouped: Dict[str, List[ThemePackage]] = {}
    for row in _approved_query(db).order_by(ThemePackage.theme_id).all():
        if not is_disabled(db, row):
            grouped.setdefault(row.theme_id, []).append(row)
    return {theme_id: _newest_first(rows) for theme_id, rows in grouped.items()}


def newest_compatible(versions: Sequence[ThemePackage]) -> Optional[ThemePackage]:
    """The newest version this host can run, or None."""
    return next((row for row in versions if is_compatible(row)), None)


def newer_compatible(versions: Sequence[ThemePackage], package: ThemePackage) -> Optional[ThemePackage]:
    """The newest compatible listed version when it is newer than ``package``, else None."""
    latest = newest_compatible(versions)
    if latest is None or version_key(latest.version) <= version_key(package.version):
        return None
    return latest


# ---------------------------------------------------------------------------
# What a listing shows
# ---------------------------------------------------------------------------


def _publisher(db: Session, theme_id: str, verdict: signature_verify.Verdict) -> Dict[str, Any]:
    """The id's publisher; ``official`` only when it is reserved and the shown version's signature verifies."""
    name = publisher_of(theme_id)
    row = db.query(ThemePublisher).filter(ThemePublisher.publisher == name).first()
    reserved = name in RESERVED_PUBLISHERS or bool(row is not None and row.reserved)
    return {"name": name, "official": reserved and verdict.verified}


def _file_url(package: ThemePackage, path: str) -> Optional[str]:
    listed = package.file_manifest if isinstance(package.file_manifest, dict) else {}
    return module_url(package.digest, path) if path in listed else None


class PreviewKind(NamedTuple):
    """One theme.json list of screenshots: its field (also each preset's key in the detail), its folder, and
    the detail's key for the ones no preset claims."""

    field: str
    folder: str
    other_key: str


WEB_PREVIEWS = PreviewKind("previews", "assets/previews/", "otherPreviews")
PHONE_PREVIEWS = PreviewKind("phonePreviews", "assets/previews/phone/", "otherPhonePreviews")
PREVIEW_KINDS: Tuple[PreviewKind, ...] = (WEB_PREVIEWS, PHONE_PREVIEWS)


def _preview_paths(package: ThemePackage, kind: PreviewKind) -> List[str]:
    """The screenshots theme.json lists under ``kind`` that the package holds, in theme.json's order.

    The first one is the cover (``_cover``). The theme picks its order (Play lists ``ember.png`` first,
    its default look's chat), so it is never re-sorted by name. ``manifest.validate_manifest`` refuses a
    list that names a file twice.
    """
    named = (package.manifest or {}).get(kind.field) or []
    return [path for path in named if isinstance(path, str) and _file_url(package, path) is not None]


def _cover(package: ThemePackage, kind: PreviewKind) -> Optional[str]:
    """The files URL of the first screenshot of ``kind`` the package holds, or None."""
    paths = _preview_paths(package, kind)
    return module_url(package.digest, paths[0]) if paths else None


def _stem(path: str, kind: PreviewKind) -> str:
    """A screenshot's file name without its folder and extension (theme.json's patterns allow one of each)."""
    return path[len(kind.folder):].rsplit(EXTENSION_SEPARATOR, 1)[0]


def _owner_preset(stem: str, presets: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """``(preset, screen)`` a preview's file stem belongs to; the longest matching preset id wins."""
    for preset in sorted(presets, key=len, reverse=True):
        if stem == preset:
            return preset, None
        if stem.startswith(preset + SCREEN_SEPARATOR):
            return preset, stem[len(preset) + len(SCREEN_SEPARATOR):]
    return None, None


def _claimed(
    package: ThemePackage, presets: Sequence[str], kind: PreviewKind,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """``({preset: [{screen, url}]}, [unclaimed url])`` for one list of screenshots, in theme.json's order."""
    owned = [
        (_owner_preset(_stem(path, kind), presets), module_url(package.digest, path))
        for path in _preview_paths(package, kind)
    ]
    claimed = {
        preset: [{"screen": screen, "url": url} for (owner, screen), url in owned if owner == preset]
        for preset in presets
    }
    return claimed, [url for (owner, _), url in owned if owner is None]


def previews_by_preset(package: ThemePackage) -> Dict[str, Any]:
    """Each preset with its web and phone screenshots (per screen), and the screenshots no preset claims."""
    listed = package.file_manifest if isinstance(package.file_manifest, dict) else {}
    presets = sorted(preset_ids(listed))
    default_look = (package.manifest or {}).get("look")
    lists = {kind: _claimed(package, presets, kind) for kind in PREVIEW_KINDS}
    return {
        "presets": [
            {"id": preset, "default": preset == default_look,
             **{kind.field: claimed[preset] for kind, (claimed, _) in lists.items()}}
            for preset in presets
        ],
        **{kind.other_key: other for kind, (_, other) in lists.items()},
    }


def _signature_verdict(package: ThemePackage) -> signature_verify.Verdict:
    """The version's signature verdict for a listing. A keys file that does not load fails closed: the
    signature reads ``invalid`` (never Official) and the listings still answer; the paste route keeps
    refusing loudly (C-05b)."""
    try:
        return signature_verify.package_verdict(package)
    except signature_verify.SigningKeysError as bad:
        logger.error("Theme signing keys did not load; %s is shown unverified: %s", package.theme_id, bad)
        return signature_verify.Verdict(signature_verify.STATE_INVALID)


def _version_entry(db: Session, package: ThemePackage) -> Dict[str, Any]:
    return {
        "packageId": str(package.id),
        "engines": (package.manifest or {}).get("engines"),
        "compatible": is_compatible(package),
        "revoked": revocation_body(revocation_of(db, package)),
        "signature": _signature_verdict(package).view(),
        "createdAt": None if package.created_at is None else package.created_at.isoformat(),
    }


def installed_state(
    db: Session, workspace_id: UUID, theme_id: str, latest: Optional[ThemePackage],
) -> Optional[Dict[str, Any]]:
    """This workspace's active install of the theme, and the newer listed version it could move to."""
    install = active_installation(db, workspace_id, theme_id)
    package = None if install is None else db.get(ThemePackage, install.package_id)
    if install is None or not package_is_visible(package, workspace_id):
        return None
    from_listing = package.owner_type == MARKETPLACE_OWNER
    newer = latest is not None and version_key(latest.version) > version_key(package.version)
    return {
        "installId": str(install.id),
        "packageId": str(package.id),
        "version": package.version,
        "fromListing": from_listing,
        "updatePackageId": str(latest.id) if from_listing and newer else None,
    }


def listing_summary(db: Session, workspace_id: UUID, versions: Sequence[ThemePackage]) -> Dict[str, Any]:
    """One card of the Themes tab: the newest version this host runs (else the newest one)."""
    latest = newest_compatible(versions)
    shown = latest or versions[0]
    manifest = shown.manifest or {}
    verdict = _signature_verdict(shown)
    return {
        "listingId": listing_id_of(shown.theme_id),
        "themeId": shown.theme_id,
        "name": manifest.get("name"),
        "description": manifest.get("description"),
        "publisher": _publisher(db, shown.theme_id, verdict),
        "signature": verdict.view(),
        "version": shown.version,
        "packageId": str(shown.id),
        "compatible": latest is not None,
        "uses": list(manifest.get("uses") or []),
        "pricing": manifest.get("pricing") or FREE_LABEL,
        "cover": _cover(shown, WEB_PREVIEWS),
        "phoneCover": _cover(shown, PHONE_PREVIEWS),
        "revoked": revocation_body(revocation_of(db, shown)),
        "installed": installed_state(db, workspace_id, shown.theme_id, latest),
    }


def list_listings(db: Session, workspace_id: UUID) -> List[Dict[str, Any]]:
    """Every listing, by theme id."""
    return [listing_summary(db, workspace_id, versions) for versions in all_listings(db).values()]


def _versions_or_404(db: Session, listing_id: str) -> List[ThemePackage]:
    theme_id = theme_id_of(listing_id)
    versions = [] if theme_id is None else listed_versions(db, theme_id)
    if not versions:
        raise ThemeInstallError(NOT_FOUND, LISTING_NOT_FOUND)
    return versions


def listing_detail(db: Session, workspace_id: UUID, listing_id: str) -> Dict[str, Any]:
    """The detail page: the card, previews per preset, docs, layout facts and the versions map."""
    versions = _versions_or_404(db, listing_id)
    summary = listing_summary(db, workspace_id, versions)
    shown = next(row for row in versions if str(row.id) == summary["packageId"])
    manifest = shown.manifest or {}
    latest = newest_compatible(versions)
    return {
        **summary,
        **previews_by_preset(shown),
        "readme": _file_url(shown, README_PATH),
        "changelog": _file_url(shown, CHANGELOG_PATH),
        "base": manifest.get("base"),
        "tone": manifest.get("tone"),
        "minWidth": manifest.get("minWidth"),
        "pages": sorted((manifest.get("pages") or {}).keys()),
        "routes": list(manifest.get("routes") or []),
        "latest": None if latest is None else latest.version,
        "versions": {row.version: _version_entry(db, row) for row in versions},
    }


# ---------------------------------------------------------------------------
# Install by reference
# ---------------------------------------------------------------------------


def _install_target(versions: Sequence[ThemePackage], package_id: Optional[UUID]) -> ThemePackage:
    """The version to install: the one asked for (listed, else 404), or the newest compatible one."""
    if package_id is None:
        target = newest_compatible(versions)
        if target is None:
            raise ThemeInstallError(CONFLICT, NO_COMPATIBLE)
        return target
    target = next((row for row in versions if row.id == package_id), None)
    if target is None:
        raise ThemeInstallError(NOT_FOUND, VERSION_NOT_FOUND)
    if not is_compatible(target):
        raise ThemeInstallError(CONFLICT, (
            f"Version {target.version} needs another Automatos (this host is {THEME_HOST_VERSION}, SDK {SDK_VERSION}). "
            "Choose a version marked compatible."
        ))
    return target


def _installed_refusal(db: Session, install: ThemeInstallation) -> ThemeInstallError:
    """Why a workspace that already runs this theme can't install it again from the listing."""
    package = db.get(ThemePackage, install.package_id)
    theme = echo(install.theme_id)
    if package is not None and package.owner_type == MARKETPLACE_OWNER:
        return ThemeInstallError(CONFLICT, (
            f"This workspace has {theme} {package.version} installed. "
            "Update it from Settings, Appearance, Themes."
        ))
    return ThemeInstallError(CONFLICT, (
        f"This workspace runs its own upload of {theme}. Uninstall it first to install the Marketplace version."
    ))


def _existing_or_refusal(db: Session, install: ThemeInstallation, target: ThemePackage) -> ThemeInstallation:
    if install.package_id == target.id:
        return install
    raise _installed_refusal(db, install)


def install_listing(
    db: Session, workspace_id: UUID, user_pk: Optional[int], listing_id: str, package_id: Optional[UUID],
) -> Tuple[ThemeInstallation, bool]:
    """Install a listed version by reference; ``(install, created)``.

    The same version installed already is answered with that install (``created`` False); any
    other install of the theme in this workspace is a 409 naming the way forward (update, or
    uninstall a private upload). The row is flushed, not committed: the router commits once.
    """
    versions = _versions_or_404(db, listing_id)
    target = _install_target(versions, package_id)
    existing = active_installation(db, workspace_id, target.theme_id, lock=True)
    if existing is not None:
        return _existing_or_refusal(db, existing, target), False
    newer = newer_compatible(versions, target)
    install = ThemeInstallation(
        workspace_id=workspace_id, theme_id=target.theme_id, package_id=target.id, installed_by=user_pk,
        pending_package_id=None if newer is None else newer.id,
    )
    db.add(install)
    try:
        db.flush()
    except IntegrityError:
        # uq_theme_installations_active: another request installed this theme first. This one
        # has only read and added the row, so rolling back loses nothing; answer what won.
        db.rollback()
        logger.info("theme %s was installed concurrently in workspace %s", target.theme_id, workspace_id)
        winner = active_installation(db, workspace_id, target.theme_id)
        if winner is None:
            raise ThemeInstallError(CONFLICT, CONCURRENT_INSTALL) from None
        return _existing_or_refusal(db, winner, target), False
    return install, True
