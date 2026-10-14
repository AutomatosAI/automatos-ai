"""PRD-259 §5 — a workspace's skins in the database: private imports, installations, visibility.

Tenant isolation: every query here is filtered by the caller's workspace. A private
skin (``owner_type`` 'workspace') is visible only to the workspace that imported it;
a Marketplace skin only where it is installed. Another workspace's skin is simply not
found, so the routes answer 404, never 403.

``marketplace_skins.skin`` holds the validated skin.json plus the asset manifest under
``MANIFEST_KEY``; :func:`skin_of` and :func:`manifest_of` split them again.

Synchronous (a plain SQLAlchemy session): called from plain-``def`` routes only.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from uuid import UUID

from sqlalchemy import and_, exists, or_
from sqlalchemy.orm import Session

from core.models.marketplace_skins import MarketplaceSkin, SkinInstallation
from services.skins.package import SkinPackage

MANIFEST_KEY = "manifest"
OWNER_WORKSPACE = "workspace"
OWNER_MARKETPLACE = "marketplace"
STATUS_APPROVED = "approved"
# The skin.json fields the style dropdown reads, beside the row's own id, slug, name and version.
PAYLOAD_FIELDS: Tuple[str, ...] = ("base", "tones", "tokens", "fonts", "shape", "mascot", "preview")


class StaleFiles(NamedTuple):
    """Files a re-import replaced: the old prefix and the paths under it no manifest names any more."""

    prefix: str
    paths: Tuple[str, ...]


def stored_document(skin: Dict[str, Any], manifest: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The ``skin`` column's value: the validated skin.json plus its asset manifest."""
    return {**skin, MANIFEST_KEY: manifest}


def skin_of(row: MarketplaceSkin) -> Dict[str, Any]:
    """The validated skin.json a row stores, without the manifest."""
    return {key: value for key, value in (row.skin or {}).items() if key != MANIFEST_KEY}


def manifest_of(row: MarketplaceSkin) -> Dict[str, Dict[str, Any]]:
    """``{path: {type, size, sha256}}``: the only files the assets route serves for this row."""
    manifest = (row.skin or {}).get(MANIFEST_KEY)
    return dict(manifest) if isinstance(manifest, dict) else {}


def installed_item(row: MarketplaceSkin) -> Dict[str, Any]:
    """One skin as the style dropdown reads it (GET /api/skins/installed, POST /api/skins/import)."""
    skin = skin_of(row)
    return {
        "id": str(row.id),
        "slug": row.slug,
        "name": row.name,
        "version": row.version,
        **{field: skin.get(field) for field in PAYLOAD_FIELDS},
        "ownerType": row.owner_type,
    }


def _workspace(workspace_id: Any) -> UUID:
    return UUID(str(workspace_id))


def _own_private(workspace_id: UUID):
    return and_(MarketplaceSkin.owner_type == OWNER_WORKSPACE, MarketplaceSkin.workspace_id == workspace_id)


def _active_installation(workspace_id: UUID):
    return and_(SkinInstallation.workspace_id == workspace_id, SkinInstallation.uninstalled_at.is_(None))


def list_installed(db: Session, workspace_id: Any) -> List[Dict[str, Any]]:
    """This workspace's active installations joined to their skins, never another workspace's private skin."""
    ws = _workspace(workspace_id)
    rows = (
        db.query(MarketplaceSkin)
        .join(SkinInstallation, SkinInstallation.skin_id == MarketplaceSkin.id)
        .filter(_active_installation(ws))
        .filter(or_(_own_private(ws), MarketplaceSkin.owner_type == OWNER_MARKETPLACE))
        .order_by(MarketplaceSkin.name, MarketplaceSkin.id)
        .all()
    )
    return [installed_item(row) for row in rows]


def visible_skin(db: Session, workspace_id: Any, skin_id: UUID) -> Optional[MarketplaceSkin]:
    """The skin when this workspace may read its files: its own private skin, or one installed in it."""
    ws = _workspace(workspace_id)
    installed_here = exists().where(SkinInstallation.skin_id == MarketplaceSkin.id, _active_installation(ws))
    installed = and_(MarketplaceSkin.owner_type == OWNER_MARKETPLACE, installed_here)
    return (
        db.query(MarketplaceSkin)
        .filter(MarketplaceSkin.id == skin_id, or_(_own_private(ws), installed))
        .first()
    )


def _stale_files(row: Optional[MarketplaceSkin], prefix: str, manifest: Dict[str, Any]) -> StaleFiles:
    if row is None:
        return StaleFiles(prefix, ())
    kept = set(manifest) if row.asset_prefix == prefix else set()
    return StaleFiles(row.asset_prefix, tuple(sorted(set(manifest_of(row)) - kept)))


def _upsert_private_row(
    db: Session, ws: UUID, row: Optional[MarketplaceSkin], user_pk: Optional[int], package: SkinPackage, prefix: str,
) -> MarketplaceSkin:
    skin = package.skin
    values = {
        "name": skin["name"],
        "version": skin["version"],
        "skin": stored_document(skin, package.manifest),
        "asset_prefix": prefix,
        "preview_path": skin.get("preview"),
        "status": STATUS_APPROVED,
        "rejection_reason": None,
    }
    if row is None:
        row = MarketplaceSkin(
            slug=skin["slug"], owner_type=OWNER_WORKSPACE, workspace_id=ws, author_user_id=user_pk, **values,
        )
        db.add(row)
    else:
        for column, value in values.items():
            setattr(row, column, value)
    db.flush()
    return row


def _install(db: Session, ws: UUID, row: MarketplaceSkin, user_pk: Optional[int]) -> None:
    active = (
        db.query(SkinInstallation)
        .filter(SkinInstallation.skin_id == row.id, _active_installation(ws))
        .with_for_update()
        .first()
    )
    if active is None:
        db.add(SkinInstallation(workspace_id=ws, skin_id=row.id, version=row.version, installed_by=user_pk))
    else:
        active.version = row.version
    db.flush()


def save_private_import(
    db: Session, workspace_id: Any, user_pk: Optional[int], package: SkinPackage, prefix: str,
    store: Callable[[], None],
) -> Tuple[MarketplaceSkin, StaleFiles]:
    """Upsert this workspace's private skin for the package's slug and install it; commits.

    The same slug again replaces the row's content and version. ``store`` writes the files:
    it runs once the rows are flushed (the slug's row locked, a concurrent first import
    already refused by the unique index) and before the commit, so a refused or failed
    import never writes files, and a failed write commits nothing. Returns the row and
    the files the previous import stored that the new one no longer names.
    """
    ws = _workspace(workspace_id)
    existing = (
        db.query(MarketplaceSkin)
        .filter(_own_private(ws), MarketplaceSkin.slug == package.skin["slug"])
        .with_for_update()
        .first()
    )
    stale = _stale_files(existing, prefix, package.manifest)
    row = _upsert_private_row(db, ws, existing, user_pk, package, prefix)
    _install(db, ws, row, user_pk)
    try:
        store()
    except Exception:
        db.rollback()
        raise
    db.commit()
    db.refresh(row)
    return row, stale
