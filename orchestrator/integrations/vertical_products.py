"""A registered vertical moves onto a product record (PRD-274 Wave 5f, decisions 27 and 28).

A vertical (``integrations/<vertical>/provision.py``) that names its product's settings becomes a product: a
``products`` row whose slug is the vertical's, whose key is the vertical's internal key, and whose owner workspace
pays for every workspace the vertical provisions (Wave 5d). The vertical's code keeps its hooks (allowed origins,
the post-provision step, the widget defaults, the external-id key); the record holds its roster and the key each
workspace is minted, which ``provision_vertical`` then reads.

``seed_vertical_products`` runs on every boot, on the boot leader (``seed_vertical_products_on_boot``):

- **The record**, once: when the vertical has none, and both its internal key setting and its owner setting are
  set (the owner a live workspace), it is created with the key's hash, so the deployed key keeps working with no
  client change. The key's value is never stored or logged. Until then the vertical runs as before, on its
  environment key (``api/verticals``).
- **Its existing workspaces join it**: one per external id, the live one when there is one, else the newest,
  carrying ``settings.product_id`` (and ``source_external_id``, for workspaces made before the generic flow).
  Their end users keep their memories until erased (decision 28). Run again, it attaches only what is new.

The environment key stays one release (Dave, 13 Nov 2026): the record exists once a boot has seeded it, then the
setting and this seed's key read are deleted, and the variable comes off Railway.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

KEY_PREFIX_SEEDED = "env"  # an environment key's own characters are never shown

_ATTACH = text("""
UPDATE workspaces w
   SET settings = w.settings || jsonb_build_object('product_id', CAST(:product AS text), 'source_external_id', t.ext)
  FROM (SELECT DISTINCT ON (ext) id, ext
          FROM (SELECT id, created_at, (COALESCE(is_active, false) AND deleted_at IS NULL) AS live,
                       COALESCE(settings->>:id_key, settings->>'source_external_id') AS ext
                  FROM workspaces
                 WHERE (settings->>'vertical' = :vertical OR settings->>'source' = :vertical)
                   AND settings->>'product_id' IS NULL) candidates
         WHERE ext IS NOT NULL
         ORDER BY ext, live DESC, created_at DESC) t
 WHERE w.id = t.id
   AND NOT EXISTS (SELECT 1 FROM workspaces o
                    WHERE o.settings->>'product_id' = CAST(:product AS text)
                      AND o.settings->>'source_external_id' = t.ext)
""")


_DETACH = text("UPDATE workspaces SET settings = settings - 'product_id' WHERE settings->>'product_id' = :product "
               "AND settings->>'source_external_id' = :external_id AND id::text <> :workspace")


def join(db: Any, product: Any, workspace: Any, external_id: str) -> None:
    """The vertical's workspace for ``external_id`` carries its product. An older one for the same external id (a
    shop that reinstalled: the old one deactivated) stops carrying it, so the product has one workspace per
    external id (``uq_workspaces_product_external_id``). Flushes."""
    settings = dict(workspace.settings or {})
    if settings.get("product_id") == str(product.id):
        return
    db.execute(_DETACH, {"product": str(product.id), "external_id": external_id, "workspace": str(workspace.id)})
    workspace.settings = {**settings, "product_id": str(product.id), "source_external_id": external_id}
    db.flush()


def key_matches(vertical: str, presented: str) -> Optional[bool]:
    """Whether ``presented`` is the key of the vertical's product record, compared in constant time; None when
    the vertical has no record yet (its environment key is checked then, ``api/verticals``)."""
    import hmac

    from core.database.database import SessionLocal
    from core.models.products import Product
    from services.products import key_hash

    db = SessionLocal()
    try:
        stored = db.query(Product.key_hash).filter(Product.slug == vertical).scalar()
    finally:
        db.close()
    if stored is None:
        return None
    return bool(presented) and hmac.compare_digest(stored, key_hash(presented))


def product_of(db: Any, vertical: str) -> Optional[Any]:
    """The vertical's product record, or None while it has none."""
    from core.models.products import Product

    return db.query(Product).filter(Product.slug == vertical).first()


def _setting(name: Optional[str]) -> str:
    from config import config

    return (getattr(config, name, "") or "").strip() if name else ""


def _live_workspace(db: Any, workspace_id: str) -> bool:
    row = db.execute(text("SELECT 1 FROM workspaces WHERE id::text = :id AND deleted_at IS NULL"),
                     {"id": workspace_id}).first()
    return row is not None


def _create(db: Any, vertical: str, provisioner: Any, key: str, owner: str) -> Any:
    from core.models.products import Product
    from services.products import key_hash

    product = Product(slug=vertical, name=getattr(provisioner, "product_name", vertical), owner_workspace_id=owner,
                      key_hash=key_hash(key), key_prefix=KEY_PREFIX_SEEDED,
                      agent_slugs=list(provisioner.agent_slugs), ops_manager_slug=provisioner.ops_manager_slug,
                      key_type=getattr(provisioner, "key_type", "public"),
                      key_scopes=list(provisioner.key_permissions),
                      reuse_existing_key=bool(getattr(provisioner, "reuse_existing_key", False)))
    db.add(product)
    db.commit()
    logger.warning("[products] %s is now product %s, owned by workspace %s", vertical, product.id, owner)
    return product


def seed_vertical_product(db: Any, vertical: str, provisioner: Any) -> Dict[str, Any]:
    """The vertical's record (created when it has none and its settings allow), and how many of its
    workspaces joined it on this run. Nothing while products are off (``PRODUCTS_ENABLED``). Commits."""
    from config import config

    if not config.PRODUCTS_ENABLED:
        return {"vertical": vertical, "product": None, "reason": "products are off"}
    product = product_of(db, vertical)
    if product is None:
        key, owner = _setting(getattr(provisioner, "key_setting", None)), _setting(
            getattr(provisioner, "owner_setting", None))
        if not (key and owner):
            return {"vertical": vertical, "product": None, "reason": "its key or owner setting is not set"}
        if not _live_workspace(db, owner):
            logger.error("[products] %s's owner setting names no live workspace: no product made", vertical)
            return {"vertical": vertical, "product": None, "reason": "its owner is not a live workspace"}
        product = _create(db, vertical, provisioner, key, owner)
    attached = db.execute(_ATTACH, {"product": str(product.id), "vertical": vertical,
                                    "id_key": getattr(provisioner, "external_id_key", "source_external_id")}).rowcount
    db.commit()
    if attached:
        logger.warning("[products] %d of %s's workspaces joined product %s", attached, vertical, product.id)
    return {"vertical": vertical, "product": str(product.id), "attached": attached}


def seed_vertical_products(db: Any) -> list:
    """Every registered vertical that names its product's settings, seeded (``seed_vertical_product``)."""
    from integrations.provisioning import PROVISIONER_REGISTRY

    return [seed_vertical_product(db, vertical, provisioner)
            for vertical, provisioner in sorted(PROVISIONER_REGISTRY.items())
            if getattr(provisioner, "owner_setting", None)]


def seed_vertical_products_on_boot() -> None:
    """The boot leader's per-boot step (``main._boot_phase_1_core``): the verticals' products seeded, whatever
    the seed fingerprint (they depend on settings and the database, not on seed files). A failure is logged, and
    the next boot tries again; it never stops the boot."""
    from core.database.database import get_db_session

    try:
        with get_db_session() as db:
            logger.info("Vertical products: %s", seed_vertical_products(db))
    except Exception:
        logger.exception("Vertical products seed failed; the next boot tries again")
