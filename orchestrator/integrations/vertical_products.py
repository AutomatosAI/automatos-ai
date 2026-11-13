"""A registered vertical is a product (PRD-274 Wave 5f, decisions 27–31).

A vertical's product record is a ``products`` row whose slug is the vertical's: its key is the vertical's
internal key, and its owner workspace pays for every workspace the vertical provisions (Wave 5d). The vertical's
code keeps its hooks (allowed origins, the post-provision step, the widget defaults, the external-id key); the
record holds its roster and the key each workspace is minted, which ``provision_vertical`` reads.

- ``key_matches``: the vertical's machine key is its record's (``api/verticals``); a vertical with no record is
  dark (503). An admin replaces the key (``POST /api/admin/products/{id}/key``) or creates the record.
- ``join``: a provisioned workspace carries its vertical's product, one per external id.

The records were seeded once from the verticals' environment keys, which are gone since (decision 30).
"""
from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import text

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


def key_matches(vertical: str, presented: str, db: Optional[Any] = None) -> Optional[bool]:
    """Whether ``presented`` is the key of the vertical's product record, compared in constant time; None when
    the vertical has no record (``api/verticals`` then answers 503). Reads on ``db``, the request's own session,
    when the caller has one, so a request takes no second connection; else on a session of its own."""
    import hmac

    from core.models.products import Product
    from services.products import key_hash

    if db is not None:
        stored = db.query(Product.key_hash).filter(Product.slug == vertical).scalar()
    else:
        from core.database.database import SessionLocal

        with SessionLocal() as own:
            stored = own.query(Product.key_hash).filter(Product.slug == vertical).scalar()
    if stored is None:
        return None
    return bool(presented) and hmac.compare_digest(stored, key_hash(presented))


def product_of(db: Any, vertical: str) -> Optional[Any]:
    """The vertical's product record, or None while it has none."""
    from core.models.products import Product

    return db.query(Product).filter(Product.slug == vertical).first()
