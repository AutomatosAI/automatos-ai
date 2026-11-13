"""A product finds or creates its customers' workspaces (PRD-274 Wave 5b, §5.1).

``find_or_create`` is ``provision_vertical``'s flow (``integrations/provisioning``) reading the product's
record instead of a provisioner in code: the workspace for the product's ``external_id``, or a new one; the
product's roster seeded into a workspace with no agents yet (``seed_roster_once``); a key of the product's type
and scopes (``workspace_key``). It differs where a product must:

- **A product reaches only its own.** A workspace is the product's when it carries its
  ``settings.product_id``; another product's workspace with the same external id is never found. One workspace
  per (product, external id), held by a unique index (``uq_workspaces_product_external_id``): of two requests
  that create the same one at once, the second finds the first's.
- **A closed workspace isn't reopened by a find** (``WorkspaceClosed``, 409): it is restored on purpose
  (decision 25, 5c), or purged after its grace, after which the external id makes a new one.
- **The product's row is share-locked to the first commit**, so an admin's delete of the product
  (``api/admin_products``, which locks it for update) waits, and then counts the new workspace and refuses.

A public key needs the origins it may be used from (``allowed_domains``), checked before anything is written.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from core.models.products import Product
from core.models.workspaces import Workspace

logger = logging.getLogger(__name__)

SOURCE = "product"
_THEIRS = text("settings->>'product_id' = :product AND settings->>'source_external_id' = :external_id")


class WorkspaceClosed(Exception):
    """The product's workspace for this external id is closed (soft-deleted)."""


class ProductGone(Exception):
    """The product was deleted while the request waited for it."""


class WorkspaceRequestInvalid(ValueError):
    """A request the product's record rejects (a public key with no origins)."""


@dataclass(frozen=True)
class _Provisioner:
    """The product's record in the shape ``seed_roster_once`` and ``workspace_key`` read."""

    agent_slugs: List[str]
    ops_manager_slug: Optional[str]
    key_type: str
    key_permissions: List[str]
    reuse_existing_key: bool

    @classmethod
    def of(cls, product: Product) -> "_Provisioner":
        return cls(list(product.agent_slugs or []), product.ops_manager_slug, product.key_type,
                   list(product.key_scopes or []), bool(product.reuse_existing_key))


def _theirs(db: Any, product: Product, external_id: str) -> Optional[Workspace]:
    return (db.query(Workspace).filter(_THEIRS)
            .params(product=str(product.id), external_id=external_id).first())


def _lock(db: Any, product: Product) -> None:
    """Share-locks the product's row to the next commit; ``ProductGone`` when an admin deleted it meanwhile."""
    product_id = product.id
    if db.query(Product.id).filter(Product.id == product_id).with_for_update(read=True).first() is None:
        db.rollback()
        raise ProductGone(str(product_id))


def _new_workspace(product: Product, external_id: str, name: str, metadata: Dict[str, Any]) -> Workspace:
    from integrations.provisioning import workspace_slug

    settings = {"product_id": str(product.id), "source": SOURCE, "source_external_id": external_id,
                "product_metadata": metadata}
    return Workspace(id=uuid4(), name=name, slug=workspace_slug(external_id), plan="basic", is_personal=False,
                     is_active=True, webhook_key=uuid4().hex, settings=settings)


def _found_or_made(db: Any, product: Product, external_id: str, name: str,
                   metadata: Dict[str, Any]) -> Tuple[Workspace, bool]:
    """The product's workspace for ``external_id`` and False, or a new one (flushed) and True."""
    _lock(db, product)
    workspace = _theirs(db, product, external_id)
    if workspace is not None:
        return workspace, False
    workspace = _new_workspace(product, external_id, name, metadata)
    db.add(workspace)
    try:
        db.flush()
    except IntegrityError:  # another request made it first, and has committed it
        db.rollback()
        _lock(db, product)
        workspace = _theirs(db, product, external_id)
        if workspace is None:
            raise
        return workspace, False
    logger.info("[products] created workspace %s for product %s, external id %r", workspace.id, product.slug,
                external_id)
    return workspace, True


def find_or_create(db: Any, product: Product, *, external_id: str, name: str, metadata: Dict[str, Any],
                   allowed_domains: List[str]) -> Dict[str, Any]:
    """The product's workspace for ``external_id``, found or created, with its roster and a key. Commits.
    Raises ``WorkspaceRequestInvalid`` (nothing written), ``WorkspaceClosed`` and ``ProductGone``."""
    from integrations.provisioning import seed_roster_once, workspace_key

    if product.key_type == "public" and not allowed_domains:
        raise WorkspaceRequestInvalid("allowed_domains is required: this product's workspaces get a public key")
    workspace, is_new = _found_or_made(db, product, external_id, name, metadata)
    if workspace.deleted_at is not None:
        db.rollback()
        raise WorkspaceClosed(external_id)
    provisioner = _Provisioner.of(product)
    installed = seed_roster_once(db, workspace, provisioner, f"product:{product.slug}")
    key = workspace_key(db, workspace, provisioner, name=f"{product.name} key ({external_id})",
                        allowed_domains=allowed_domains)
    db.commit()
    return {"workspace_id": str(workspace.id), "name": workspace.name, "external_id": external_id,
            "api_key": key["key"] if key else None, "key_minted": key is not None,
            "agents_installed": installed, "is_new": is_new}
