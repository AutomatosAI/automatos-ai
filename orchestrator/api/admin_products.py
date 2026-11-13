"""Platform admin: products (PRD-274 "Automatos for products", Wave 5a, §5.1, §8-3).

A product is an outside app that runs its customers' workspaces on Automatos (``core/models/products.py``).
These routes create, read, change and delete its record, and replace its key:

- ``GET /api/admin/products`` and ``GET /api/admin/products/{id}``: the records, never a key's hash.
- ``POST /api/admin/products``: creates one and mints its key, returned once in ``key``.
- ``PATCH /api/admin/products/{id}``: its name, roster, minted key and default retention. Its slug and owner
  workspace don't change: the owner pays for every workspace the product has (§8-2).
- ``POST /api/admin/products/{id}/key``: a new key, returned once; the old one stops working at once.
- ``DELETE /api/admin/products/{id}``: refused (409) while any workspace carries its ``settings.product_id``.

Each change is recorded in the owner workspace's audit log. Platform admins only (403 for anyone else),
in both editions, while ``PRODUCTS_ENABLED`` is on (404 otherwise). Nothing reads the record yet: 5b's
find-or-create is the first.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.api_keys import VALID_PERMISSIONS
from api.end_user_memory import RETENTION_MAX_DAYS
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.products import PRODUCT_NAME_MAX, PRODUCT_SLUG_MAX, Product
from core.utils.timestamps import utc_iso

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/products", tags=["Admin Products"])

SLUG_PATTERN = rf"^[a-z0-9][a-z0-9-]{{0,{PRODUCT_SLUG_MAX - 1}}}$"
AGENT_SLUG_MAX = 100
ROSTER_MAX = 50
NOT_FOUND = "No product with that id."
SLUG_TAKEN = "A product already has that slug."
HAS_WORKSPACES = "The product still has workspaces; close and purge them first."


class _Editable(BaseModel):
    """What an admin may set on a product, at creation and after."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(None, min_length=1, max_length=PRODUCT_NAME_MAX)
    agent_slugs: Optional[List[str]] = Field(None, max_length=ROSTER_MAX,
                                             description="Marketplace agents each customer's workspace starts with")
    ops_manager_slug: Optional[str] = Field(None, max_length=AGENT_SLUG_MAX,
                                            description="The roster agent the others report to; one of agent_slugs")
    key_type: Optional[Literal["public", "server"]] = None
    key_scopes: Optional[List[str]] = Field(None, description="The scopes of the key each customer's workspace gets")
    default_retention_days: Optional[int] = Field(None, ge=1, le=RETENTION_MAX_DAYS,
                                                  description="Days an end user's memories are kept when the product "
                                                              "records none for them; none: until erased")

    @field_validator("agent_slugs")
    @classmethod
    def _distinct_slugs(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is not None and (len(set(value)) != len(value) or any(not s or len(s) > AGENT_SLUG_MAX for s in value)):
            raise ValueError(f"agent_slugs must be distinct, each 1 to {AGENT_SLUG_MAX} characters")
        return value

    @field_validator("key_scopes")
    @classmethod
    def _known_scopes(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        unknown = [scope for scope in value or [] if scope not in VALID_PERMISSIONS]
        if unknown:
            raise ValueError(f"Unknown scopes: {unknown}. Valid: {VALID_PERMISSIONS}")
        return value


class ProductCreate(_Editable):
    slug: str = Field(..., pattern=SLUG_PATTERN, description="Lower-case letters, digits and hyphens")
    name: str = Field(..., min_length=1, max_length=PRODUCT_NAME_MAX)
    owner_workspace_id: UUID = Field(..., description="The workspace that pays for the product's workspaces")


def _require_admin(ctx: RequestContext) -> None:
    from config import config
    from core.auth.roles import caller_is_admin

    if not config.PRODUCTS_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")
    if not ctx.user or not caller_is_admin(ctx.user):
        raise HTTPException(status_code=403, detail="Admin access required")


def _view(product: Product) -> Dict[str, Any]:
    """A product as the routes answer it: everything but its key's hash."""
    return {
        "id": str(product.id), "slug": product.slug, "name": product.name,
        "owner_workspace_id": str(product.owner_workspace_id), "key_prefix": product.key_prefix,
        "agent_slugs": list(product.agent_slugs or []), "ops_manager_slug": product.ops_manager_slug,
        "key_type": product.key_type, "key_scopes": list(product.key_scopes or []),
        "default_retention_days": product.default_retention_days,
        "created_at": utc_iso(product.created_at), "updated_at": utc_iso(product.updated_at),
    }


def _product_or_404(db: Session, product_id: UUID) -> Product:
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    return product


def _check_owner(db: Session, workspace_id: UUID) -> None:
    """The owner is a live workspace, and not itself one of a product's workspaces (422 otherwise)."""
    from core.models.workspaces import Workspace

    owner = db.get(Workspace, workspace_id)
    if owner is None or owner.deleted_at is not None:
        raise HTTPException(status_code=422, detail="owner_workspace_id is not a live workspace")
    if (owner.settings or {}).get("product_id"):
        raise HTTPException(status_code=422, detail="owner_workspace_id is a product's workspace; it can't own one")


def _check_roster(db: Session, agent_slugs: List[str], ops_manager_slug: Optional[str]) -> None:
    """Every slug names an approved marketplace agent, and the ops manager is one of them (422 otherwise)."""
    from core.models.core import Agent

    if ops_manager_slug is not None and ops_manager_slug not in agent_slugs:
        raise HTTPException(status_code=422, detail="ops_manager_slug must be one of agent_slugs")
    if not agent_slugs:
        return
    found = {slug for (slug,) in db.query(Agent.slug).filter(
        Agent.owner_type == "marketplace", Agent.is_approved.is_(True), Agent.slug.in_(agent_slugs))}
    missing = sorted(set(agent_slugs) - found)
    if missing:
        raise HTTPException(status_code=422, detail=f"No approved marketplace agent with slug: {missing}")


def _audit(db: Session, ctx: RequestContext, product: Dict[str, Any], action: str, details: Dict[str, Any]) -> None:
    """Records the change to ``product`` (its view) in the owner workspace's audit log, once it is committed."""
    from core.auth.actor import resolve_internal_user_id
    from core.workspaces.audit import AuditService

    AuditService(db).log(
        workspace_id=product["owner_workspace_id"], user_id=resolve_internal_user_id(db, ctx),
        action=f"product.{action}", resource_type="product", resource_id=product["id"],
        resource_name=product["name"], details={**details, "actor_email": getattr(ctx.user, "email", None)})


@router.get("")
def list_products(ctx: RequestContext = Depends(get_request_context_hybrid),
                  db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Every product, by slug."""
    _require_admin(ctx)
    products = db.query(Product).order_by(Product.slug).all()
    return {"products": [_view(product) for product in products], "total": len(products)}


@router.get("/{product_id}")
def get_product(product_id: UUID, ctx: RequestContext = Depends(get_request_context_hybrid),
                db: Session = Depends(get_db)) -> Dict[str, Any]:
    """One product, with how many workspaces carry it."""
    from services.products import workspaces_of

    _require_admin(ctx)
    product = _product_or_404(db, product_id)
    return {**_view(product), "workspaces": workspaces_of(db, product.id)}


@router.post("", status_code=201)
def create_product(body: ProductCreate, ctx: RequestContext = Depends(get_request_context_hybrid),
                   db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Creates a product and mints its key; the key is in this answer only."""
    from services.products import mint_key

    _require_admin(ctx)
    _check_owner(db, body.owner_workspace_id)
    agent_slugs = list(body.agent_slugs or [])
    _check_roster(db, agent_slugs, body.ops_manager_slug)
    key = mint_key()
    product = Product(slug=body.slug, name=body.name, owner_workspace_id=body.owner_workspace_id,
                      key_hash=key.key_hash, key_prefix=key.key_prefix, agent_slugs=agent_slugs,
                      ops_manager_slug=body.ops_manager_slug, key_type=body.key_type or "server",
                      key_scopes=list(body.key_scopes or []), default_retention_days=body.default_retention_days)
    db.add(product)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=SLUG_TAKEN) from exc
    db.refresh(product)
    view = _view(product)
    _audit(db, ctx, view, "created", {"slug": product.slug, "key_prefix": product.key_prefix})
    logger.info("[admin] product %s created for workspace %s", product.slug, product.owner_workspace_id)
    return {"product": view, "key": key.plaintext}


@router.patch("/{product_id}")
def update_product(product_id: UUID, body: _Editable, ctx: RequestContext = Depends(get_request_context_hybrid),
                   db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Changes what the body names; the rest stays. ``default_retention_days: null`` clears it."""
    _require_admin(ctx)
    product = _product_or_404(db, product_id)
    changes = body.model_dump(exclude_unset=True)
    for field in ("name", "agent_slugs", "key_type", "key_scopes"):
        if field in changes and changes[field] is None:
            raise HTTPException(status_code=422, detail=f"{field} can't be null")
    if "agent_slugs" in changes or "ops_manager_slug" in changes:
        _check_roster(db, changes.get("agent_slugs", list(product.agent_slugs or [])),
                      changes.get("ops_manager_slug", product.ops_manager_slug))
    for field, value in changes.items():
        setattr(product, field, value)
    db.commit()
    db.refresh(product)
    view = _view(product)
    _audit(db, ctx, view, "updated", {"changed": sorted(changes)})
    return view


@router.post("/{product_id}/key")
def replace_product_key(product_id: UUID, ctx: RequestContext = Depends(get_request_context_hybrid),
                        db: Session = Depends(get_db)) -> Dict[str, Any]:
    """A new key for the product, in this answer only; the old key stops working now."""
    from services.products import mint_key

    _require_admin(ctx)
    product = _product_or_404(db, product_id)
    key = mint_key()
    previous = product.key_prefix
    product.key_hash, product.key_prefix = key.key_hash, key.key_prefix
    db.commit()
    db.refresh(product)
    view = _view(product)
    _audit(db, ctx, view, "key_replaced", {"previous_key_prefix": previous, "key_prefix": key.key_prefix})
    return {"product": view, "key": key.plaintext}


@router.delete("/{product_id}")
def delete_product(product_id: UUID, ctx: RequestContext = Depends(get_request_context_hybrid),
                   db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Deletes the product, and with it its key; refused (409) while any workspace carries it. The product's
    row is locked while it is checked, so a find-or-create (5b), which reads it first, waits for the delete."""
    from services.products import workspaces_of

    _require_admin(ctx)
    product = db.query(Product).filter(Product.id == product_id).with_for_update().first()
    if product is None:
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    if workspaces_of(db, product.id):
        db.rollback()
        raise HTTPException(status_code=409, detail=HAS_WORKSPACES)
    gone = _view(product)
    db.delete(product)
    db.commit()
    _audit(db, ctx, gone, "deleted", {"slug": gone["slug"]})
    return {"deleted": True, "product": gone}
