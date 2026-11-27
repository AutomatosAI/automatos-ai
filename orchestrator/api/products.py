"""A product's own routes, called with its key (PRD-274 "Automatos for products", Wave 5b, §5.1).

``POST /api/products/{product_id}/workspaces`` finds or creates the workspace of one of the product's customers,
by the product's own id for that customer (``external_id``): the workspace carries ``settings.product_id``, starts
with the product's roster, and gets a key of the product's type and scopes (``services/product_workspaces``). It
answers 201 with a new workspace and 200 with one found. ``api_key`` is the workspace's new key, once; it is null
when the product reuses keys and the workspace already has an active one.

Its workspaces, by id (``services/product_lifecycle``, decision 25):

- ``GET /api/products/{product_id}/workspaces``: the product's workspaces, oldest first; closed ones too with
  ``include_closed=true``, each with when its purge is due.
- ``DELETE /api/products/{product_id}/workspaces/{workspace_id}``: closes it, 202. It stops working at once,
  and is purged after ``PRODUCT_CLOSE_GRACE_DAYS`` (7) unless it is restored. Closing it again changes nothing.
- ``POST /api/products/{product_id}/workspaces/{workspace_id}/restore``: opens it again within the grace (409
  when it isn't closed, 410 when its grace has passed).
- ``PATCH /api/products/{product_id}/workspaces/{workspace_id}``: its own monthly credit cap (``credit_cap``;
  ``null``: back to the product's default), answered with the cap in force and what it spent this month.

The caller is the product's server, with ``Authorization: Bearer <the product's key>`` (``core/models/products``),
not a signed-in person, so these routes take no workspace context. A wrong key, or a product that doesn't exist,
is 401 either way. 404 while ``PRODUCTS_ENABLED`` is off. A product reaches only its own workspaces.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from config import config
from core.database.database import get_db
from core.models.products import Product

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/products", tags=["Products"])

EXTERNAL_ID_PATTERN = r"^[A-Za-z0-9._:@/-]{1,255}$"
METADATA_MAX_BYTES = 8192
DOMAINS_MAX = 50
CAP_MAX = 100_000_000  # credits a month (a million dollars at Automatos prices): a typo's guard, not a plan
BAD_KEY = "Invalid product key"
CLOSED = "That workspace is closed. Restore it, or wait for its purge to make a new one."
NOT_THEIRS = "This product has no workspace with that id."
NOT_CLOSED = "That workspace isn't closed."
PAST_GRACE = "That workspace's grace has passed; its purge is due, and it can't be restored."


class WorkspaceRequest(BaseModel):
    external_id: str = Field(..., pattern=EXTERNAL_ID_PATTERN,
                             description="The product's own id for this customer (letters, digits and . _ : @ / -)")
    name: str = Field(..., min_length=1, max_length=255, description="The workspace's name, when it is created")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Kept on a new workspace, for the product")
    allowed_domains: List[str] = Field(default_factory=list, max_length=DOMAINS_MAX,
                                       description="Origins a public key may be used from (required for one)")

    @field_validator("metadata")
    @classmethod
    def _small(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if len(json.dumps(value)) > METADATA_MAX_BYTES:
            raise ValueError(f"metadata is over {METADATA_MAX_BYTES} bytes")
        return value


def _bearer(authorization: Optional[str]) -> Optional[str]:
    scheme, _, token = (authorization or "").partition(" ")
    return token.strip() or None if scheme.lower() == "bearer" else None


def the_product(product_id: UUID, authorization: Optional[str] = Header(None),
                db: Session = Depends(get_db)) -> Product:
    """The product the path names, when the caller presents its key (401 otherwise)."""
    from services.products import product_for_key

    if not config.PRODUCTS_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")
    product = product_for_key(db, product_id, _bearer(authorization))
    if product is None:
        raise HTTPException(status_code=401, detail=BAD_KEY, headers={"WWW-Authenticate": "Bearer"})
    return product


@router.post("/{product_id}/workspaces")
def find_or_create_workspace(body: WorkspaceRequest, response: Response, product: Product = Depends(the_product),
                             db: Session = Depends(get_db)) -> Dict[str, Any]:
    """The workspace for ``external_id``: 201 when it was created, 200 when found; 409 when it is closed."""
    from services.product_workspaces import ProductGone, WorkspaceClosed, WorkspaceRequestInvalid, find_or_create

    try:
        result = find_or_create(db, product, external_id=body.external_id, name=body.name, metadata=body.metadata,
                                allowed_domains=body.allowed_domains)
    except WorkspaceRequestInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WorkspaceClosed as exc:
        raise HTTPException(status_code=409, detail=CLOSED) from exc
    except ProductGone as exc:
        raise HTTPException(status_code=401, detail=BAD_KEY) from exc
    response.status_code = 201 if result["is_new"] else 200
    return result


@router.get("/{product_id}/workspaces")
def list_workspaces(include_closed: bool = Query(False), limit: int = Query(100, ge=1, le=200),
                    offset: int = Query(0, ge=0), product: Product = Depends(the_product),
                    db: Session = Depends(get_db)) -> Dict[str, Any]:
    """The product's workspaces, oldest first; closed ones only with ``include_closed``."""
    from services.product_lifecycle import list_theirs

    rows = list_theirs(db, product, include_closed=include_closed, limit=limit, offset=offset)
    return {"workspaces": rows, "limit": limit, "offset": offset}


@router.delete("/{product_id}/workspaces/{workspace_id}", status_code=202)
def close_workspace(workspace_id: UUID, product: Product = Depends(the_product),
                    db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Closes the workspace now (202): its purge is due when ``purge_after`` passes, unless it is restored."""
    from services.product_lifecycle import NotTheirs, close

    try:
        return close(db, product, workspace_id)
    except NotTheirs as exc:
        raise HTTPException(status_code=404, detail=NOT_THEIRS) from exc


@router.post("/{product_id}/workspaces/{workspace_id}/restore")
def restore_workspace(workspace_id: UUID, product: Product = Depends(the_product),
                      db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Opens a closed workspace again, within its grace."""
    from services.product_lifecycle import NotClosed, NotTheirs, PastGrace, restore

    try:
        return restore(db, product, workspace_id)
    except NotTheirs as exc:
        raise HTTPException(status_code=404, detail=NOT_THEIRS) from exc
    except NotClosed as exc:
        raise HTTPException(status_code=409, detail=NOT_CLOSED) from exc
    except PastGrace as exc:
        raise HTTPException(status_code=410, detail=PAST_GRACE) from exc


class CapRequest(BaseModel):
    credit_cap: Optional[int] = Field(..., ge=1, le=CAP_MAX,
                                      description="Credits this workspace may spend a calendar month (UTC); "
                                                  "null: the product's default")


@router.patch("/{product_id}/workspaces/{workspace_id}")
def set_workspace_cap(workspace_id: UUID, body: CapRequest, product: Product = Depends(the_product),
                      db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Sets one workspace's monthly credit cap; a capped workspace's platform-key work stops at it, as it does at
    an owner's empty balance, with the code ``workspace_cap_reached``."""
    from services.product_lifecycle import NotTheirs, set_cap

    try:
        return set_cap(db, product, workspace_id, body.credit_cap)
    except NotTheirs as exc:
        raise HTTPException(status_code=404, detail=NOT_THEIRS) from exc
