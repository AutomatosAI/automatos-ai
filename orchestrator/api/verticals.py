"""Generic vertical-provisioning API (PRD-183 S5, F076).

``POST /api/verticals/{vertical}/provision`` is the ONE install-time entry point
for standing up an Automatos workspace for any vertical. It looks up the
vertical's :class:`~integrations.provisioning.VerticalProvisioner` and runs the
generic flow — so a second vertical (booking, support, …) provisions without
forking ``api/shopify.py``. The Shopify Remix app targets
``/api/verticals/shopify/provision``; ``api/shopify.py`` retains only its thin
compat wrapper that delegates here.

Authenticated with the same fail-closed internal API key as the Shopify routes
(the integration/app server is the caller, not a browser).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.database.database import get_db
from api.shopify import _verify_internal_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/verticals", tags=["Verticals"])


class VerticalProvisionRequest(BaseModel):
    external_id: str = Field(..., description="Vertical-scoped external id (e.g. shop domain).")
    name: str = Field(..., description="Workspace display name.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerticalProvisionResponse(BaseModel):
    id: str
    public_id: str
    name: str
    api_key: str = Field(..., description="Public widget API key — shown once.")
    agents_installed: int
    is_new: bool


@router.post("/{vertical}/provision", response_model=VerticalProvisionResponse)
async def provision_vertical_workspace(
    vertical: str,
    request: VerticalProvisionRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_internal_key),
):
    """Provision a workspace for ``vertical`` via the generic provisioner.

    Idempotent (an existing active workspace for ``external_id`` is returned
    without re-seeding). Returns 404 for a vertical with no registered
    provisioner.
    """
    from integrations.provisioning import provision_vertical

    try:
        result = provision_vertical(
            db=db,
            vertical=vertical,
            external_id=request.external_id,
            name=request.name,
            metadata=request.metadata,
        )
    except ValueError as e:
        # Unknown vertical → 404 (no provisioner registered).
        raise HTTPException(status_code=404, detail=str(e))

    logger.info(
        "Provisioned %s workspace %s for %s (%d agents)",
        vertical, result["id"], request.external_id, result["agents_installed"],
    )
    return VerticalProvisionResponse(**result)
