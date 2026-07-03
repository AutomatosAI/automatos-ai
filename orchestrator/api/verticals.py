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
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models.workspaces import Workspace
from api.shopify import _verify_internal_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/verticals", tags=["Verticals"])


def _resolve_workspace_by_external_id(db: Session, vertical: str, external_id: str) -> Workspace:
    """Resolve the active workspace for ``(vertical, external_id)`` or raise 404.

    This is the machine-surface counterpart to the user-facing GDPR endpoints'
    ``ctx.workspace_id``: a webhook arrives with only an internal key + the shop /
    external id and no session, so the workspace is resolved from the external id.

    It uses the SAME indexed lookup the generic provision flow
    (``provision_vertical``) and the Shopify ``/events`` / ``/sync`` /
    ``/deactivate`` routes use — filtering on the vertical provisioner's stamped
    ``external_id_key`` (e.g. ``shopify_domain``), falling back to the canonical
    ``source_external_id``. If nothing matches we raise 404 and **never** fall back
    to a blank/wrong workspace — erasing the wrong tenant is the failure mode this
    guards against.
    """
    from integrations.provisioning import PROVISIONER_REGISTRY

    provisioner = PROVISIONER_REGISTRY.get(vertical)
    id_key = getattr(provisioner, "external_id_key", "source_external_id") if provisioner else "source_external_id"

    workspace = (
        db.query(Workspace)
        .filter(
            Workspace.settings[id_key].astext == external_id,
            Workspace.is_active.is_(True),
        )
        .first()
    )
    if workspace is None and id_key != "source_external_id":
        # Belt-and-braces: a workspace stamped only with the canonical key.
        workspace = (
            db.query(Workspace)
            .filter(
                Workspace.settings["source_external_id"].astext == external_id,
                Workspace.is_active.is_(True),
            )
            .first()
        )
    if workspace is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active {vertical} workspace for external_id '{external_id}'",
        )
    return workspace


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


# ===================================================================
# PRD-183 — machine-to-machine GDPR surface (Shopify compliance webhooks)
# ===================================================================
#
# A Shopify GDPR webhook (customers/redact, shop/redact, customers/data_request)
# arrives machine-to-machine with the internal key + a shop domain and NO user /
# workspace session, so it cannot use the user-facing ``/api/v1/gdpr/*`` endpoints
# (those require a logged-in workspace admin and resolve the workspace from
# ``ctx.workspace_id``). These endpoints are the internal-key-authed twin: the
# workspace is resolved from ``external_id`` (the shop domain), and the erasure /
# export is delegated to the SAME ``services.gdpr_service`` cascade W11 built —
# which audits every operation. The user-facing endpoints keep their admin auth
# untouched; this is a separate surface.


class VerticalGdprEraseSubjectRequest(BaseModel):
    external_id: str = Field(..., description="Vertical-scoped external id (e.g. shop domain).")
    subject_id: str = Field(..., description="Data subject id to erase (e.g. Shopify customer id).")


class VerticalGdprEraseRequest(BaseModel):
    external_id: str = Field(..., description="Vertical-scoped external id (e.g. shop domain).")


def _webhook_actor(vertical: str, external_id: str) -> str:
    """Stable ``requested_by`` provenance for an internal-key GDPR call."""
    return f"{vertical}-webhook:{external_id}"


@router.post("/{vertical}/gdpr/erase-subject")
async def gdpr_erase_subject(
    vertical: str,
    request: VerticalGdprEraseSubjectRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_internal_key),
) -> Dict[str, Any]:
    """Erase a single data subject within the workspace resolved from ``external_id``.

    Backs Shopify ``customers/redact``. Delegates to
    ``gdpr_service.erase_data_subject`` (audited; stores lacking a subject tag are
    reported in ``gaps``). 404 if no active workspace matches the external id.
    """
    workspace = _resolve_workspace_by_external_id(db, vertical, request.external_id)

    from services.gdpr_service import erase_data_subject

    result = erase_data_subject(
        db,
        workspace_id=workspace.id,
        subject_id=request.subject_id,
        requested_by=_webhook_actor(vertical, request.external_id),
    )
    logger.info(
        "[PRD-183 GDPR] erase-subject vertical=%s external_id=%s workspace=%s subject=%s",
        vertical, request.external_id, workspace.id, request.subject_id,
    )
    return result


@router.post("/{vertical}/gdpr/erase")
async def gdpr_erase_workspace(
    vertical: str,
    request: VerticalGdprEraseRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_internal_key),
) -> Dict[str, Any]:
    """Erase the whole workspace resolved from ``external_id`` (irreversible cascade).

    Backs Shopify ``shop/redact`` (fired 48h after uninstall). The confirmation
    echo the user-facing endpoint requires is unnecessary here: the caller cannot
    name a workspace id at all — it is resolved server-side from the shop domain,
    and a non-matching shop 404s rather than erasing anything. Delegates to
    ``gdpr_service.erase_workspace`` (audited).
    """
    workspace = _resolve_workspace_by_external_id(db, vertical, request.external_id)

    from services.gdpr_service import erase_workspace

    result = erase_workspace(
        db,
        workspace.id,
        requested_by=_webhook_actor(vertical, request.external_id),
    )
    logger.info(
        "[PRD-183 GDPR] erase-workspace vertical=%s external_id=%s workspace=%s",
        vertical, request.external_id, workspace.id,
    )
    return result


@router.get("/{vertical}/gdpr/export")
async def gdpr_export_workspace(
    vertical: str,
    external_id: str,
    customer_id: Optional[str] = None,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_internal_key),
) -> JSONResponse:
    """Export the workspace resolved from ``external_id`` as a portable JSON bundle.

    Backs Shopify ``customers/data_request``. ``customer_id`` is accepted for
    parity with the Shopify payload and audit provenance; the export is
    workspace-scoped (the platform has no data-subject tag on the derived stores —
    the same gap the subject-level erasure documents). Delegates to
    ``gdpr_service.export_workspace`` (audited). 404 if no workspace matches.
    """
    workspace = _resolve_workspace_by_external_id(db, vertical, external_id)

    from services.gdpr_service import export_workspace

    bundle = export_workspace(
        db,
        workspace.id,
        requested_by=_webhook_actor(vertical, external_id),
    )
    logger.info(
        "[PRD-183 GDPR] export vertical=%s external_id=%s workspace=%s customer_id=%s",
        vertical, external_id, workspace.id, customer_id,
    )
    return JSONResponse(content=bundle)
