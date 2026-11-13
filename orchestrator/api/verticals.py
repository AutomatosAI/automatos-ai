"""Generic vertical-provisioning API (PRD-183 S5, F076).

``POST /api/verticals/{vertical}/provision`` is the ONE install-time entry point
for standing up an Automatos workspace for any vertical. It looks up the
vertical's :class:`~integrations.provisioning.VerticalProvisioner` and runs the
generic flow — so a second vertical (booking, support, …) provisions without
forking ``api/shopify.py``. The Shopify Remix app targets
``/api/verticals/shopify/provision``; ``api/shopify.py`` retains only its thin
compat wrapper that delegates here.

Authenticated with a fail-closed PER-VERTICAL internal API key (the
integration/app server is the caller, not a browser): the key of the vertical's
product record (PRD-274 W5f), resolved from the path's ``{vertical}`` — so the
BudStacks app cannot provision Shopify workspaces with its own key, and a
vertical with no product record answers 503 (dark), mirroring
``_verify_internal_key``'s posture (see the 2026-08-01 boot-guard outage note in
config.py for why this is a request-time check, never a boot-fatal one).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models.workspaces import Workspace

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/verticals", tags=["Verticals"])


def _verify_vertical_internal_key(
    vertical: str, authorization: str = Header(...)
) -> None:
    """Fail-closed per-vertical internal key (``api.shopify._verify_internal_key`` calls it too): the key of the
    vertical's product record (PRD-274 W5f, ``integrations/vertical_products``), compared by hash in constant time,
    so an admin's key replacement takes effect at once. A vertical with no record is dark (503), never open."""
    from integrations.vertical_products import key_matches

    token = authorization.replace("Bearer ", "").strip()
    matched = key_matches(vertical, token)
    if matched is None:
        raise HTTPException(status_code=503, detail=f"{vertical} provisioning not configured")
    if not matched:
        raise HTTPException(status_code=401, detail="Invalid internal API key")


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


def _shop_workspaces(db: Session, vertical: str, external_id: str) -> List[Workspace]:
    """Every workspace provisioned for ``(vertical, external_id)``, active or not, by the same
    keys ``_resolve_workspace_by_external_id`` reads (a reinstall provisions a new one, so a
    shop can have several)."""
    from integrations.provisioning import PROVISIONER_REGISTRY

    provisioner = PROVISIONER_REGISTRY.get(vertical)
    id_key = getattr(provisioner, "external_id_key", "source_external_id") if provisioner else "source_external_id"
    found: Dict[Any, Workspace] = {}
    for key in dict.fromkeys((id_key, "source_external_id")):
        for workspace in db.query(Workspace).filter(Workspace.settings[key].astext == external_id).all():
            found.setdefault(workspace.id, workspace)
    return list(found.values())


def _workspaces_to_erase(db: Session, vertical: str, external_id: str) -> List[Workspace]:
    """``shop/redact``'s workspaces (#1442): every one of the shop's that is no longer active,
    deactivated on uninstall (``shop/redact`` comes 48 hours after it) or closed by an erase a
    retry finishes; a reinstalled shop's live workspace is kept. A shop with only active ones
    erases those, as before. 404 when the shop has none: never a blank or wrong workspace."""
    found = _shop_workspaces(db, vertical, external_id)
    gone = [w for w in found if not w.is_active or getattr(w, "deleted_at", None) is not None]
    if not (gone or found):
        raise HTTPException(status_code=404, detail=f"No {vertical} workspace for external_id '{external_id}'")
    return gone or found


class VerticalProvisionRequest(BaseModel):
    external_id: str = Field(..., description="Vertical-scoped external id (e.g. shop domain).")
    name: str = Field(..., description="Workspace display name.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerticalProvisionResponse(BaseModel):
    id: str
    public_id: str
    name: str
    api_key: Optional[str] = Field(
        None,
        description=(
            "Public widget API key — shown once when minted. Null when the "
            "vertical declares reuse_existing_key and an active key already "
            "exists (the caller keeps its stored copy)."
        ),
    )
    key_minted: bool = Field(
        True, description="Whether this call minted a new key."
    )
    agents_installed: int
    is_new: bool


@router.post("/{vertical}/provision", response_model=VerticalProvisionResponse)
def provision_vertical_workspace(
    vertical: str,
    request: VerticalProvisionRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_vertical_internal_key),
):
    """Provision a workspace for ``vertical`` via the generic provisioner.

    Idempotent (an existing active workspace for ``external_id`` is returned
    without re-seeding). Returns 404 for a vertical with no registered
    provisioner.
    """
    from integrations.provisioning import VerticalConfigError, provision_vertical

    try:
        result = provision_vertical(
            db=db,
            vertical=vertical,
            external_id=request.external_id,
            name=request.name,
            metadata=request.metadata,
        )
    except VerticalConfigError as e:
        # Malformed by the vertical's own rules (e.g. missing origin allowlist).
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        # Unknown vertical → 404 (no provisioner registered).
        raise HTTPException(status_code=404, detail=str(e))

    logger.info(
        "Provisioned %s workspace %s for %s (%d agents)",
        vertical, result["id"], request.external_id, result["agents_installed"],
    )
    return VerticalProvisionResponse(**result)


class VerticalDomainsRequest(BaseModel):
    external_id: str = Field(..., description="Vertical-scoped external id.")
    domains: List[str] = Field(
        ..., min_length=1, description="Replacement origin allowlist (hostnames or origins)."
    )


@router.patch("/{vertical}/provision/domains")
def update_vertical_key_domains(
    vertical: str,
    request: VerticalDomainsRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_vertical_internal_key),
) -> Dict[str, Any]:
    """Re-sync the origin allowlist on the workspace's active public widget keys.

    Public keys are origin-locked; when a tenant's storefront hostname changes
    (e.g. a BudStacks custom-domain move) the allowlist must follow or the
    widget 403s on the new host. Replaces ``allowed_domains`` on every ACTIVE
    public key of the workspace resolved from ``external_id`` — inactive and
    server keys are untouched. Domains are normalized through the vertical's
    own ``allowed_domains`` rule so the same validation applies as at
    provision time. 404 when no workspace matches; 422 when the vertical's
    rules reject the list.
    """
    from core.models.sdk_api_keys import SdkApiKey
    from integrations.provisioning import (
        PROVISIONER_REGISTRY,
        VerticalConfigError,
    )

    workspace = _resolve_workspace_by_external_id(db, vertical, request.external_id)

    provisioner = PROVISIONER_REGISTRY.get(vertical)
    if provisioner is None:
        raise HTTPException(status_code=404, detail=f"Unknown vertical '{vertical}'")

    try:
        domains = provisioner.allowed_domains(
            request.external_id, {"domains": request.domains}
        )
    except VerticalConfigError as e:
        raise HTTPException(status_code=422, detail=str(e))

    keys = (
        db.query(SdkApiKey)
        .filter(
            SdkApiKey.workspace_id == workspace.id,
            SdkApiKey.key_type == "public",
            SdkApiKey.is_active.is_(True),
        )
        .all()
    )
    for key in keys:
        key.allowed_domains = list(domains)
    db.commit()

    logger.info(
        "[verticals] domains re-sync vertical=%s external_id=%s workspace=%s keys=%d domains=%d",
        vertical, request.external_id, workspace.id, len(keys), len(domains),
    )
    return {"updated_keys": len(keys), "domains": domains}


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
def gdpr_erase_subject(
    vertical: str,
    request: VerticalGdprEraseSubjectRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_vertical_internal_key),
) -> Dict[str, Any]:
    """Erase a single data subject within the workspace resolved from ``external_id``.

    Backs Shopify ``customers/redact``. Delegates to
    ``gdpr_service.erase_data_subject`` (audited; stores lacking a subject tag are
    reported in ``gaps``). 404 if no active workspace matches the external id.
    """
    workspace = _resolve_workspace_by_external_id(db, vertical, request.external_id)

    from services.gdpr_service import erase_data_subject

    try:
        result = erase_data_subject(
            db,
            workspace_id=workspace.id,
            subject_id=request.subject_id,
            requested_by=_webhook_actor(vertical, request.external_id),
        )
    except Exception:
        # CodeQL py/stack-trace-exposure: never echo the service's own exception
        # text back to this webhook caller; it is already logged with its trace.
        logger.exception(
            "[PRD-183 GDPR] erase-subject failed vertical=%s external_id=%s workspace=%s",
            vertical, request.external_id, workspace.id,
        )
        raise HTTPException(status_code=500, detail="Subject erasure failed") from None
    logger.info(
        "[PRD-183 GDPR] erase-subject vertical=%s external_id=%s workspace=%s subject=%s",
        vertical, request.external_id, workspace.id, request.subject_id,
    )
    return result


@router.post("/{vertical}/gdpr/erase")
def gdpr_erase_workspace(
    vertical: str,
    request: VerticalGdprEraseRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_vertical_internal_key),
) -> Dict[str, Any]:
    """Erase the whole workspace resolved from ``external_id`` (irreversible cascade).

    Backs Shopify ``shop/redact`` (fired 48h after uninstall). The confirmation
    echo the user-facing endpoint requires is unnecessary here: the caller cannot
    name a workspace id at all — it is resolved server-side from the shop domain,
    and a non-matching shop 404s rather than erasing anything. Delegates to
    ``gdpr_service.erase_workspace`` (audited). #1442: the uninstalled shop's
    workspaces are found though no longer active (``_workspaces_to_erase``), and an
    incomplete erasure answers 503 so the retry finishes it.
    """
    from api.gdpr import erasure_answer
    from services.gdpr_service import erase_workspace

    workspaces = _workspaces_to_erase(db, vertical, request.external_id)
    actor = _webhook_actor(vertical, request.external_id)
    try:
        results = [erase_workspace(db, workspace.id, requested_by=actor) for workspace in workspaces]
    except Exception:
        # CodeQL py/stack-trace-exposure: see gdpr_erase_subject above. A 500 is retried too.
        logger.exception(
            "[PRD-183 GDPR] erase-workspace failed vertical=%s external_id=%s workspaces=%s",
            vertical, request.external_id, [str(w.id) for w in workspaces],
        )
        raise HTTPException(status_code=500, detail="Workspace erasure failed") from None
    logger.info(
        "[PRD-183 GDPR] erase-workspace vertical=%s external_id=%s workspaces=%s complete=%s",
        vertical, request.external_id, [str(w.id) for w in workspaces], [r.get("complete") for r in results],
    )
    return erasure_answer(results)  # #1442: an incomplete erasure is a 503 the caller retries


@router.get("/{vertical}/gdpr/export")
def gdpr_export_workspace(
    vertical: str,
    external_id: str,
    customer_id: Optional[str] = None,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_vertical_internal_key),
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
