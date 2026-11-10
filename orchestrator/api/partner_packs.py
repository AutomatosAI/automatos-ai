"""Partner packs: the read-only tools a workspace installs on a partner's own API (PRD-275).

``GET /api/partner-packs`` lists the workspace's packs. ``PUT /api/partner-packs/{name}``
installs a pack, or replaces the one of that name, with the credential its calls use.
``DELETE /api/partner-packs/{name}`` removes it. All three are for the workspace's managers
(``workspace:manage``), as creating the credential is, and all three answer 404 while
``PARTNER_TOOLS_ENABLED`` is off. Plain ``def`` routes: FastAPI runs them in its threadpool,
off the event loop.

Installing a pack offers it to no agent. An agent is offered a pack's tools when its
``configuration.partner_packs`` lists the pack (``PUT /api/agents/{id}``).
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from modules.tools.partner.offer import partner_tools_enabled, taken_names
from modules.tools.partner.pack import PackError, full_tool_name, parse_pack
from modules.tools.partner.store import InstalledPack, install_pack, installed_packs, remove_pack

router = APIRouter(prefix="/api/partner-packs", tags=["partner-packs"])

OFF = "Partner tools are off on this server (PARTNER_TOOLS_ENABLED)."
NOT_FOUND, UNPROCESSABLE = 404, 422


class InstallRequest(BaseModel):
    """A pack, and the id of the workspace's ``generic_api`` credential its calls use."""

    pack: Dict[str, Any]
    credential_id: int


def _require_on() -> None:
    if not partner_tools_enabled():
        raise HTTPException(status_code=NOT_FOUND, detail=OFF)


def _described(installed: InstalledPack) -> Dict[str, Any]:
    pack = installed.pack
    return {
        "name": pack.name, "title": pack.title, "version": pack.version, "description": pack.description,
        "credential_id": installed.credential_id,
        "tools": [{"name": full_tool_name(pack.name, tool), "description": tool.description, "scope": tool.scope,
                   "operations": [operation.name for operation in tool.operations]} for tool in pack.tools],
    }


@router.get("", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
def list_partner_packs(ctx: RequestContext = Depends(get_request_context_hybrid),
                       db: Session = Depends(get_db)) -> Dict[str, Any]:
    """The packs this workspace has installed, and the tools each one gives an agent that lists it."""
    _require_on()
    return {"packs": [_described(installed) for installed in installed_packs(db, ctx.workspace_id)]}


@router.put("/{name}", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
def install_partner_pack(name: str, body: InstallRequest,
                         ctx: RequestContext = Depends(get_request_context_hybrid),
                         db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Install the pack, or replace the one of its name. 422 says what a pack must change."""
    _require_on()
    try:
        pack = parse_pack(body.pack)
        if pack.name != name:
            raise PackError(f"The pack names itself '{pack.name}'; install it at /api/partner-packs/{pack.name}.")
        clashes = taken_names(db, pack)
        if clashes:
            raise PackError(f"These tool names are already tools of the platform's own: {clashes}.")
        return _described(install_pack(db, ctx.workspace_id, pack, body.credential_id))
    except PackError as refused:
        raise HTTPException(status_code=UNPROCESSABLE, detail=str(refused)) from None


@router.delete("/{name}", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
def remove_partner_pack(name: str, ctx: RequestContext = Depends(get_request_context_hybrid),
                        db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Remove the pack: its tools are offered to no agent from the next turn."""
    _require_on()
    if not remove_pack(db, ctx.workspace_id, name):
        raise HTTPException(status_code=NOT_FOUND, detail="No pack of that name is installed in this workspace.")
    return {"removed": name}
