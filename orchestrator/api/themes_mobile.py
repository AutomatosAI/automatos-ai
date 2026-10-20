"""PRD-261 M-04 — ``GET /api/themes/mobile/{install_id}``: an install's theme as data for the phone.

The answer (``services/themes/mobile.py``, plan §2.2.11) is the member's theme colours, its font
files and its mascot files, as data and file URLs only: the phone never loads a theme's JS or CSS.

Authorised as ``GET /api/themes/boot/{install_id}`` (``api/themes_files.py``): the install's
workspace comes from the install row, and the caller must belong to it (``membership.py``), so
``ctx.workspace_id`` is never read and an ``X-Workspace-ID`` naming another workspace changes
nothing. An unknown, malformed or foreign install id is the boot route's one 404. Callers are a
Clerk-signed-in person and the local edition's operator; any other caller (an API key, SaaS's
anonymous fallback, a paired phone's ``device`` caller, which theme membership does not know yet)
belongs to no workspace and gets the 404.

The file URLs are served by ``GET /api/themes/files/...`` under the same membership, read through
the same reader dependency. ``Cache-Control: private, no-store``: the answer follows the member's
look. A plain ``def``: the database and the object store are synchronous.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from api.themes_files import BOOT_HEADERS, BOOT_NOT_FOUND, STORAGE_DOWN, get_theme_file_reader
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from services.themes.membership import caller_membership
from services.themes.mobile import mobile_theme
from services.themes.serve import FileReader, ThemeStorageUnavailable, boot_target

router = APIRouter(prefix="/api/themes/mobile", tags=["Themes"])

# The boot route's headers and refusals, so a phone and a browser are told the same.
MOBILE_HEADERS = BOOT_HEADERS
MOBILE_NOT_FOUND = BOOT_NOT_FOUND


def _install_uuid(install_id: str) -> Optional[UUID]:
    try:
        return UUID(install_id)
    except ValueError:
        return None


@router.get("/{install_id}")
def get_mobile_theme(
    install_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Any = Depends(get_db),
    reader: FileReader = Depends(get_theme_file_reader),
) -> JSONResponse:
    """The install's theme as data for the phone; ``mobile`` is null when it has none or may not run."""
    parsed = _install_uuid(install_id)
    membership = caller_membership(db, ctx)
    target = boot_target(db, membership, parsed) if parsed is not None else None
    if target is None:
        raise HTTPException(status_code=404, detail=MOBILE_NOT_FOUND, headers=MOBILE_HEADERS)
    try:
        answer = mobile_theme(db, membership, target, reader)
    except ThemeStorageUnavailable as exc:
        raise HTTPException(status_code=503, detail=STORAGE_DOWN, headers=MOBILE_HEADERS) from exc
    return JSONResponse(content=answer, headers=MOBILE_HEADERS)


__all__ = ["router"]
